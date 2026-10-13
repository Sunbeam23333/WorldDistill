import copy
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from training.model_adapter import load_diffusers_training_model
from training.student_export import _resolve_inference_steps, export_student, file_sha256, load_student_pipeline, read_student_manifest


def tiny_wan():
    diffusers = pytest.importorskip("diffusers")
    return diffusers.WanTransformer3DModel(num_attention_heads=2, attention_head_dim=8, in_channels=2,
                                         out_channels=2, text_dim=16, freq_dim=8, ffn_dim=32, num_layers=1)


@pytest.fixture
def bundle(tmp_path):
    base = tmp_path / "base"
    teacher = tiny_wan()
    teacher.save_pretrained(base / "transformer")
    (base / "model_index.json").write_text(json.dumps({"_class_name": "WanPipeline", "transformer": ["diffusers", "WanTransformer3DModel"]}))
    student = load_diffusers_training_model(base)
    with torch.no_grad():
        for parameter in student.parameters():
            parameter.add_(0.125)
    checkpoint = tmp_path / "trainer_state.pt"
    torch.save({"student_model": student.state_dict(), "step": 7, "args": {"distill_method": "step_distill", "denoising_step_list": [1000, 750, 500, 250]}}, checkpoint)
    destination = tmp_path / "student"
    export_student(str(base), str(checkpoint), str(destination))
    return base, checkpoint, destination, student, teacher


def test_every_exported_parameter_is_the_actual_student(bundle):
    _, _, destination, student, teacher = bundle
    restored = load_diffusers_training_model(destination)
    for name, expected in student.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], expected, rtol=0, atol=0)
    assert not torch.equal(next(restored.parameters()), next(teacher.parameters()))
    manifest = read_student_manifest(destination)
    assert "transformer/config.json" in manifest["artifacts"]
    assert manifest["global_step"] == 7 and manifest["num_inference_steps"] == 4


def test_pipeline_cannot_ignore_student_overrides(bundle):
    diffusers = pytest.importorskip("diffusers")
    _, _, destination, _, teacher = bundle
    with patch.object(diffusers.DiffusionPipeline, "from_pretrained", return_value=SimpleNamespace(transformer=teacher)):
        with pytest.raises(RuntimeError, match="ignored the trained student"):
            load_student_pipeline(str(destination))


def test_pipeline_receives_exact_exported_weights(bundle):
    diffusers = pytest.importorskip("diffusers")
    _, _, destination, student, _ = bundle
    def construct(source, **kwargs):
        return SimpleNamespace(transformer=kwargs["transformer"])
    with patch.object(diffusers.DiffusionPipeline, "from_pretrained", side_effect=construct):
        pipeline, manifest = load_student_pipeline(str(destination))
    for name, parameter in pipeline.transformer.state_dict().items():
        torch.testing.assert_close(parameter, student.model.state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["config", "extra_weight", "empty_weights"])
def test_incomplete_or_modified_bundle_is_rejected(bundle, kind):
    _, _, destination, _, _ = bundle
    if kind == "config":
        path = destination / "transformer/config.json"
        path.write_text(path.read_text() + "\n")
    elif kind == "extra_weight":
        (destination / "transformer/extra.safetensors").write_bytes(b"not exported")
    else:
        path = destination / "worlddistill_export.json"
        manifest = json.loads(path.read_text())
        manifest["weights"] = {}
        path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="integrity|inventory"):
        load_diffusers_training_model(destination)


def test_wrong_base_pipeline_cannot_substitute_teacher_weights(bundle, tmp_path):
    _, _, destination, _, _ = bundle
    wrong_base = tmp_path / "wrong"
    wrong_base.mkdir()
    (wrong_base / "model_index.json").write_text(json.dumps({"_class_name": "StableDiffusionPipeline", "unet": ["diffusers", "UNet2DConditionModel"]}))
    with pytest.raises(ValueError, match="pipeline class differs"):
        load_student_pipeline(str(destination), base_model=str(wrong_base))


@pytest.mark.parametrize("steps", [0, -1, 2.5, True])
def test_invalid_step_count_is_not_silently_replaced(steps, tmp_path):
    with pytest.raises(ValueError, match="positive integer"):
        export_student("unused", "unused", str(tmp_path / "student"), num_steps=steps)


@pytest.mark.parametrize("unsupported", ["use_lora", "use_dual_model"])
def test_unmerged_lora_and_legacy_dual_fail_before_export(bundle, unsupported, tmp_path):
    base, checkpoint, _, _, _ = bundle
    state = torch.load(checkpoint, weights_only=False)
    state["args"][unsupported] = True
    torch.save(state, checkpoint)
    destination = tmp_path / "unsupported"
    with pytest.raises(ValueError, match="explicit merge"):
        export_student(str(base), str(checkpoint), str(destination))
    assert not destination.exists()


@pytest.mark.parametrize("meanflow", [False, True])
def test_lora_export_merges_actual_student_and_meanflow_time_conditioner(tmp_path, meanflow):
    pytest.importorskip("peft")
    from training.trainer_args import TrainerArgs
    from training.train_distill import _apply_lora
    from training.utils.meanflow import MeanFlowTimeAdapter
    base = tmp_path / "base"
    tiny_wan().save_pretrained(base / "transformer")
    (base / "model_index.json").write_text(json.dumps({"_class_name": "WanPipeline",
        "transformer": ["diffusers", "WanTransformer3DModel"]}))
    student = load_diffusers_training_model(base)
    if meanflow:
        student = MeanFlowTimeAdapter(student, latent_channels=2, hidden_dim=12)
    args = TrainerArgs(distill_method="mean_flow_distill" if meanflow else "step_distill",
        use_lora=True, lora_rank=2, lora_alpha=4., lora_target_modules=["to_q", "to_k"],
        meanflow_latent_channels=2, meanflow_hidden_dim=12)
    student = _apply_lora(student, args).eval()
    with torch.no_grad():
        for name, parameter in student.named_parameters():
            if "lora_B" in name or (meanflow and "modules_to_save.default" in name):
                parameter.add_(0.05)
    inputs = {"hidden_states": torch.randn(1, 2, 2, 2, 2), "timestep": torch.tensor([700.]),
              "encoder_hidden_states": torch.randn(1, 3, 16)}
    if meanflow:
        inputs["timestep_r"] = torch.tensor([300.])
    expected = student(**inputs)
    checkpoint = tmp_path / "lora.pt"
    torch.save({"student_model": student.state_dict(), "step": 4, "args": vars(args)}, checkpoint)
    destination = tmp_path / "merged"
    manifest = export_student(str(base), str(checkpoint), str(destination), num_steps=2)
    restored = load_diffusers_training_model(destination).eval()
    torch.testing.assert_close(restored(**inputs), expected, atol=2e-5, rtol=2e-5)
    assert manifest["lora_merged"]
    assert not any("lora_" in name or "modules_to_save" in name for name in restored.state_dict())


@pytest.mark.parametrize("lora,compiled", [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("layout", ["legacy", "unified_v1"])
def test_dual_export_preserves_both_branches_and_shifted_native_boundary(tmp_path, lora, compiled, layout):
    from training.model_adapter import NoiseRoutedDenoiser
    from training.trainer_args import TrainerArgs
    from training.train_distill import _apply_lora
    if lora:
        pytest.importorskip("peft")
    base = tmp_path / "base"
    tiny_wan().save_pretrained(base / "transformer")
    (base / "model_index.json").write_text(json.dumps({"_class_name": "WanPipeline",
        "transformer": ["diffusers", "WanTransformer3DModel"]}))
    (base / "scheduler").mkdir()
    (base / "scheduler" / "scheduler_config.json").write_text(json.dumps({"num_train_timesteps": 20}))
    args = TrainerArgs(distill_method="step_distill", use_dual_model=True, use_lora=lora,
        lora_rank=2, lora_alpha=4., lora_target_modules=["to_q", "to_k"], sample_shift=3.,
        boundary_step_index=2, enable_torch_compile=compiled, torch_compile_scope="student")
    if layout == "unified_v1":
        args.dual_student_layout = layout
    high = load_diffusers_training_model(base)
    if lora:
        high = _apply_lora(high, args)
    low = copy.deepcopy(high)
    with torch.no_grad():
        for branch, delta in ((high, .03), (low, -.07)):
            for name, parameter in branch.named_parameters():
                if not lora or "lora_B" in name:
                    parameter.add_(delta)
    high.eval(); low.eval()
    inputs = {"hidden_states": torch.randn(2, 2, 2, 2, 2), "timestep": torch.tensor([16., 8.]),
              "encoder_hidden_states": torch.randn(2, 3, 16)}
    expected = NoiseRoutedDenoiser(high, low, .75, 20)(**inputs)
    checkpoint = tmp_path / "trainer_state.pt"
    saved_high = torch.compile(high, backend="eager") if compiled else high
    saved_low = torch.compile(low, backend="eager") if compiled else low
    saved = NoiseRoutedDenoiser(saved_high, saved_low, .75, 20) if layout == "unified_v1" else saved_high
    torch.save({"student_model": saved.state_dict(), "step": 4, "args": vars(args)}, checkpoint)
    if layout == "legacy":
        torch.save(saved_low.state_dict(), tmp_path / "student_low.pt")
    else:
        torch.save({"dual_student_layout": "unified_v1", "num_train_timesteps": 20,
                    "distill_timesteps": [20., 18., 15., 10.], "boundary_step_index": 2},
                   tmp_path / "dual_student_state.pt")
    destination = tmp_path / "dual"
    manifest = export_student(str(base), str(checkpoint), str(destination))
    restored = load_diffusers_training_model(destination).eval()
    torch.testing.assert_close(restored(**inputs), expected, atol=2e-5, rtol=2e-5)
    assert manifest["legacy_dual_consolidated"] == (layout == "legacy")
    assert manifest["requires_adapter_sampling"]
    assert manifest["low_checkpoint_sha256"] == (file_sha256(tmp_path / "student_low.pt") if layout == "legacy" else None)
    assert manifest["boundary_ratio"] == pytest.approx(.75)
    assert manifest["trained_sampling_timesteps"] == pytest.approx([20, 18, 15, 10])
    assert manifest["components"] == ["transformer", "transformer_2"]
    assert not any("lora_" in name for name in restored.state_dict())
    if layout == "unified_v1":
        assert manifest["dual_routing_state_sha256"] == file_sha256(tmp_path / "dual_student_state.pt")
        routing = torch.load(tmp_path / "dual_student_state.pt", weights_only=True)
        routing["boundary_step_index"] = 1
        torch.save(routing, tmp_path / "dual_student_state.pt")
        with pytest.raises(ValueError, match="routing state disagrees"):
            export_student(str(base), str(checkpoint), str(tmp_path / "invalid-routing"))


def test_vp_step_times_match_integer_scheduler_and_native_default_rescaling():
    from training.student_export import _step_distill_timesteps
    model = SimpleNamespace(num_train_timesteps=20, training_noise_process="vp")
    config = {"denoising_step_list": [1000, 750, 500, 250], "sample_shift": 1}
    assert _step_distill_timesteps(config, model) == [19, 15, 10, 5]
    with pytest.raises(ValueError, match="VP"):
        _step_distill_timesteps({**config, "sample_shift": 3}, model)


@pytest.mark.parametrize("lora", [False, True])
def test_native_dual_wan_meanflow_export_retains_nested_router_and_endpoint_adapter(tmp_path, lora):
    from training.model_adapter import NoiseRoutedDenoiser
    from training.trainer_args import TrainerArgs
    from training.train_distill import _apply_lora
    from training.utils.meanflow import MeanFlowTimeAdapter
    if lora:
        pytest.importorskip("peft")
    base = tmp_path / "base_dual"
    tiny_wan().save_pretrained(base / "transformer")
    tiny_wan().save_pretrained(base / "transformer_2")
    (base / "model_index.json").write_text(json.dumps({"_class_name": "WanPipeline", "boundary_ratio": .6,
        "transformer": ["diffusers", "WanTransformer3DModel"],
        "transformer_2": ["diffusers", "WanTransformer3DModel"]}))
    args = TrainerArgs(distill_method="mean_flow_distill", use_lora=lora,
        lora_rank=2, lora_alpha=4., lora_target_modules=["to_q", "to_k"],
        meanflow_latent_channels=2, meanflow_hidden_dim=12)
    student = MeanFlowTimeAdapter(load_diffusers_training_model(base), latent_channels=2, hidden_dim=12)
    if lora:
        student = _apply_lora(student, args)
    with torch.no_grad():
        for name, parameter in student.named_parameters():
            if "lora_B" in name or "time_conditioner" in name:
                parameter.add_(.03)
    student.eval()
    inputs = {"hidden_states": torch.randn(2, 2, 2, 2, 2), "timestep": torch.tensor([700., 300.]),
              "timestep_r": torch.tensor([600., 200.]), "encoder_hidden_states": torch.randn(2, 3, 16)}
    expected = student(**inputs)
    checkpoint = tmp_path / "trainer_state.pt"
    torch.save({"student_model": student.state_dict(), "step": 4, "args": vars(args)}, checkpoint)
    output = tmp_path / "meanflow_dual"
    export_student(str(base), str(checkpoint), str(output), num_steps=2)
    restored = load_diffusers_training_model(output).eval()
    assert isinstance(restored, MeanFlowTimeAdapter) and isinstance(restored.base_model, NoiseRoutedDenoiser)
    assert restored.base_model.boundary_ratio == .6
    torch.testing.assert_close(restored(**inputs), expected, atol=2e-5, rtol=2e-5)


def test_explicit_teacher_file_is_not_replaced_by_default_weights(tmp_path):
    from training.train_distill import _load_models
    teacher = tiny_wan()
    teacher.save_pretrained(tmp_path)
    custom = copy.deepcopy(teacher.state_dict())
    for value in custom.values():
        if value.is_floating_point():
            value.add_(0.25)
    checkpoint = tmp_path / "chosen_teacher.pt"
    torch.save(custom, checkpoint)
    loaded, student = _load_models(str(checkpoint), "", "wan2.1", "", torch.device("cpu"), torch.float32,
                                   student_device=torch.device("cpu"))
    for name, expected in custom.items():
        torch.testing.assert_close(loaded.model.state_dict()[name], expected, rtol=0, atol=0)
        torch.testing.assert_close(student.model.state_dict()[name], expected, rtol=0, atol=0)
    assert not any(parameter.requires_grad for parameter in loaded.parameters())
    assert all(parameter.requires_grad for parameter in student.parameters())


def test_export_uses_distinct_student_architecture_not_teacher_config(tmp_path):
    diffusers = pytest.importorskip("diffusers")
    base = tmp_path / "teacher"
    teacher = tiny_wan()
    teacher.save_pretrained(base)
    student_dir = tmp_path / "student-init"
    student_model = diffusers.WanTransformer3DModel(num_attention_heads=2, attention_head_dim=8, in_channels=2,
        out_channels=2, text_dim=16, freq_dim=8, ffn_dim=32, num_layers=2)
    student_model.save_pretrained(student_dir)
    student = load_diffusers_training_model(student_dir)
    checkpoint = tmp_path / "trained.pt"
    torch.save({"student_model": student.state_dict(), "args": {"student_model_path": str(student_dir)}, "step": 1}, checkpoint)
    destination = tmp_path / "export"
    export_student(str(base), str(checkpoint), str(destination), num_steps=4)
    restored = load_diffusers_training_model(destination)
    assert len(restored.model.blocks) == 2 and len(teacher.blocks) == 1
    for name, expected in student.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], expected, rtol=0, atol=0)


def test_declared_compiled_student_exports_eager_exact_weights(bundle, tmp_path):
    base, checkpoint, _, student, _ = bundle
    compiled = torch.compile(student, backend="eager")
    state = torch.load(checkpoint, weights_only=False)
    state["student_model"] = compiled.state_dict()
    assert all(name.startswith("_orig_mod.") for name in state["student_model"])
    state["args"].update(enable_torch_compile=True, torch_compile_scope="student")
    torch.save(state, checkpoint)
    destination = tmp_path / "compiled-export"
    export_student(str(base), str(checkpoint), str(destination))
    restored = load_diffusers_training_model(destination)
    for name, expected in student.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], expected, rtol=0, atol=0)


@pytest.mark.parametrize("invalid", ["undeclared", "mixed", "teacher_only"])
def test_compile_prefix_is_not_stripped_without_exact_contract(bundle, tmp_path, invalid):
    base, checkpoint, _, student, _ = bundle
    state = torch.load(checkpoint, weights_only=False)
    state["student_model"] = {"_orig_mod." + name: value for name, value in student.state_dict().items()}
    if invalid != "undeclared":
        state["args"].update(enable_torch_compile=True, torch_compile_scope="teacher" if invalid == "teacher_only" else "student")
    if invalid == "mixed":
        key = next(iter(state["student_model"]))
        state["student_model"][key.removeprefix("_orig_mod.")] = state["student_model"].pop(key)
    torch.save(state, checkpoint)
    destination = tmp_path / "invalid-compiled-export"
    with pytest.raises(RuntimeError, match="Missing key|Unexpected key"):
        export_student(str(base), str(checkpoint), str(destination))
    assert not destination.exists()


@pytest.mark.parametrize("method,key,count", [
    ("dmd_distill", "dmd_student_steps", 1),
    ("dmd_distill", "dmd_student_steps", 3),
    ("stream_distill", "denoising_steps_per_frame", 2),
])
def test_export_count_uses_algorithm_specific_training_setting(bundle, tmp_path, method, key, count):
    base, checkpoint, _, _, _ = bundle
    state = torch.load(checkpoint, weights_only=False)
    state["args"].update(distill_method=method, **{key: count})
    torch.save(state, checkpoint)
    manifest = export_student(str(base), str(checkpoint), str(tmp_path / "algorithm-export"))
    assert manifest["num_inference_steps"] == count
    assert manifest["inference_steps_provenance"]["source"] == f"training_args.{key}"
    assert manifest["training_sampling_config"][key] == count
    assert manifest["sampling_parity"]["enforced"] is False


@pytest.mark.parametrize("args", [
    {}, {"distill_method": "unknown"},
    {"distill_method": "consistency_distill"},
    {"distill_method": "adversarial_distill"},
    {"distill_method": "context_forcing"},
    {"distill_method": "step_distill"},
    {"distill_method": "step_distill", "denoising_step_list": []},
    {"distill_method": "step_distill", "denoising_step_list": [True]},
    {"distill_method": "dmd_distill"},
    {"distill_method": "dmd_distill", "dmd_student_steps": 0},
    {"distill_method": "stream_distill", "denoising_steps_per_frame": -2},
])
def test_ambiguous_algorithm_count_requires_explicit_override(args, tmp_path):
    source = tmp_path / "trainer_state.pt"
    with pytest.raises(ValueError, match="explicit --num_steps"):
        _resolve_inference_steps(args, {}, source, None)
    assert _resolve_inference_steps(args, {}, source, 6) == (6, {"source": "explicit_num_steps"})


def test_ambiguous_export_leaves_no_partial_bundle(bundle, tmp_path):
    base, checkpoint, _, _, _ = bundle
    state = torch.load(checkpoint, weights_only=False)
    state["args"]["distill_method"] = "consistency_distill"
    torch.save(state, checkpoint)
    destination = tmp_path / "ambiguous-export"
    with pytest.raises(ValueError, match="explicit --num_steps"):
        export_student(str(base), str(checkpoint), str(destination))
    assert not destination.exists()


def test_explicit_steps_keep_legacy_missing_schedule_export_without_replay_claim(bundle, tmp_path):
    base, checkpoint, _, _, _ = bundle
    state = torch.load(checkpoint, weights_only=False)
    state["args"].pop("denoising_step_list")
    torch.save(state, checkpoint)
    manifest = export_student(str(base), str(checkpoint), str(tmp_path / "legacy-count"), num_steps=3)
    assert manifest["num_inference_steps"] == 3
    assert manifest["trained_sampling_timesteps"] is None


def progressive_checkpoint(tmp_path, stage=0, global_step=7):
    args = {"distill_method": "progressive_distill", "progressive_stages": [64, 32, 16, 8, 4],
            "progressive_stage_steps": 10, "denoising_step_list": [1000, 750, 500, 250]}
    source = tmp_path / "trainer_state.pt"
    progressive = {"current_stage": stage, "current_teacher_steps": args["progressive_stages"][stage],
                   "current_student_steps": args["progressive_stages"][stage + 1]}
    torch.save(progressive, tmp_path / "progressive_state.pt")
    return args, {"step": global_step}, source


@pytest.mark.parametrize("stage,global_step,count", [(0, 7, 32), (1, 11, 16), (2, 24, 8), (3, 31, 4), (3, 40, 4)])
def test_progressive_export_uses_trained_checkpoint_stage_not_final_stage(tmp_path, stage, global_step, count):
    args, state, source = progressive_checkpoint(tmp_path, stage, global_step)
    steps, provenance = _resolve_inference_steps(args, state, source, None)
    assert steps == count
    assert provenance["current_stage"] == stage
    assert provenance["stage_optimizer_steps"] == global_step - stage * 10
    assert provenance["progressive_state_sha256"] == file_sha256(tmp_path / "progressive_state.pt")


@pytest.mark.parametrize("stage,global_step", [(0, 0), (1, 10), (2, 20), (3, 30), (2, 11), (0, 11)])
def test_untrained_or_inconsistent_progressive_stage_requires_explicit_steps(tmp_path, stage, global_step):
    args, state, source = progressive_checkpoint(tmp_path, stage, global_step)
    with pytest.raises(ValueError, match="explicit --num_steps"):
        _resolve_inference_steps(args, state, source, None)
    assert _resolve_inference_steps(args, state, source, 8)[0] == 8


@pytest.mark.parametrize("bad", ["missing_state", "wrong_stage", "wrong_count", "missing_duration", "invalid_schedule"])
def test_progressive_export_rejects_missing_or_mismatched_stage_evidence(tmp_path, bad):
    args, state, source = progressive_checkpoint(tmp_path)
    path = tmp_path / "progressive_state.pt"
    if bad == "missing_state":
        path.unlink()
    elif bad == "missing_duration":
        args.pop("progressive_stage_steps")
    elif bad == "invalid_schedule":
        args["progressive_stages"] = [64, 32, 12, 4]
    else:
        record = torch.load(path, weights_only=True)
        record["current_stage" if bad == "wrong_stage" else "current_student_steps"] = 999
        torch.save(record, path)
    with pytest.raises(ValueError, match="explicit --num_steps"):
        _resolve_inference_steps(args, state, source, None)
