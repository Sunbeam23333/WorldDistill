"""Audited architecture plans and explicit native descriptor declarations.

Native paths are relative to ``NativeTrainingAdapter.model``. A declaration
describes the author's tensor layout; it does not infer a valid parallel plan
from arbitrary Linear modules. No architecture is advertised by name alone.
"""
from __future__ import annotations

import copy
import torch.distributed as dist
from torch import nn

from training.parallel.tensor_parallel import (
    apply_tensor_parallel_plan, apply_wan_tensor_parallel, mark_canonical_names,
)
from training.parallel.pipeline_parallel import (
    apply_pipeline_parallel_plan, apply_wan_pipeline_parallel,
)


def _builtin_native_plan(adapter):
    from training.native_adapters.world_models import MatrixGame2TrainingAdapter, LingBotCameraTrainingAdapter
    if isinstance(adapter, MatrixGame2TrainingAdapter):
        first = ["time_embedding", "time_projection", "img_emb"]
    elif isinstance(adapter, LingBotCameraTrainingAdapter):
        first = ["time_embedding", "time_projection", "text_embedding", "patch_embedding_wancamctrl",
                 "c2ws_hidden_states_layer1", "c2ws_hidden_states_layer2"]
    else:
        return None
    # Author forward reads patch_embedding.weight.device outside its call.
    # This small convolution stays explicitly replicated; transformer/action/
    # camera blocks and the embedding/output MLPs have actual stage ownership.
    return {"schema_version": 1, "pipeline": {"blocks": "blocks", "first_stage": first,
        "last_stage": ["head"], "gradient_checkpointing_attribute": "gradient_checkpointing"}}


def _native_declaration(adapter):
    declaration = adapter.descriptor.get("parallel_plan")
    if declaration is None:
        declaration = _builtin_native_plan(adapter)
    if not isinstance(declaration, dict) or declaration.get("schema_version") != 1:
        raise ValueError("Native TP/PP requires an explicit parallel_plan with schema_version=1")
    if set(declaration) - {"schema_version", "tensor", "pipeline"}:
        raise ValueError("Unknown native parallel_plan fields")
    return copy.deepcopy(declaration)


def _path(root, path):
    if not isinstance(path, str) or not path or any(not part for part in path.split(".")):
        raise ValueError("Parallel plan paths must be nonempty dotted module paths")
    parent, _, name = path.rpartition(".")
    return root.get_submodule(parent) if parent else root, name


def _native_tensor_plan(root, declaration, size):
    tensor = declaration.get("tensor")
    if not isinstance(tensor, dict) or set(tensor) - {"modules", "divide_attributes"}:
        raise ValueError("Native tensor parallelism requires tensor.modules and optional divide_attributes")
    modules = tensor.get("modules")
    if not isinstance(modules, dict) or not modules:
        raise ValueError("Native tensor.modules must be a nonempty path-to-layout object")
    attributes = tensor.get("divide_attributes", [])
    if not isinstance(attributes, list) or len(set(attributes)) != len(attributes):
        raise ValueError("Native tensor.divide_attributes must contain unique attribute paths")
    divisions = []
    for path in attributes:
        parent, name = _path(root, path)
        value = getattr(parent, name)
        if type(value) is not int or value < 1 or value % size:
            raise ValueError(f"Native TP attribute {path} must be a positive integer divisible by tp_size")
        divisions.append((parent, name, value // size))
    for path in modules:
        _path(root, path)
        root.get_submodule(path)
    return modules, divisions


def _native_pipeline_plan(root, declaration, size):
    pipeline = declaration.get("pipeline")
    fields = {"modules", "blocks", "first_stage", "last_stage", "gradient_checkpointing_attribute"}
    if not isinstance(pipeline, dict) or set(pipeline) - fields:
        raise ValueError("Native pipeline parallelism requires an explicit pipeline layout")
    if ("modules" in pipeline) == ("blocks" in pipeline):
        raise ValueError("Declare exactly one pipeline.modules map or pipeline.blocks sequence")
    checkpoint_paths, checkpoint_attribute = set(), None
    if "blocks" in pipeline:
        block_path = pipeline["blocks"]
        _path(root, block_path)
        blocks = root.get_submodule(block_path)
        if not isinstance(blocks, (nn.ModuleList, nn.Sequential)) or len(blocks) < size:
            raise ValueError("Native PP blocks must be a sequence with at least one block per stage")
        modules = {f"{block_path}.{i}": i * size // len(blocks) for i in range(len(blocks))}
        checkpoint_paths = set(modules)
        for field, owner in (("first_stage", 0), ("last_stage", size - 1)):
            paths = pipeline.get(field, [])
            if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
                raise ValueError(f"Native pipeline.{field} must contain module paths")
            for path in paths:
                if path in modules:
                    raise ValueError("Native PP assigns a module more than once")
                modules[path] = owner
    else:
        if "first_stage" in pipeline or "last_stage" in pipeline:
            raise ValueError("first_stage/last_stage require a pipeline.blocks declaration")
        modules = pipeline["modules"]
        if not isinstance(modules, dict) or not modules:
            raise ValueError("Native pipeline.modules must be a nonempty path-to-stage object")
        checkpoint_paths = set(modules)
    for path, owner in modules.items():
        _path(root, path)
        root.get_submodule(path)
        if type(owner) is not int or not 0 <= owner < size:
            raise ValueError(f"Native PP owner for {path} must be an integer stage in pp_size")
    attr = pipeline.get("gradient_checkpointing_attribute")
    if attr is not None:
        parent, name = _path(root, attr)
        value = getattr(parent, name, False)
        if type(value) is not bool:
            raise ValueError("Native gradient checkpointing attribute must be boolean")
        checkpoint_attribute = (parent, name, value)
    return modules, checkpoint_paths, checkpoint_attribute


def apply_model_parallel_plan(model, *, tp_group=None, pp_group=None):
    """Apply a built-in Wan plan or a validated native descriptor plan.

    Groups of size one are no-ops. TP precedes PP. All paths in checkpoint
    metadata remain the original full model paths, including adapter prefixes.
    """
    from training.native_model_adapter import NativeTrainingAdapter
    tp_size = dist.get_world_size(tp_group) if tp_group is not None else 1
    pp_size = dist.get_world_size(pp_group) if pp_group is not None else 1
    if tp_size == pp_size == 1:
        return model
    candidates = [(prefix, module) for prefix, module in model.named_modules()
                  if isinstance(module, NativeTrainingAdapter)]
    if not candidates:
        if tp_size > 1:
            apply_wan_tensor_parallel(model, tp_group)
        if pp_size > 1:
            apply_wan_pipeline_parallel(model, pp_group)
        return model
    if getattr(model, "_worlddistill_native_parallel_applied", False):
        raise ValueError("Native parallel plans cannot be applied twice")
    prepared = []
    for prefix, adapter in candidates:
        declaration = _native_declaration(adapter)
        tensor = _native_tensor_plan(adapter.model, declaration, tp_size) if tp_size > 1 else None
        pipeline = _native_pipeline_plan(adapter.model, declaration, pp_size) if pp_size > 1 else None
        prepared.append((f"{prefix}.model".lstrip("."), tensor, pipeline))
    mark_canonical_names(model)
    for prefix, tensor, pipeline in prepared:
        if tensor is not None:
            modules, divisions = tensor
            apply_tensor_parallel_plan(model, tp_group, {f"{prefix}.{path}": style for path, style in modules.items()})
            for parent, name, value in divisions:
                setattr(parent, name, value)
        if pipeline is not None:
            modules, checkpoint_paths, checkpoint_attribute = pipeline
            apply_pipeline_parallel_plan(model, pp_group, {f"{prefix}.{path}": owner for path, owner in modules.items()})
            if checkpoint_attribute is not None:
                parent, name, enabled = checkpoint_attribute
                setattr(parent, name, False)
                for path in checkpoint_paths:
                    model.get_submodule(f"{prefix}.{path}").activation_checkpointing = enabled
    model._worlddistill_native_parallel_applied = True
    return model
