from __future__ import annotations

import unittest

from cuda_compat import (
    minimum_cuda_issue,
    inspect_torch_cuda,
    profile_cuda_device,
    resolve_attention_config,
    select_attention_backend,
    validate_lightx2v_quant_backend,
)


class CudaCompatibilityTests(unittest.TestCase):
    def test_target_data_center_profiles(self) -> None:
        cases = {
            (8, 0): ("ampere", "11.0", False, False),
            (9, 0): ("hopper", "11.8", True, False),
            (10, 0): ("blackwell-datacenter", "12.8", True, True),
            (10, 3): ("blackwell-datacenter", "12.9", True, True),
        }
        for capability, expected in cases.items():
            profile = profile_cuda_device("test", capability)
            self.assertEqual(
                (profile.architecture, profile.minimum_cuda, profile.supports_fp8, profile.supports_fp4),
                expected,
            )

    def test_ada_profile_preserves_sage2_4090_configs(self) -> None:
        ada = profile_cuda_device("RTX 4090", (8, 9))

        self.assertEqual(ada.architecture, "ada")
        self.assertEqual(
            select_attention_backend("sage_attn2", ada, {"sage_attn2", "torch_sdpa"}),
            "sage_attn2",
        )

    def test_h20_uses_its_product_specific_cuda_floor(self) -> None:
        h20 = profile_cuda_device("NVIDIA H20", (9, 0))
        h20_3e = profile_cuda_device("NVIDIA H20-3e", (9, 0))
        h100 = profile_cuda_device("NVIDIA H100", (9, 0))
        h200 = profile_cuda_device("NVIDIA H200", (9, 0))

        self.assertEqual(h20.minimum_cuda, "12.2")
        self.assertEqual(h20_3e.minimum_cuda, "12.2")
        self.assertEqual(h100.minimum_cuda, "11.8")
        self.assertEqual(h200.minimum_cuda, "11.8")
        self.assertIsNotNone(minimum_cuda_issue(h20, "12.1"))
        self.assertIsNone(minimum_cuda_issue(h20, "12.2"))
        self.assertEqual(
            select_attention_backend(
                "auto",
                h20,
                {"flash_attn2", "flash_attn3", "torch_sdpa"},
            ),
            "flash_attn2",
        )
        self.assertEqual(
            select_attention_backend(
                "flash_attn3",
                h20,
                {"flash_attn3", "torch_sdpa"},
            ),
            "torch_sdpa",
        )

    def test_attention_backend_falls_back_by_architecture(self) -> None:
        a100 = profile_cuda_device("A100", (8, 0))
        h20 = profile_cuda_device("H20", (9, 0))
        b300 = profile_cuda_device("B300", (10, 3))

        self.assertEqual(select_attention_backend("auto", a100, {"flash_attn2", "torch_sdpa"}), "flash_attn2")
        self.assertEqual(select_attention_backend("auto", h20, {"flash_attn2", "torch_sdpa"}), "flash_attn2")
        self.assertEqual(select_attention_backend("auto", b300, {"sage_attn3", "torch_sdpa"}), "torch_sdpa")
        self.assertEqual(select_attention_backend("sage_attn3", b300, {"sage_attn3", "torch_sdpa"}), "torch_sdpa")
        with self.assertRaises(RuntimeError):
            select_attention_backend(
                "sage_attn3",
                b300,
                {"sage_attn3", "torch_sdpa"},
                strict=True,
            )
        self.assertEqual(select_attention_backend("sage_attn3", a100, {"torch_sdpa"}), "torch_sdpa")

    def test_strict_attention_selection_rejects_missing_kernel(self) -> None:
        profile = profile_cuda_device("A100", (8, 0))
        with self.assertRaises(RuntimeError):
            select_attention_backend("flash_attn3", profile, {"torch_sdpa"}, strict=True)

    def test_explicit_installed_backend_must_match_device_profile(self) -> None:
        profile = profile_cuda_device("A100", (8, 0))
        available = {"flash_attn3", "torch_sdpa"}

        self.assertEqual(
            select_attention_backend("flash_attn3", profile, available),
            "torch_sdpa",
        )
        with self.assertRaises(RuntimeError):
            select_attention_backend("flash_attn3", profile, available, strict=True)

    def test_attention_config_resolves_dense_fields_only(self) -> None:
        profile = profile_cuda_device("A100", (8, 0))
        config = {
            "attn_type": "flash_attn3",
            "self_attn_1_type": "auto",
            "parallel": {"seq_p_attn_type": "ulysses"},
        }

        changes = resolve_attention_config(
            config,
            profile=profile,
            available_backends={"flash_attn3", "torch_sdpa"},
        )

        self.assertEqual(config["attn_type"], "torch_sdpa")
        self.assertEqual(config["self_attn_1_type"], "torch_sdpa")
        self.assertEqual(config["parallel"]["seq_p_attn_type"], "ulysses")
        self.assertIn("attn_type", changes)

    def test_b300_requires_cuda_12_9(self) -> None:
        profile = profile_cuda_device("B300", (10, 3))
        self.assertIsNotNone(minimum_cuda_issue(profile, "12.8"))
        self.assertIsNone(minimum_cuda_issue(profile, "12.9"))

    def test_sm120_and_sm121_require_runtime_smoke_before_sage3(self) -> None:
        for capability in ((12, 0), (12, 1)):
            with self.subTest(capability=capability):
                profile = profile_cuda_device("Blackwell client GPU", capability)
                self.assertEqual(profile.attention_preference, ("torch_sdpa",))
                self.assertEqual(
                    select_attention_backend(
                        "auto",
                        profile,
                        {"sage_attn3", "flash_attn2", "flash_attn3", "torch_sdpa"},
                    ),
                    "torch_sdpa",
                )
                with self.assertRaises(RuntimeError):
                    select_attention_backend(
                        "sage_attn3",
                        profile,
                        {"sage_attn3", "torch_sdpa"},
                        strict=True,
                    )

    def test_every_unverified_capability_uses_only_native_sdpa(self) -> None:
        capabilities = (
            (7, 5),
            (8, 6),
            (9, 1),
            (10, 1),
            (10, 2),
            (10, 4),
            (11, 0),
            (11, 9),
            (12, 2),
            (13, 0),
        )
        for capability in capabilities:
            with self.subTest(capability=capability):
                profile = profile_cuda_device("unverified GPU", capability)
                self.assertEqual(profile.architecture, "unverified-cuda")
                self.assertEqual(profile.attention_preference, ("torch_sdpa",))
                self.assertEqual(
                    select_attention_backend(
                        "auto",
                        profile,
                        {"sage_attn3", "flash_attn3", "torch_sdpa"},
                    ),
                    "torch_sdpa",
                )

    def test_vendored_low_bit_extension_is_not_generic_blackwell(self) -> None:
        for capability in ((8, 0), (9, 0), (10, 0), (10, 3)):
            with self.subTest(capability=capability):
                with self.assertRaisesRegex(RuntimeError, "sm_120a"):
                    validate_lightx2v_quant_backend(
                        "nvfp4",
                        capability,
                        callables_available=True,
                    )

        with self.assertRaisesRegex(RuntimeError, "callable lightx2v_kernel"):
            validate_lightx2v_quant_backend(
                "mxfp4",
                (12, 0),
                callables_available=False,
            )
        validate_lightx2v_quant_backend(
            "mxfp4",
            (12, 0),
            callables_available=True,
        )
        validate_lightx2v_quant_backend(
            "fp8-triton",
            (10, 0),
            callables_available=False,
        )

    def test_strict_report_distinguishes_native_arch_from_ptx_fallback(self) -> None:
        class FakeCuda:
            @staticmethod
            def is_available():
                return True

            @staticmethod
            def get_arch_list():
                return ["sm_80", "sm_90"]

            @staticmethod
            def device_count():
                return 1

            @staticmethod
            def get_device_name(index):
                return "B200"

            @staticmethod
            def get_device_capability(index):
                return (10, 0)

        class FakeVersion:
            cuda = "12.8"

        class FakeTorch:
            __version__ = "test"
            version = FakeVersion()
            cuda = FakeCuda()

        report = inspect_torch_cuda(FakeTorch())

        self.assertFalse(report["devices"][0]["native_arch_compiled"])
        self.assertTrue(any("native sm_100" in issue for issue in report["issues"]))


if __name__ == "__main__":
    unittest.main()
