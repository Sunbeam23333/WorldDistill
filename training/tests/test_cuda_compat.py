from __future__ import annotations

import unittest

from cuda_compat import minimum_cuda_issue, profile_cuda_device, select_attention_backend


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

    def test_attention_backend_falls_back_by_architecture(self) -> None:
        a100 = profile_cuda_device("A100", (8, 0))
        h20 = profile_cuda_device("H20", (9, 0))
        b300 = profile_cuda_device("B300", (10, 3))

        self.assertEqual(select_attention_backend("auto", a100, {"flash_attn2", "torch_sdpa"}), "flash_attn2")
        self.assertEqual(select_attention_backend("auto", h20, {"flash_attn3", "torch_sdpa"}), "flash_attn3")
        self.assertEqual(select_attention_backend("auto", b300, {"sage_attn3", "torch_sdpa"}), "sage_attn3")
        self.assertEqual(select_attention_backend("sage_attn3", a100, {"torch_sdpa"}), "torch_sdpa")

    def test_strict_attention_selection_rejects_missing_kernel(self) -> None:
        profile = profile_cuda_device("A100", (8, 0))
        with self.assertRaises(RuntimeError):
            select_attention_backend("flash_attn3", profile, {"torch_sdpa"}, strict=True)

    def test_b300_requires_cuda_12_9(self) -> None:
        profile = profile_cuda_device("B300", (10, 3))
        self.assertIsNotNone(minimum_cuda_issue(profile, "12.8"))
        self.assertIsNone(minimum_cuda_issue(profile, "12.9"))


if __name__ == "__main__":
    unittest.main()
