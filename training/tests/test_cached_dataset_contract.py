from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from training.data.video_dataset import CachedLatentDataset, VideoDataset


class CachedLatentDatasetContractTests(unittest.TestCase):
    def _write_manifest(self, root: Path, item: dict) -> Path:
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps([item]), encoding="utf-8")
        return manifest

    def test_missing_latent_fails_during_manifest_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = self._write_manifest(
                root,
                {"sample_id": "clip-0001", "latent_path": "latents/missing.pt"},
            )
            with self.assertRaisesRegex(FileNotFoundError, r"clip-0001.*missing\.pt"):
                CachedLatentDataset(str(manifest), cache_dir=str(root))

    def test_corrupt_latent_fails_instead_of_returning_zeros(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            corrupt_path = root / "corrupt.pt"
            corrupt_path.write_text("not a torch checkpoint", encoding="utf-8")
            manifest = self._write_manifest(
                root,
                {"sample_id": "clip-corrupt", "latent_path": corrupt_path.name},
            )
            dataset = CachedLatentDataset(str(manifest), cache_dir=str(root))
            with self.assertRaisesRegex(RuntimeError, r"clip-corrupt.*corrupt\.pt"):
                dataset[0]

    def test_valid_latent_preserves_its_real_shape_and_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            expected = torch.arange(30, dtype=torch.float32).reshape(2, 3, 5, 1)
            latent_path = root / "latent.pt"
            torch.save(expected, latent_path)
            manifest = self._write_manifest(
                root,
                {"sample_id": "clip-valid", "latent_path": latent_path.name},
            )
            sample = CachedLatentDataset(str(manifest), cache_dir=str(root))[0]
            self.assertTrue(torch.equal(sample["latents"], expected))
            self.assertEqual(sample["num_frames"], 3)


class RawConditioningContractTests(unittest.TestCase):
    def _dataset(self, root: Path) -> VideoDataset:
        manifest = root / "manifest.json"
        manifest.write_text("[]", encoding="utf-8")
        return VideoDataset(str(manifest), video_dir=str(root))

    def test_declared_missing_conditioning_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = self._dataset(root)
            with self.assertRaisesRegex(FileNotFoundError, r"camera.*missing\.pt"):
                dataset._load_optional_tensor("missing.pt", "camera")

    def test_declared_corrupt_conditioning_fails_fast(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            corrupt = root / "corrupt.pt"
            corrupt.write_text("not a checkpoint", encoding="utf-8")
            dataset = self._dataset(root)
            with self.assertRaisesRegex(RuntimeError, r"action.*corrupt\.pt"):
                dataset._load_optional_tensor(corrupt.name, "action")


if __name__ == "__main__":
    unittest.main()
