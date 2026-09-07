"""Export a training checkpoint as a provenance-checked Diffusers student bundle."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.student_export import export_student


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--checkpoint", required=True, help="Checkpoint file, checkpoint directory, or training output root")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_steps", type=int)
    parser.add_argument("--student_architecture", help="Exact student Diffusers config directory if its training-time location moved")
    args = parser.parse_args()
    print(json.dumps(export_student(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
