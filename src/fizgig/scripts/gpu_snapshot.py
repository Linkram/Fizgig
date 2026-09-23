"""One-shot GUI launch metadata; never run matrix kernels or load bitsandbytes."""
import json
import argparse
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

if __name__ == "__main__":
    from fizgig.utils.capabilities import detect
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-kernels", action="store_true",
                        help="Only for an explicitly selected experimental precision")
    args = parser.parse_args()
    print(json.dumps(asdict(detect(probe_kernels=args.probe_kernels))), flush=True)
