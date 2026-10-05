"""Read GPU capabilities outside the GUI process and emit a JSON snapshot."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fizgig.utils.capabilities import detect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-kernels", action="store_true",
                        help="Explicitly test GPU kernels (disabled by default).")
    args = parser.parse_args()
    print(json.dumps(asdict(detect(probe_kernels=args.probe_kernels))))


if __name__ == "__main__":
    main()
