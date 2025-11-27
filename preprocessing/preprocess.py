#!/usr/bin/env python3
import argparse
import logging

from evenet.control.global_config import global_config
from preprocessing.helper import preprocess  # the function we defined

def generate_assignment_names(event_info):
    assignment_names = []
    assignment_map = []

    for p, c in event_info.product_particles.items():
        for pp, dp in c.items():
            assignment_names.append(f"TARGETS/{p}/{pp}")
            assignment_map.append((p, pp, dp))

    return assignment_names, assignment_map


def parse_args():
    parser = argparse.ArgumentParser(
        description="EveNet Preprocessing Tool — convert NPZ → Parquet with optional splitting"
    )

    # ------------------------------------------------------------------
    # Input options
    # ------------------------------------------------------------------
    parser.add_argument("--files", nargs="*", default=None,
                        help="A list of NPZ files to preprocess (will be split by --split_ratio)")
    parser.add_argument("--file", type=str, default=None,
                        help="Single NPZ file (shortcut for --files <file>)")

    parser.add_argument("--train", nargs="*", default=None,
                        help="Explicit list of training NPZ files")
    parser.add_argument("--val", nargs="*", default=None,
                        help="Explicit list of validation NPZ files")
    parser.add_argument("--test", nargs="*", default=None,
                        help="Explicit list of test NPZ files")

    # ------------------------------------------------------------------
    # Split ratio
    # ------------------------------------------------------------------
    parser.add_argument(
        "--split_ratio", type=str, default="1,0,0",
        help="Train,Val,Test split ratios. Example: --split_ratio 0.8,0.1,0.1"
    )

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    parser.add_argument("--store_dir", type=str, required=True,
                        help="Output directory for Parquet + metadata")

    # ------------------------------------------------------------------
    # Global EveNet config (YAML)
    # ------------------------------------------------------------------
    parser.add_argument("--config", type=str, required=True,
                        help="Global EveNet config YAML")

    # Verbose
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable extra logging")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.DEBUG)

    # Load global EveNet config
    global_config.load_yaml(args.config)

    # Parse split ratios
    try:
        split_ratio = tuple(float(x) for x in args.split_ratio.split(","))
    except:
        raise ValueError("Invalid split_ratio format. Use comma-separated floats, e.g. 0.8,0.1,0.1")

    if len(split_ratio) != 3:
        raise ValueError("split_ratio must have exactly three values: train,val,test")

    # Resolve input sources
    # Priority: explicit train/val/test > files list > single file
    if args.train or args.val or args.test:
        train_files = args.train
        val_files = args.val
        test_files = args.test
        files = None
    else:
        if args.file:
            files = [args.file]
        else:
            files = args.files
        train_files = val_files = test_files = None

    k1, k2 = global_config.event_info.classification_names[0].split('/')
    process_ids = global_config.event_info.class_label[k1][k2][0]

    assignment_keys, assignment_key_map = generate_assignment_names(global_config.event_info)

    # Run preprocessing
    preprocess(
        files=files,
        train=train_files,
        val=val_files,
        test=test_files,
        split_ratio=split_ratio,
        store_dir=args.store_dir,
        global_config=global_config,
        unique_process_ids=process_ids,
        assignment_keys=assignment_keys,
        verbose=args.verbose,
    )

if __name__ == "__main__":
    main()
