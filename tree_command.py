import os
import argparse

def limited_tree(root, max_files=5, prefix="", is_root=True):
    entries = sorted(os.listdir(root))
    files = [e for e in entries if os.path.isfile(os.path.join(root, e))]
    dirs  = [e for e in entries if os.path.isdir(os.path.join(root, e))]

    if is_root:
        print(root)

    for i, f in enumerate(files[:max_files]):
        connector = "└── " if i == len(files[:max_files]) - 1 and not dirs else "├── "
        print(prefix + connector + f)

    for i, d in enumerate(dirs):
        connector = "└── " if i == len(dirs) - 1 else "├── "
        print(prefix + connector + d)
        new_prefix = prefix + ("    " if i == len(dirs) - 1 else "│   ")
        limited_tree(os.path.join(root, d), max_files, new_prefix, is_root=False)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Print a limited tree view of a directory."
    )
    # Positional root path with default; also allow --root for clarity
    parser.add_argument(
        "root",
        nargs="?",
        default="data",
        help="Root directory to start from (default: data)",
    )
    parser.add_argument(
        "--root",
        dest="root_opt",
        help="Root directory to start from (overrides positional)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=5,
        help="Max number of files to show per directory (default: 5)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root = args.root_opt or args.root
    if not os.path.exists(root):
        raise SystemExit(f"Error: root path not found: {root}")
    if not os.path.isdir(root):
        raise SystemExit(f"Error: root path is not a directory: {root}")
    limited_tree(root, max_files=args.max_files)
