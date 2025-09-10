import os

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

if __name__ == "__main__":
    limited_tree("data", max_files=5)
