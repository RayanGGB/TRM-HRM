#!/usr/bin/env python3
"""Move heavy run artifacts out of results_main before pushing to GitHub."""

from __future__ import annotations

import shutil
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    results_root = repo_root / "results_main"
    non_git_root = repo_root / "artifacts" / "results_main_non_git"

    if not results_root.exists():
        print("No results_main folder found. Nothing to do.")
        return

    heavy_dirs = []
    for p in results_root.rglob("*"):
        if p.is_dir() and p.name in {"checkpoints", "tensorboard"}:
            heavy_dirs.append(p)

    moved_dirs = 0
    for src in sorted(heavy_dirs, key=lambda x: len(x.parts), reverse=True):
        rel = src.relative_to(results_root)
        dst = non_git_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            for child in src.iterdir():
                target = dst / child.name
                if target.exists():
                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                shutil.move(str(child), str(target))
            src.rmdir()
        else:
            shutil.move(str(src), str(dst))
        moved_dirs += 1

    moved_pkls = 0
    for src in list(results_root.rglob("*.pkl")):
        rel = src.relative_to(results_root)
        dst = non_git_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        moved_pkls += 1

    print(f"Moved heavy directories: {moved_dirs}")
    print(f"Moved .pkl files: {moved_pkls}")


if __name__ == "__main__":
    main()
