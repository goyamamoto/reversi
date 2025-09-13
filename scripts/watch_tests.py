#!/usr/bin/env python3
import argparse
import os
import sys
import time
import subprocess
from pathlib import Path


def list_py_files(root: Path, tests: Path, extra: list[Path] | None = None) -> list[Path]:
    paths: list[Path] = []
    include_dirs = [root, tests]
    if extra:
        include_dirs.extend(extra)
    for base in include_dirs:
        if not base.exists():
            continue
        for p in base.rglob("*.py"):
            # skip virtualenvs and caches
            if "/venv/" in p.as_posix() or p.name.startswith(".") or "__pycache__" in p.as_posix():
                continue
            paths.append(p)
    # de-duplicate
    return sorted(set(paths))


def snapshot_mtimes(files: list[Path]) -> dict[Path, float]:
    return {p: p.stat().st_mtime for p in files if p.exists()}


def has_changes(prev: dict[Path, float], files: list[Path]) -> bool:
    now = snapshot_mtimes(files)
    if prev.keys() != now.keys():
        return True
    for p, t in now.items():
        if prev.get(p) != t:
            return True
    return False


def run_tests(tests: str, pattern: str) -> int:
    # Use pytest to run tests; it will auto-discover under tests/
    cmd = [sys.executable, "-m", "pytest"]
    print(f"\n[watch] Running: {' '.join(cmd)}")
    res = subprocess.run(cmd)
    print(f"[watch] Exit: {res.returncode}")
    return res.returncode


def main():
    ap = argparse.ArgumentParser(description="Watch Python files and re-run unittest on changes")
    ap.add_argument("--tests", default="tests", help="Tests directory")
    ap.add_argument("--pattern", default="test_*.py", help="Test filename pattern")
    ap.add_argument("--interval", type=float, default=0.5, help="Polling interval seconds")
    args = ap.parse_args()

    root = Path.cwd()
    tests = root / args.tests
    files = list_py_files(root, tests)
    mtimes = snapshot_mtimes(files)

    print("[watch] Watching .py files. Press Ctrl+C to stop.")
    # run immediately
    run_tests(args.tests, args.pattern)

    try:
        while True:
            time.sleep(args.interval)
            files = list_py_files(root, tests)
            if has_changes(mtimes, files):
                mtimes = snapshot_mtimes(files)
                run_tests(args.tests, args.pattern)
    except KeyboardInterrupt:
        print("\n[watch] Stopped.")


if __name__ == "__main__":
    main()
