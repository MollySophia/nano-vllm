#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    where = f" (cwd={cwd})" if cwd is not None else ""
    print(f"+ {' '.join(cmd)}{where}", flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def build_prepare_cmd(args: argparse.Namespace, repo_root: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "prepare_test_env.py"),
        "--repo-root",
        str(repo_root),
    ]
    if args.skip_albatross:
        cmd.append("--skip-albatross")
    if args.albatross_url is not None:
        cmd.extend(["--albatross-url", args.albatross_url])
    if args.albatross_dest is not None:
        cmd.extend(["--albatross-dest", args.albatross_dest])
    if args.albatross_ref is not None:
        cmd.extend(["--albatross-ref", args.albatross_ref])
    return cmd


def build_unittest_cmd(args: argparse.Namespace) -> list[str]:
    if args.tests:
        return [sys.executable, "-m", "unittest", *args.tests]
    return [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        args.discover_start_dir,
        "-p",
        args.discover_pattern,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the local test environment and run nano-vllm unit tests."
        )
    )
    parser.add_argument(
        "--repo-root",
        default=Path(__file__).resolve().parents[1],
        type=Path,
        help="nano-vllm repository root. Defaults to this script's parent repo.",
    )
    parser.add_argument(
        "--skip-prepare-test-env",
        action="store_true",
        help="Skip the prepare_test_env.py step and run tests directly.",
    )
    parser.add_argument(
        "--skip-albatross",
        action="store_true",
        help="Pass through to prepare_test_env.py to skip preparing Albatross.",
    )
    parser.add_argument(
        "--albatross-url",
        default=None,
        help="Override the Albatross Git URL passed to prepare_test_env.py.",
    )
    parser.add_argument(
        "--albatross-dest",
        default=None,
        help="Override the Albatross checkout path passed to prepare_test_env.py.",
    )
    parser.add_argument(
        "--albatross-ref",
        default=None,
        help="Override the Albatross ref passed to prepare_test_env.py.",
    )
    parser.add_argument(
        "--discover-start-dir",
        default="tests",
        help="Start directory for unittest discovery. Ignored when --tests is set.",
    )
    parser.add_argument(
        "--discover-pattern",
        default="test_*.py",
        help="Filename pattern for unittest discovery. Ignored when --tests is set.",
    )
    parser.add_argument(
        "tests",
        nargs="*",
        help=(
            "Optional unittest modules/files to run directly. "
            "Defaults to unittest discovery."
        ),
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    if not repo_root.exists():
        raise SystemExit(f"repo root does not exist: {repo_root}")

    if not args.skip_prepare_test_env:
        run(build_prepare_cmd(args, repo_root), cwd=repo_root)

    run(build_unittest_cmd(args), cwd=repo_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
