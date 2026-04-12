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


def ensure_git_repo(dest: Path) -> None:
    if not (dest / ".git").exists():
        raise SystemExit(f"{dest} exists but is not a git repository.")


def prepare_albatross_repo(dest: Path, repo_url: str | None, ref: str | None) -> None:
    if dest.exists():
        ensure_git_repo(dest)
        if repo_url is not None:
            run(["git", "remote", "set-url", "origin", repo_url], cwd=dest)
        run(["git", "fetch", "--all", "--tags"], cwd=dest)
    else:
        if repo_url is None:
            raise SystemExit(
                f"{dest} does not exist. Pass --albatross-url to specify the repository URL."
            )
        clone_cmd = ["git", "clone"]
        if ref is not None:
            clone_cmd.extend(["--branch", ref, "--single-branch"])
        clone_cmd.extend([repo_url, str(dest)])
        run(clone_cmd)

    if ref is not None:
        run(["git", "checkout", ref], cwd=dest)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the local test environment for nano-vllm. "
            "Clone or refresh the reference Albatross repository used by comparison tests."
        )
    )
    parser.add_argument(
        "--repo-root",
        default=Path(__file__).resolve().parents[1],
        type=Path,
        help="nano-vllm repository root. Defaults to this script's parent repo.",
    )
    parser.add_argument(
        "--skip-albatross",
        action="store_true",
        help="Skip preparing the Albatross checkout.",
    )
    parser.add_argument(
        "--albatross-url",
        default="https://github.com/Triang-jyed-driung/Albatross",
        help="Git URL for the Albatross repository.",
    )
    parser.add_argument(
        "--albatross-dest",
        default="Albatross",
        help="Relative path under --repo-root for the Albatross checkout.",
    )
    parser.add_argument(
        "--albatross-ref",
        default="fp16",
        help="Git ref to check out after clone/fetch. Defaults to the fp16 branch.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    if not repo_root.exists():
        raise SystemExit(f"repo root does not exist: {repo_root}")

    print(f"repo root: {repo_root}", flush=True)
    if not args.skip_albatross:
        dest = (repo_root / args.albatross_dest).resolve()
        prepare_albatross_repo(dest, args.albatross_url, args.albatross_ref)
        print(f"prepared Albatross at {dest}", flush=True)
    else:
        print("skipped Albatross preparation", flush=True)

    print("test environment preparation complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
