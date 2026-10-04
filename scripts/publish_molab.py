"""Publish the notebook and data bundle to GitHub for molab syncing.

molab has no upload API; its supported automation is GitHub mirroring. This script

1. checks GitHub authentication and creates the repository if it does not exist,
2. points the notebook's ``MOLAB_BUNDLE_URL`` at a release asset,
3. builds ``dist/molab_bundle.zip`` and uploads it to the ``molab-bundle`` release,
4. commits and pushes the notebook and scripts.

After the one-time manual step (molab -> New notebook -> Mirror from GitHub), every
``git push`` updates the molab notebook automatically.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
NOTEBOOK = REPO_ROOT / "notebooks" / "04_pooling_surgery.py"
URL_LINE = re.compile(
    r'^([ \t]*)MOLAB_BUNDLE_URL = (?:\([^)]*\)|"[^"]*")',
    re.MULTILINE | re.DOTALL,
)
RELEASE_TAG = "molab-bundle"
ASSET_NAME = "molab_bundle.zip"
DRY_RUN = False


def run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess:
    """Run a command from the repository root, honouring ``--dry-run``."""
    if DRY_RUN:
        print("dry-run:", " ".join(command))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
    return subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=capture,
    )


def probe(command: list[str]) -> int:
    """Run a read-only command and return its exit code without raising."""
    if DRY_RUN:
        print("dry-run:", " ".join(command))
        return 1
    return subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True).returncode


def set_bundle_url(url: str) -> bool:
    """Rewrite the notebook's bundle URL placeholder; return True when it changed."""

    def replacement(match: re.Match) -> str:
        indent = match.group(1)
        prefix, _, asset = url.partition("/releases/latest/download/")
        return (
            f"{indent}MOLAB_BUNDLE_URL = (\n"
            f'{indent}    "{prefix}/"\n'
            f'{indent}    "releases/latest/download/{asset}"\n'
            f"{indent})"
        )

    source = NOTEBOOK.read_text()
    updated, replacements = URL_LINE.subn(replacement, source, count=1)
    if replacements != 1:
        raise SystemExit("could not find the MOLAB_BUNDLE_URL placeholder in the notebook")
    if updated == source:
        return False
    NOTEBOOK.write_text(updated)
    return True


def main() -> None:
    global DRY_RUN

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="GitHub repository as owner/name")
    parser.add_argument("--private", action="store_true", help="create a private repository")
    parser.add_argument("--skip-push", action="store_true", help="prepare locally, never push")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = parser.parse_args()
    DRY_RUN = args.dry_run

    bundle_url = f"https://github.com/{args.repo}/releases/latest/download/{ASSET_NAME}"
    if args.dry_run:
        print("repository:      ", args.repo)
        print("bundle asset URL:", bundle_url)
        print("placeholder found:", bool(URL_LINE.search(NOTEBOOK.read_text())))
        print("would push, rebuild the bundle, create/update the release, and print the")
        print("one-time molab mirror step")
        return

    run(["gh", "auth", "status"], capture=True)

    if probe(["gh", "repo", "view", args.repo]) != 0:
        visibility = "--private" if args.private else "--public"
        run(
            [
                "gh",
                "repo",
                "create",
                args.repo,
                visibility,
                "--source",
                ".",
                "--remote",
                "origin",
            ]
        )
    if "origin" not in run(["git", "remote"], capture=True).stdout.split():
        run(["git", "remote", "add", "origin", f"https://github.com/{args.repo}.git"])

    if not args.skip_push:
        run(["git", "push", "-u", "origin", "HEAD"])

    changed = set_bundle_url(bundle_url)

    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from build_molab_bundle import build_zip

    archive = build_zip()
    print(f"built {archive.relative_to(REPO_ROOT)}")

    if args.skip_push:
        print("skip-push: repository and bundle prepared locally only")
        return

    if changed:
        run(["git", "add", str(NOTEBOOK.relative_to(REPO_ROOT))])
        run(["git", "commit", "-m", "chore(molab): point bundle download at the GitHub release"])
    run(["git", "push", "origin", "HEAD"])

    existing = subprocess.run(
        ["gh", "release", "view", RELEASE_TAG, "--repo", args.repo],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if existing.returncode == 0:
        run(
            [
                "gh",
                "release",
                "upload",
                RELEASE_TAG,
                str(archive),
                "--clobber",
                "--repo",
                args.repo,
            ]
        )
    else:
        run(
            [
                "gh",
                "release",
                "create",
                RELEASE_TAG,
                str(archive),
                "--repo",
                args.repo,
                "--title",
                "molab data bundle",
                "--notes",
                "Self-extracting data bundle for notebooks/04_pooling_surgery.py.",
            ]
        )

    branch = run(["git", "branch", "--show-current"], capture=True).stdout.strip()
    print()
    print("one-time molab step:")
    print("  molab -> New notebook -> Mirror from GitHub -> paste:")
    print(f"  https://github.com/{args.repo}/blob/{branch}/notebooks/04_pooling_surgery.py")
    print()
    print("after that, every `git push` updates the molab notebook automatically.")


if __name__ == "__main__":
    main()
