"""Static guards for the read-only notebook contract.

The notebooks may read raw run artifacts and export derived figures, but they must never write
inside ``results/``: those directories hold immutable script-generated artifacts. Static
analysis is enough here and keeps the guard cheap.
"""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((REPO_ROOT / "notebooks").glob("*.py"))
WRITE_METHODS = {"savefig", "write_text", "to_csv", "to_parquet", "mkdir"}


def test_notebooks_do_not_write_under_results() -> None:
    """No notebook write call targets results/, and figure exports root at analysis/."""
    assert NOTEBOOKS, "expected notebook sources"
    for path in NOTEBOOKS:
        source = path.read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "figures_dir":
                        segment = (ast.get_source_segment(source, node.value) or "").strip()
                        assert segment.startswith('Path("analysis")'), (
                            f"{path.name}: figures_dir must be rooted at analysis/, got {segment!r}"
                        )
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in WRITE_METHODS
            ):
                segment = ast.get_source_segment(source, node) or ""
                assert "results" not in segment, (
                    f"{path.name}: write call references results/: {segment}"
                )


def test_analysis_directory_is_gitignored() -> None:
    """Notebook figure exports must not be committed accidentally."""
    ignored = (REPO_ROOT / ".gitignore").read_text()
    assert "analysis/" in ignored
