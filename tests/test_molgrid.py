"""Chemistry-safe preparation tests for the custom molecule-grid widget."""

import pandas as pd

from influence_pruning.molgrid import (
    MoleculeGrid,
    draw_svg,
    molecule_records,
    scaffold_atoms,
    scaffold_of,
    scaffold_summary,
)


def _manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "role": ["full_training", "full_training", "full_training", "evaluation"],
            "candidate_id": ["c1", "c2", "c3", "e1"],
            "original_id": ["mol-1", "mol-2", "mol-3", "mol-4"],
            "source": ["alpha", "alpha", "beta", "alpha"],
            "canonical_smiles": [
                "c1ccccc1CC(=O)O",
                "C1CCCCC1CCO",
                "c1ccncc1CC(=O)N",
                "c1ccncc1",
            ],
            "model_target": [1.0, 2.0, float("nan"), 1.5],
        }
    )


def _scores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": ["c1", "c2", "c3"],
            "raw_score_mean": [0.01, -0.02, 0.005],
            "max_tanimoto_to_selection": [0.9, 0.4, 0.7],
        }
    )


def test_draw_svg_returns_inline_svg_without_xml_declaration() -> None:
    svg = draw_svg("c1ccccc1")
    assert svg.startswith("<svg")
    assert "<?xml" not in svg
    assert "<path" in svg or "<line" in svg


def test_draw_svg_handles_invalid_smiles_with_a_placeholder() -> None:
    svg = draw_svg("not-a-molecule")
    assert svg.startswith("<svg")
    assert "invalid structure" in svg


def test_scaffold_helpers_roundtrip() -> None:
    scaffold = scaffold_of("c1ccccc1CC(=O)O")
    assert scaffold
    atoms = scaffold_atoms("c1ccccc1CC(=O)O", scaffold)
    assert atoms
    assert scaffold_of("not-a-molecule") == ""
    assert scaffold_atoms("not-a-molecule", scaffold) == ()


def test_molecule_records_schema_cap_and_missing_values() -> None:
    records = molecule_records(_manifest(), _scores(), ["c1", "e1", "missing", "c1"])
    assert [record["id"] for record in records] == ["c1", "e1"]
    training, evaluation = records
    assert training["score"] == 0.01
    assert training["similarity"] == 0.9
    assert training["source"] == "alpha"
    assert evaluation["score"] is None
    assert evaluation["label"] == 1.5
    assert evaluation["scaffold"] == "c1ccncc1"
    assert evaluation["smiles"] == "c1ccncc1"
    assert all(record["svg"].startswith("<svg") for record in records)


def test_molecule_records_cap_and_highlight() -> None:
    records = molecule_records(
        _manifest(),
        _scores(),
        ["c1", "c2", "c3", "e1"],
        cap=2,
        highlight_scaffold="c1ccccc1",
    )
    assert [record["id"] for record in records] == ["c1", "c2"]
    assert "highlight" in records[0]["svg"] or "fill" in records[0]["svg"]


def test_scaffold_summary_ranks_both_directions() -> None:
    summary = scaffold_summary(_manifest(), _scores(), min_count=1, top=2)
    assert set(summary["direction"]) == {"Helpful", "Harmful"}
    assert summary.loc[summary["direction"].eq("Helpful"), "mean_score"].max() == 0.01
    assert summary.loc[summary["direction"].eq("Harmful"), "mean_score"].min() == -0.02
    assert {"scaffold", "n", "mean_score", "direction", "short"} <= set(summary.columns)


def test_molecule_grid_is_an_anywidget_with_synced_traits() -> None:
    grid = MoleculeGrid(molecules=[], title="Test")
    assert set(grid.traits(sync=True)) >= {"molecules", "selected", "title"}
    assert grid.selected == []
    assert grid.title == "Test"
