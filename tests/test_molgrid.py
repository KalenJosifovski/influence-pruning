"""Chemistry-safe preparation tests for the custom molecule-grid widget."""

import pandas as pd

from influence_pruning.molgrid import (
    MoleculeGrid,
    ScaffoldBars,
    Speaker,
    draw_svg,
    molecule_records,
    scaffold_atoms,
    scaffold_of,
    scaffold_summary,
    selection_ids,
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


def test_scaffold_bars_is_an_anywidget_with_synced_traits() -> None:
    bars = ScaffoldBars(scaffolds=[], selected="c1ccccc1", title="Test")
    assert set(bars.traits(sync=True)) >= {"scaffolds", "selected", "title"}
    assert bars.selected == "c1ccccc1"


def test_speaker_is_an_anywidget_with_synced_traits() -> None:
    speaker = Speaker(text="Hello", label="Listen")
    assert set(speaker.traits(sync=True)) >= {"text", "label"}
    assert speaker.text == "Hello"
    assert speaker.label == "Listen"


def test_selection_ids_resolves_every_marimo_payload_shape() -> None:
    lookup = {0: ["a", "b"], 1: ["c"]}
    # Box/lasso extraction: list of points with only curve/point coordinates.
    assert selection_ids([{"curveNumber": 0, "pointIndex": 1}], lookup) == ["b"]
    # Dict wrapper from older selection payloads.
    assert selection_ids({"points": [{"customdata": ["c", 1.0]}]}, lookup) == ["c"]
    # Click payloads: named field, list customdata, dict customdata, and pointNumber.
    assert selection_ids([{"candidate_id": "z"}], lookup) == ["z"]
    assert selection_ids([{"customdata": {"candidate_id": "q"}}], lookup) == ["q"]
    assert selection_ids(
        [{"curveNumber": 0, "pointNumber": 0}, {"curveNumber": 1, "pointIndex": 0}],
        lookup,
    ) == ["a", "c"]
    # Empty and malformed payloads never raise.
    assert selection_ids([], lookup) == []
    assert selection_ids(None, lookup) == []
    assert selection_ids([{"curveNumber": 9, "pointIndex": 3}], lookup) == []
