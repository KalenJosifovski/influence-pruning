"""Attribution scoring, aggregation, and synthetic deletion-orientation tests.

Tests that need the upstream explainer are skipped unless ``tree-influence`` is importable;
it is installed only in the isolated Pixi ``influence`` environment.
"""

import builtins
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from pytest import raises

from influence_pruning.attribution import (
    BENEFICIAL_FIRST,
    SCORE_COLUMNS,
    BoostInAttributor,
    aggregate_scores,
    require_boostin,
)
from influence_pruning.config import ModelSettings
from influence_pruning.errors import InfluenceDependencyError
from influence_pruning.model import fit_predict, fit_regressor
from influence_pruning.partitions import LabelledPartition


def _influence_available() -> bool:
    """Return whether the isolated influence dependency is importable."""
    try:
        require_boostin()
    except InfluenceDependencyError:
        return False
    return True


requires_influence = pytest.mark.skipif(
    not _influence_available(),
    reason="tree-influence is installed only in the isolated influence environment",
)


def test_missing_influence_dependency_is_reported(monkeypatch) -> None:
    """A missing explainer fails with the required environment named in the message."""
    real_import = builtins.__import__

    def blocked(name: str, *args: object, **kwargs: object):
        if name.startswith("tree_influence"):
            raise ModuleNotFoundError("blocked for the dependency test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with raises(InfluenceDependencyError, match="pixi run -e influence"):
        require_boostin()


def _per_seed_scores() -> pd.DataFrame:
    """Fabricate a per-seed score frame in the shape produced by scoring."""
    rows: list[dict[str, object]] = []
    candidate_values = {
        "biogen:a:KEYA": (0.5, 1.5),
        "biogen:b:KEYB": (0.5, 1.5),  # ties with a in both seeds
        "polaris:c:KEYC": (0.1, 0.2),
        "expansionrx:d:KEYD": (0.9, 0.3),
    }
    for seed, offset in enumerate((0.0, 1.0)):
        for candidate, (first, second) in candidate_values.items():
            source = candidate.split(":")[0]
            rows.append(
                {
                    "source": source,
                    "origin_class": "native" if source == "expansionrx" else "donor",
                    "candidate_id": candidate,
                    "original_id": candidate.split(":")[1],
                    "inchikey": candidate.split(":")[2],
                    "model_target": 1.0,
                    "model_seed": seed,
                    "raw_score": (first if seed == 0 else second) + offset,
                    "within_seed_rank": -1,
                    "train_prediction": 1.0,
                    "train_residual": 0.0,
                    "n_train": 4,
                    "n_selection": 2,
                }
            )
    return pd.DataFrame(rows)


def test_aggregate_scores_produce_unique_stable_ranks() -> None:
    """Ranks are unique per seed, tie-broken by candidate ID, and stable across runs."""
    scores = _per_seed_scores()
    first = aggregate_scores(scores)
    second = aggregate_scores(scores)
    assert list(first.columns) == list(SCORE_COLUMNS)
    for seed in (0, 1):
        ranks = first.loc[first["model_seed"].eq(seed), "within_seed_rank"]
        assert sorted(ranks.tolist()) == [1, 2, 3, 4]
    tied = first.drop_duplicates("candidate_id").set_index("candidate_id")
    seed_zero = (
        first.loc[first["model_seed"].eq(0)].set_index("candidate_id")["within_seed_rank"].to_dict()
    )
    # a and b tie in seed 0; a wins the tie because its candidate ID sorts first.
    assert seed_zero["biogen:a:KEYA"] < seed_zero["biogen:b:KEYB"]
    aggregate = tied["aggregate_rank"].to_dict()
    assert sorted(aggregate.values()) == [1, 2, 3, 4]
    assert aggregate == (
        second.drop_duplicates("candidate_id").set_index("candidate_id")["aggregate_rank"].to_dict()
    )
    assert bool(tied["beneficial_first"].all())
    assert bool((tied["score_direction"] == "positive_is_beneficial").all())


def test_aggregate_rejects_double_aggregation() -> None:
    """Aggregating an aggregated frame is rejected."""
    once = aggregate_scores(_per_seed_scores())
    with raises(Exception, match="already aggregated"):
        aggregate_scores(once)


@dataclass(frozen=True)
class _Toy:
    """A synthetic pooled toy problem for the sign calibration."""

    training: LabelledPartition
    selection: LabelledPartition
    evaluation_x: np.ndarray
    evaluation_y: np.ndarray
    candidate_positions: np.ndarray
    n_helpful: int
    n_harmful: int
    settings: ModelSettings


def _toy(seed: int = 0, n_bits: int = 64) -> _Toy:
    """Build a pooled toy with helpful and deliberately harmful candidates."""
    rng = np.random.default_rng(seed)
    settings = ModelSettings(n_estimators=50, max_depth=4, n_jobs=1)

    def draw(count: int) -> tuple[np.ndarray, np.ndarray]:
        features = (rng.random((count, n_bits)) < 0.1).astype(np.uint8)
        signal = features[:, :8].sum(axis=1) * 0.5
        return features, (signal + 0.1 * rng.standard_normal(count)).astype(np.float32)

    base_x, base_y = draw(120)
    selection_x, selection_y = draw(40)
    evaluation_x, evaluation_y = draw(60)
    helpful_x, helpful_y = draw(12)
    harmful_x, harmful_y = draw(12)
    candidate_x = np.vstack([helpful_x, harmful_x])
    candidate_y = np.concatenate([helpful_y, harmful_y + 4.0])
    train_x = np.vstack([base_x, candidate_x])
    train_y = np.concatenate([base_y, candidate_y])

    def frame(x: np.ndarray, y: np.ndarray, prefix: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "candidate_id": [
                    f"toy:{prefix}{index}:KEY{prefix}{index}" for index in range(len(y))
                ],
                "source": "toy",
                "origin_class": "native",
                "original_id": [f"{prefix}{index}" for index in range(len(y))],
                "inchikey": [f"KEY{prefix}{index}" for index in range(len(y))],
                "model_target": y.astype(float),
                "fingerprint": [row for row in x],
            }
        )

    training_frame = frame(train_x, train_y, "t")
    selection_frame = frame(selection_x, selection_y, "s")
    return _Toy(
        training=LabelledPartition(
            name="full_training", frame=training_frame, fingerprints=train_x
        ),
        selection=LabelledPartition(
            name="influence_selection", frame=selection_frame, fingerprints=selection_x
        ),
        evaluation_x=evaluation_x,
        evaluation_y=evaluation_y,
        candidate_positions=np.arange(len(base_x), len(train_x)),
        n_helpful=len(helpful_x),
        n_harmful=len(harmful_x),
        settings=settings,
    )


@requires_influence
def test_scoring_covers_every_training_row_once_per_seed() -> None:
    """Every training row has one score per seed and one aggregate rank."""
    toy = _toy()
    scores = BoostInAttributor().score(toy.training, toy.selection, toy.settings, (0, 1))
    assert list(scores.columns) == list(SCORE_COLUMNS)
    per_seed = scores.groupby("model_seed")["candidate_id"].nunique()
    assert set(per_seed.tolist()) == {toy.training.size}
    aggregate = scores.drop_duplicates("candidate_id")
    assert aggregate["aggregate_rank"].nunique() == toy.training.size
    assert bool(np.isfinite(scores["raw_score"]).all())
    assert bool((scores["beneficial_first"] == BENEFICIAL_FIRST).all())


@requires_influence
def test_synthetic_deletion_orientation_holds() -> None:
    """Conditional deletion validation: low-score deletion helps more than high-score deletion.

    The full pooled model is scored against the selection surface, a high-score batch and a
    low-score batch are deleted from that same pool, and both pruned models are evaluated on a
    third held-out surface. Checks: harmful shifted-label observations receive lower pooled
    influence than helpful ones, and the low-score deletion improves held-out RMSE by more than
    the high-score deletion. This is deletion-only validation; addition utility is never
    measured, and a high-score batch is a deletion arm, not an addition policy.
    """
    toy = _toy()
    scores = BoostInAttributor().score(toy.training, toy.selection, toy.settings, (0, 1))
    aggregate = scores.drop_duplicates("candidate_id")
    toy_index = aggregate["original_id"].astype(str).str.removeprefix("t").astype(int)
    candidates = aggregate.loc[toy_index >= toy.candidate_positions.min()].copy()
    candidates["relative_index"] = toy_index.loc[candidates.index] - toy.candidate_positions.min()
    candidates = candidates.sort_values("aggregate_rank", kind="stable")
    helpful_mean = float(
        candidates.loc[candidates["relative_index"] < toy.n_helpful, "raw_score_mean"].mean()
    )
    harmful_mean = float(
        candidates.loc[candidates["relative_index"] >= toy.n_helpful, "raw_score_mean"].mean()
    )
    assert helpful_mean > harmful_mean

    baseline = float(
        np.sqrt(
            np.mean(
                np.square(
                    toy.evaluation_y
                    - fit_predict(
                        toy.training.fingerprints,
                        toy.training.frame["model_target"].to_numpy(dtype=float),
                        toy.evaluation_x,
                        toy.settings,
                        0,
                    )
                )
            )
        )
    )
    high_ids = set(candidates.head(toy.n_helpful)["candidate_id"])
    low_ids = set(candidates.tail(toy.n_harmful)["candidate_id"])
    assert not (high_ids & low_ids)

    def rmse_without(removed: set[str]) -> float:
        keep = ~toy.training.frame["candidate_id"].isin(removed).to_numpy()
        return float(
            np.sqrt(
                np.mean(
                    np.square(
                        toy.evaluation_y
                        - fit_predict(
                            toy.training.fingerprints[keep],
                            toy.training.frame.loc[keep, "model_target"].to_numpy(dtype=float),
                            toy.evaluation_x,
                            toy.settings,
                            0,
                        )
                    )
                )
            )
        )

    delta_low = baseline - rmse_without(low_ids)
    delta_high = baseline - rmse_without(high_ids)
    assert delta_low >= delta_high


def test_addition_utility_calibration_is_removed() -> None:
    """No addition-utility calibration remains; deletion is the only calibration story."""
    import influence_pruning.attribution as attribution_module

    assert not hasattr(attribution_module, "measure_sign_calibration")
    assert not hasattr(attribution_module, "rmse")


def test_explicit_base_score_does_not_change_predictions() -> None:
    """The influence model and the outcome model are the same pooled model.

    The score model is constructed with an explicit ``base_score`` because the tree-influence
    parser requires a float bias, while every outcome fit uses the default constructor. On a
    deterministic synthetic dataset the two construction paths must produce identical
    predictions; this locks the "same pooled model" interpretation against a future XGBoost
    release silently changing base-score handling.
    """
    rng = np.random.default_rng(0)
    features = (rng.random((180, 48)) < 0.15).astype(np.uint8)
    target = (features[:, :6].sum(axis=1) * 0.4 + 0.05 * rng.standard_normal(180)).astype(
        np.float32
    )
    settings = ModelSettings(n_estimators=25, max_depth=4, n_jobs=1)

    default_model = fit_regressor(features, target, settings, 0)
    influence_model = fit_regressor(features, target, settings, 0, require_explicit_base_score=True)
    default_predictions = default_model.predict(features.astype(np.float32))
    influence_predictions = influence_model.predict(features.astype(np.float32))
    np.testing.assert_allclose(default_predictions, influence_predictions, rtol=1e-6, atol=1e-6)
