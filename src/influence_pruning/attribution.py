"""Pooled BoostIn attribution with estimator-neutral interface.

Score orientation and conditional deletion validation
-----------------------------------------------------
``BENEFICIAL_FIRST`` records the sign convention: for a training observation, a higher mean
local influence over the influence-selection molecules means the observation supports the
target-like loss surface. The tree-influence convention is that a positive local influence
decreases the loss of the selected test example. The orientation is validated only through
conditional deletion: a full pooled model is scored against a separate selection surface, a
high-score batch and a low-score batch are deleted from that same pool, and both pruned models
are evaluated on a third held-out surface. The expected ordering is that low-score deletion
helps more than high-score deletion. No addition-utility calibration is used or reported.

The attributor receives only the training and influence-selection partitions. Its ``score``
signature has no evaluation argument, so evaluation chemistry is unreachable by construction.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from influence_pruning.config import ModelSettings
from influence_pruning.errors import InfluenceDependencyError
from influence_pruning.model import fit_regressor
from influence_pruning.partitions import LabelledPartition
from influence_pruning.similarity import max_tanimoto

BENEFICIAL_FIRST = True
SCORE_DIRECTION = "positive_is_beneficial"
SCORE_MODEL = "full_pooled:XGBRegressor:BoostIn"

SCORE_COLUMNS = (
    "source",
    "origin_class",
    "candidate_id",
    "original_id",
    "inchikey",
    "model_target",
    "model_seed",
    "raw_score",
    "within_seed_rank",
    "train_prediction",
    "train_residual",
    "n_train",
    "n_selection",
    "raw_score_mean",
    "raw_score_sd",
    "rank_mean",
    "rank_sd",
    "n_score_seeds",
    "aggregate_rank",
    "max_tanimoto_to_selection",
    "beneficial_first",
    "score_direction",
    "score_model",
)


class TrainingAttributor(Protocol):
    """Estimator-neutral interface for training-observation attribution."""

    def score(
        self,
        training: LabelledPartition,
        selection: LabelledPartition,
        model_settings: ModelSettings,
        seeds: tuple[int, ...],
    ) -> pd.DataFrame:
        """Score every training observation against the selection surface."""
        ...


def require_boostin() -> type:
    """Import and return the upstream ``BoostIn`` explainer.

    :returns: the ``BoostIn`` class.
    :raises InfluenceDependencyError: tree-influence is absent or incompatible, naming the
        required environment in the message.
    """
    try:
        from tree_influence.explainers import BoostIn
    except Exception as exc:  # pragma: no cover - exercised by the dependency test
        raise InfluenceDependencyError(
            "the BoostIn study requires the tree-influence package, which is isolated in its "
            "own Pixi environment because it pins numpy<2; run "
            "'pixi run -e influence ipp <command> --config <config>' (that environment "
            "excludes the default feature, so it exposes the ipp task)"
        ) from exc
    return BoostIn


def aggregate_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-seed scores into unique, stable cross-seed ranks.

    Within each seed, training observations are ranked by descending raw influence with the
    candidate identifier as the deterministic tie-break. The primary cross-seed score is the
    mean within-seed rank, because raw influence magnitudes are not comparable across fitted
    trees. The global aggregate rank is unique and stable across all training candidates.

    :param scores: one row per training molecule and scoring seed.
    :returns: `scores` with cross-seed aggregate columns merged in.
    :raises InfluencePruningError: the frame already carries aggregate columns.
    """
    from influence_pruning.errors import RunError

    required = {
        "source",
        "origin_class",
        "candidate_id",
        "original_id",
        "inchikey",
        "model_target",
        "model_seed",
        "raw_score",
        "within_seed_rank",
        "train_prediction",
        "train_residual",
        "n_train",
        "n_selection",
    }
    missing = [column for column in required if column not in scores.columns]
    if missing:
        raise RunError(f"scores are missing columns {missing}")
    if "aggregate_rank" in scores.columns:
        raise RunError("scores are already aggregated; aggregate each scoring run exactly once")
    if "max_tanimoto_to_selection" not in scores.columns:
        scores = scores.assign(max_tanimoto_to_selection=float("nan"))

    ranked = scores.copy()
    ranked["within_seed_rank"] = (
        ranked.sort_values(
            ["model_seed", "raw_score", "candidate_id"],
            ascending=[True, not BENEFICIAL_FIRST, True],
            kind="stable",
        )
        .groupby("model_seed", sort=False)
        .cumcount()
        .add(1)
        .reindex(ranked.index)
        .astype(int)
    )
    aggregate = (
        ranked.groupby(
            ["source", "origin_class", "candidate_id", "original_id", "inchikey"],
            as_index=False,
        )
        .agg(
            raw_score_mean=("raw_score", "mean"),
            raw_score_sd=("raw_score", "std"),
            rank_mean=("within_seed_rank", "mean"),
            rank_sd=("within_seed_rank", "std"),
            n_score_seeds=("model_seed", "nunique"),
            model_target=("model_target", "first"),
        )
        .sort_values(["rank_mean", "candidate_id"], kind="stable")
        .reset_index(drop=True)
    )
    aggregate["raw_score_sd"] = aggregate["raw_score_sd"].fillna(0.0)
    aggregate["rank_sd"] = aggregate["rank_sd"].fillna(0.0)
    aggregate["aggregate_rank"] = np.arange(1, len(aggregate) + 1, dtype=int)
    merged = ranked.merge(
        aggregate,
        on=["source", "origin_class", "candidate_id", "original_id", "inchikey", "model_target"],
        how="left",
        validate="many_to_one",
    )
    merged["beneficial_first"] = BENEFICIAL_FIRST
    merged["score_direction"] = SCORE_DIRECTION
    merged["score_model"] = SCORE_MODEL
    return (
        merged.loc[:, list(SCORE_COLUMNS)]
        .sort_values(["aggregate_rank", "model_seed"], kind="stable")
        .reset_index(drop=True)
    )


@dataclass(slots=True)
class BoostInAttributor:
    """BoostIn scores computed in one full pooled model per scoring seed.

    The central property of this study: every training observation is scored conditionally on
    the same full pooled model from which pruning arms are later deleted. Donor sources are
    never scored independently.
    """

    def score(
        self,
        training: LabelledPartition,
        selection: LabelledPartition,
        model_settings: ModelSettings,
        seeds: tuple[int, ...],
    ) -> pd.DataFrame:
        """Score every training row of the pooled model against a selection surface.

        :param training: full pooled training partition (native plus every donor source).
        :param selection: target-like influence-selection partition.
        :param model_settings: model parameters shared with every outcome fit.
        :param seeds: model seeds used for scoring.
        :returns: per-molecule, per-seed scores with cross-seed aggregate ranks.
        :raises InfluenceDependencyError: tree-influence is unavailable.
        :raises RunError: influence values are non-finite or the matrix shape is inconsistent.
        """
        from influence_pruning.errors import RunError

        boostin = require_boostin()
        train_x = training.fingerprints
        train_y = training.frame["model_target"].to_numpy(dtype=float)
        selection_x = selection.fingerprints
        selection_y = selection.frame["model_target"].to_numpy(dtype=float)
        expected_shape = (training.size, selection.size)
        provenance = training.frame.loc[
            :,
            [
                "source",
                "origin_class",
                "candidate_id",
                "original_id",
                "inchikey",
                "model_target",
            ],
        ].reset_index(drop=True)

        frames: list[pd.DataFrame] = []
        for seed in seeds:
            model = fit_regressor(
                train_x, train_y, model_settings, seed, require_explicit_base_score=True
            )
            train_prediction = np.asarray(
                model.predict(np.asarray(train_x, dtype=np.float32)), dtype=float
            )
            explainer = boostin().fit(model, train_x, train_y)
            matrix = np.asarray(
                explainer.get_local_influence(selection_x, selection_y), dtype=float
            )
            if matrix.shape != expected_shape:
                raise RunError(
                    f"influence matrix shape {matrix.shape} does not match the fitted "
                    f"training rows and selection molecules {expected_shape}"
                )
            if not bool(np.isfinite(matrix).all()):
                raise RunError("influence matrix contains non-finite values")
            raw = matrix.mean(axis=1)
            if not bool(np.isfinite(raw).all()):
                raise RunError("training influence scores contain non-finite values")
            frames.append(
                pd.DataFrame(
                    {
                        "source": provenance["source"].to_numpy(),
                        "origin_class": provenance["origin_class"].to_numpy(),
                        "candidate_id": provenance["candidate_id"].to_numpy(),
                        "original_id": provenance["original_id"].to_numpy(),
                        "inchikey": provenance["inchikey"].to_numpy(),
                        "model_target": provenance["model_target"].to_numpy(dtype=float),
                        "model_seed": seed,
                        "raw_score": raw,
                        "within_seed_rank": -1,
                        "train_prediction": train_prediction,
                        "train_residual": provenance["model_target"].to_numpy(dtype=float)
                        - train_prediction,
                        "n_train": training.size,
                        "n_selection": selection.size,
                    }
                )
            )
        scores = pd.concat(frames, ignore_index=True)
        aggregated = aggregate_scores(scores)
        similarity = max_tanimoto(training.fingerprints, selection.fingerprints)
        similarity_by_candidate = dict(
            zip(training.frame["candidate_id"].astype(str), similarity, strict=True)
        )
        aggregated["max_tanimoto_to_selection"] = (
            aggregated["candidate_id"].astype(str).map(similarity_by_candidate).astype(float)
        )
        return aggregated
