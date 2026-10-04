"""Cluster bootstrap over evaluation chemistry for paired condition effects.

The bootstrap resamples whole evaluation clusters and recomputes each condition's RMSE from the
sampled molecules on every resample; it never averages precomputed molecule errors. Every
resampled effect is returned so the run can persist the full distribution for the notebooks.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    """Paired effect with a cluster-bootstrap percentile interval.

    :param comparison_id: stable comparison identifier.
    :param settings: human-readable bootstrap settings.
    :param effect: paired effect (mean over matched seeds of ``statistic_A - statistic_B``).
    :param effect_definition: human-readable definition of the paired statistic.
    :param ci_low: 2.5th percentile of the bootstrap distribution.
    :param ci_high: 97.5th percentile of the bootstrap distribution.
    """

    comparison_id: str
    settings: str
    effect: float
    effect_definition: str
    ci_low: float
    ci_high: float
    n_resamples: int
    n_molecules: int
    n_clusters: int


@dataclass(frozen=True, slots=True)
class PairedBootstrapResult:
    """A bootstrap interval together with every resampled effect.

    :param interval: compact comparison record for the effects table.
    :param draws: ``(n_resamples,)`` resampled paired effects.
    """

    interval: BootstrapInterval
    draws: np.ndarray


def cluster_bootstrap_statistic(
    statistic: Callable[[np.ndarray], float],
    clusters: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
) -> tuple[float, np.ndarray]:
    """Percentile distribution for an arbitrary statistic under cluster resampling.

    :param statistic: callable mapping a molecule-index array to a scalar; it is evaluated on
        the full sample for the point estimate and on each resample otherwise.
    :param clusters: ``(n_molecules,)`` integer cluster labels of the evaluation molecules.
    :param n_resamples: number of cluster-bootstrap resamples.
    :param seed: bootstrap seed.
    :returns: ``(point estimate, resampled effects)``.
    """
    labels = np.asarray(clusters).astype(int)
    unique_ids, order, positions = _cluster_layout(labels)
    n_clusters = unique_ids.size
    point = float(statistic(np.arange(labels.size)))
    rng = np.random.default_rng(seed)
    values = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        multiplicities = rng.multinomial(n_clusters, np.full(n_clusters, 1.0 / n_clusters))
        values[index] = statistic(np.repeat(order, multiplicities[positions]))
    return point, values


def cluster_bootstrap_delta(
    observed: np.ndarray,
    predictions_a: np.ndarray,
    predictions_b: np.ndarray,
    clusters: np.ndarray,
    *,
    comparison_id: str,
    n_resamples: int,
    seed: int,
    settings: str = "",
    effect_definition: str = "rmse_a_minus_rmse_b",
) -> PairedBootstrapResult:
    """Bootstrap a paired RMSE difference by resampling evaluation clusters.

    Both conditions' RMSEs are recomputed from the sampled molecules on the same resample;
    per-seed RMSEs are averaged over matched seeds within each resample.

    :param observed: ``(n_molecules,)`` observed targets.
    :param predictions_a: ``(n_molecules, n_seeds)`` predictions of condition A.
    :param predictions_b: ``(n_molecules, n_seeds)`` predictions of condition B.
    :param clusters: ``(n_molecules,)`` integer cluster labels of the evaluation molecules.
    :param comparison_id: stable comparison identifier.
    :param n_resamples: number of cluster-bootstrap resamples.
    :param seed: bootstrap seed.
    :param settings: human-readable bootstrap settings for the artifact.
    :param effect_definition: human-readable definition of the paired statistic.
    :returns: paired effect, percentile interval, and every resampled effect.
    """
    observed_values = np.asarray(observed, dtype=float)
    pred_a = np.asarray(predictions_a, dtype=float)
    pred_b = np.asarray(predictions_b, dtype=float)
    cluster_labels = np.asarray(clusters).astype(int)
    n_clusters = int(np.unique(cluster_labels).size)
    errors_a = (observed_values[:, None] - pred_a) ** 2
    errors_b = (observed_values[:, None] - pred_b) ** 2

    def statistic(index: np.ndarray) -> float:
        """Seed-averaged paired RMSE difference on the sampled molecules."""
        return _delta(errors_a, errors_b, index)

    effect, draws = cluster_bootstrap_statistic(
        statistic, cluster_labels, n_resamples=n_resamples, seed=seed
    )
    ci_low, ci_high = np.percentile(draws, [2.5, 97.5])
    interval = BootstrapInterval(
        comparison_id=comparison_id,
        settings=settings,
        effect=float(effect),
        effect_definition=effect_definition,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        n_resamples=int(n_resamples),
        n_molecules=int(observed_values.size),
        n_clusters=int(n_clusters),
    )
    return PairedBootstrapResult(interval=interval, draws=draws)


def cluster_bootstrap_median_control(
    observed: np.ndarray,
    ranked: np.ndarray,
    controls: np.ndarray,
    clusters: np.ndarray,
    *,
    comparison_id: str,
    n_resamples: int,
    seed: int,
    settings: str = "",
    effect_definition: str = "median_control_rmse_minus_ranked_rmse",
) -> PairedBootstrapResult:
    """Bootstrap a ranked arm against the median RMSE of an entire control policy.

    For every evaluation-cluster resample, each control draw's RMSE is recomputed from the
    sampled molecules, averaged over matched seeds; the statistic is the median of those
    control RMSEs minus the ranked arm's seed-averaged RMSE. Positive values mean the ranked
    deletion arm outperformed the median control policy. The point estimate is computed the
    same way on the full evaluation set. No control draw is selected, filtered, or weighted by
    its observed effect.

    Conditional on the recorded control draws, the resamples capture evaluation-chemistry
    uncertainty; draw-to-draw variation is a separately reported descriptive distribution and
    is not covered by this interval.

    :param observed: ``(n_molecules,)`` observed targets.
    :param ranked: ``(n_molecules, n_seeds)`` predictions of the ranked condition.
    :param controls: ``(n_draws, n_molecules, n_seeds)`` predictions of every control draw.
    :param clusters: ``(n_molecules,)`` integer cluster labels of the evaluation molecules.
    :param comparison_id: stable comparison identifier.
    :param n_resamples: number of cluster-bootstrap resamples.
    :param seed: bootstrap seed.
    :param settings: human-readable bootstrap settings for the artifact.
    :param effect_definition: human-readable definition of the paired statistic.
    :returns: point estimate, percentile interval, and every resampled effect.
    :raises ValueError: the prediction arrays are not shaped as declared or no control draw
        is available.
    """
    observed_values = np.asarray(observed, dtype=float)
    ranked_values = np.asarray(ranked, dtype=float)
    control_values = np.asarray(controls, dtype=float)
    if ranked_values.ndim != 2:
        raise ValueError("ranked predictions must be (n_molecules, n_seeds)")
    if control_values.ndim != 3 or control_values.shape[1:] != ranked_values.shape:
        raise ValueError(
            "controls must be (n_draws, n_molecules, n_seeds) aligned with the ranked predictions"
        )
    if control_values.shape[0] == 0:
        raise ValueError("a median-control comparison requires at least one control draw")
    cluster_labels = np.asarray(clusters).astype(int)
    n_clusters = int(np.unique(cluster_labels).size)
    ranked_errors = (observed_values[:, None] - ranked_values) ** 2
    control_errors = (observed_values[None, :, None] - control_values) ** 2

    def statistic(index: np.ndarray) -> float:
        """Median control-policy RMSE minus the ranked arm's RMSE on the sampled molecules."""
        ranked_rmse = np.sqrt(ranked_errors[index].mean(axis=0)).mean()
        control_rmses = np.sqrt(control_errors[:, index, :].mean(axis=1)).mean(axis=1)
        return float(np.median(control_rmses) - ranked_rmse)

    effect, draws = cluster_bootstrap_statistic(
        statistic, cluster_labels, n_resamples=n_resamples, seed=seed
    )
    ci_low, ci_high = np.percentile(draws, [2.5, 97.5])
    interval = BootstrapInterval(
        comparison_id=comparison_id,
        settings=settings,
        effect=float(effect),
        effect_definition=effect_definition,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        n_resamples=int(n_resamples),
        n_molecules=int(observed_values.size),
        n_clusters=int(n_clusters),
    )
    return PairedBootstrapResult(interval=interval, draws=draws)


def _delta(errors_a: np.ndarray, errors_b: np.ndarray, index: np.ndarray) -> float:
    """Recompute the seed-averaged RMSE difference on a resampled molecule index."""
    rmse_a = np.sqrt(errors_a[index].mean(axis=0)).mean()
    rmse_b = np.sqrt(errors_b[index].mean(axis=0)).mean()
    return float(rmse_a - rmse_b)


def _cluster_layout(clusters: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return unique cluster ids, molecule indices grouped by cluster, and positions."""
    unique_ids = np.unique(clusters)
    order = np.argsort(clusters, kind="stable")
    positions = np.searchsorted(unique_ids, clusters[order])
    return unique_ids, order, positions
