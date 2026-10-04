import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # BoostIn pruning analysis

    Read-only analysis of one completed run: score diagnostics, pruning curves against the
    random-draw reference, source/bin-stratified control comparisons with their matching
    diagnostics, paired bootstrap distributions, mixed batch composition, and selected
    structures. This notebook never fits a model, recomputes attribution, regenerates
    partitions, or recomputes bootstrap resamples; every number comes from the saved
    artifacts.

    The notebooks are read-only with respect to raw run artifacts and model computation.
    They may export derived figures to `analysis/<run-id>/`; `results/` holds only immutable
    script-generated artifacts. The stratified controls are matched on source composition and
    within-source label quantile bins, not on exact continuous label values, so non-zero
    Wasserstein distances are expected and are reported below.
    """)
    return


@app.cell
def _(mo):
    run_input = mo.ui.text(
        value="",
        label="Run directory (blank = latest completed run under results/boostin_pruning/)",
        full_width=True,
    )
    run_input
    return (run_input,)


@app.cell
def _(mo, run_input):
    from pathlib import Path

    import pandas as pd

    from influence_pruning.artifacts import verify_run

    def latest_completed_run() -> Path:
        candidates = sorted(Path("results/boostin_pruning").glob("*/run_facts.json"))
        completed = [path.parent for path in candidates if (path.parent / "COMPLETED").is_file()]
        if not completed:
            raise FileNotFoundError(
                "no completed run found under results/boostin_pruning/; run "
                "'pixi run -e influence ipp dry-run --config "
                "configs/dryrun_boostin_expansionrx_hlm.yaml' first"
            )
        return max(completed, key=lambda path: path.stat().st_mtime)

    run_dir = Path(run_input.value).expanduser() if run_input.value else latest_completed_run()
    facts = verify_run(run_dir)
    dry_run_note = (
        "**Dry run.** Capped smoke test; not a scientific result."
        if facts.get("dry_run", False)
        else "Completed real-data run."
    )
    mo.md(
        f"Run `{run_dir.name}` — endpoint `{facts['endpoint']}`, target `{facts['target']}`, "
        f"config hash `{facts['config_hash']}`, {facts['n_arms']} arms.\n\n{dry_run_note}"
    )
    return pd, run_dir


@app.cell
def _(pd, run_dir):
    scores = pd.read_parquet(run_dir / "boostin_scores.parquet")
    arm_summary = pd.read_parquet(run_dir / "arm_summary.parquet")
    arm_manifest = pd.read_parquet(run_dir / "arm_manifest.parquet")
    metrics = pd.read_parquet(run_dir / "seed_metrics.parquet")
    effects = pd.read_parquet(run_dir / "paired_effects.parquet")
    draws = pd.read_parquet(run_dir / "bootstrap_draws.parquet")
    manifest = pd.read_parquet(run_dir / "partition_manifest.parquet")
    per_molecule = scores.drop_duplicates("candidate_id").copy()
    return (
        arm_manifest,
        arm_summary,
        draws,
        effects,
        manifest,
        metrics,
        per_molecule,
        scores,
    )


@app.cell
def _(per_molecule):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    score_figure, score_axes = plt.subplots(1, 3, figsize=(15, 4))
    sns.violinplot(
        data=per_molecule,
        x="source",
        y="raw_score_mean",
        hue="source",
        legend=False,
        ax=score_axes[0],
        cut=0,
    )
    score_axes[0].set_title("Mean pooled influence by source")
    sns.scatterplot(
        data=per_molecule,
        x="model_target",
        y="raw_score_mean",
        hue="source",
        s=12,
        ax=score_axes[1],
    )
    label_correlation = (
        per_molecule[["raw_score_mean", "model_target"]].corr(method="spearman").iloc[0, 1]
    )
    score_axes[1].set_title(f"Influence vs label (Spearman rho = {label_correlation:.3f})")
    sns.scatterplot(
        data=per_molecule,
        x="max_tanimoto_to_selection",
        y="raw_score_mean",
        hue="source",
        s=12,
        ax=score_axes[2],
    )
    similarity_correlation = (
        per_molecule[["raw_score_mean", "max_tanimoto_to_selection"]]
        .corr(method="spearman")
        .iloc[0, 1]
    )
    score_axes[2].set_title(
        f"Influence vs selection similarity (Spearman rho = {similarity_correlation:.3f})"
    )
    score_figure.tight_layout()
    score_figure
    return plt, sns


@app.cell
def _(per_molecule, plt, sns):
    residual_figure, residual_axis = plt.subplots(figsize=(7, 4))
    sns.scatterplot(
        data=per_molecule,
        x="raw_score_mean",
        y=per_molecule["train_residual"].abs(),
        hue="source",
        s=12,
        ax=residual_axis,
    )
    residual_axis.set_ylabel("|full-model training residual|")
    residual_axis.set_title("Influence vs training residual magnitude")
    residual_figure.tight_layout()
    residual_figure
    return


@app.cell
def _(metrics):
    full_rmse = metrics.loc[metrics["condition"].eq("full_pooled")].set_index("seed")["rmse"]
    mean_metrics = (
        metrics.groupby(
            [
                "condition",
                "condition_kind",
                "eligible_pool",
                "stratum",
                "policy",
                "direction",
                "batch_size",
                "draw_number",
            ],
            dropna=False,
            as_index=False,
        )
        .agg(mean_rmse=("rmse", "mean"), n_seeds=("rmse", "size"))
        .assign(
            cell_group=lambda frame: (
                frame["eligible_pool"].fillna("baseline") + ":" + frame["stratum"].fillna("")
            )
        )
    )
    mean_metrics["mean_delta"] = float(full_rmse.mean()) - mean_metrics["mean_rmse"]
    return (mean_metrics,)


@app.cell
def _(mean_metrics):
    prune_groups = sorted(
        mean_metrics.loc[mean_metrics["condition_kind"].eq("pruned"), "cell_group"].unique()
    )
    batch_sizes = sorted(mean_metrics["batch_size"].dropna().unique().tolist())
    return batch_sizes, prune_groups


@app.cell
def _(batch_sizes, mean_metrics, plt, prune_groups):
    n_columns = 3
    n_rows = -(-len(prune_groups) // n_columns)
    curve_figure, curve_axes = plt.subplots(
        n_rows, n_columns, figsize=(5 * n_columns, 3.4 * n_rows), squeeze=False, sharey=True
    )
    for curve_index, curve_group in enumerate(prune_groups):
        curve_axis = curve_axes[curve_index // n_columns][curve_index % n_columns]
        curve_subset = mean_metrics.loc[mean_metrics["cell_group"].eq(curve_group)]
        curve_random = curve_subset.loc[curve_subset["policy"].eq("random")]
        for curve_size in batch_sizes:
            random_values = curve_random.loc[
                curve_random["batch_size"].eq(curve_size), "mean_delta"
            ]
            if len(random_values):
                curve_axis.scatter(
                    [curve_size] * len(random_values), random_values, color="0.65", s=14, zorder=1
                )
        for curve_policy, curve_colour, curve_label in (
            ("boostin_high", "#c44e52", "boostin_high"),
            ("boostin_low", "#4c72b0", "boostin_low"),
        ):
            curve_ranked = curve_subset.loc[curve_subset["policy"].eq(curve_policy)].sort_values(
                "batch_size"
            )
            if len(curve_ranked):
                curve_axis.plot(
                    curve_ranked["batch_size"],
                    curve_ranked["mean_delta"],
                    marker="o",
                    color=curve_colour,
                    label=curve_label,
                    zorder=3,
                )
        for curve_policy, curve_colour, curve_label in (
            ("label_source_matched_high", "#c44e52", "stratified high (mean)"),
            ("label_source_matched_low", "#4c72b0", "stratified low (mean)"),
        ):
            curve_matched = (
                curve_subset.loc[curve_subset["policy"].eq(curve_policy)]
                .groupby("batch_size")["mean_delta"]
                .mean()
            )
            if len(curve_matched):
                curve_axis.scatter(
                    curve_matched.index,
                    curve_matched.values,
                    marker="x",
                    s=45,
                    color=curve_colour,
                    zorder=4,
                    label=curve_label,
                )
        curve_axis.axhline(0.0, color="0.3", linewidth=0.8, linestyle="--")
        curve_axis.set_title(curve_group)
        curve_axis.set_xlabel("batch size")
        curve_axis.set_ylabel("mean delta RMSE (positive = pruning helped)")
        curve_axis.legend(fontsize=7)
    for curve_index in range(len(prune_groups), n_rows * n_columns):
        curve_axes[curve_index // n_columns][curve_index % n_columns].axis("off")
    curve_figure.tight_layout()
    curve_figure
    return


@app.cell
def _(effects, plt):
    matched_effects = effects.loc[effects["comparison_kind"].eq("ranked_vs_matched_median")].dropna(
        subset=["direction", "batch_size"]
    )
    if matched_effects.empty:
        matched_figure = None
    else:
        matched_figure, matched_axis = plt.subplots(figsize=(9, 4))
        matched_effects = matched_effects.assign(
            cell_group=matched_effects["eligible_pool"].astype(str)
            + ":"
            + matched_effects["stratum"].astype(str)
        )
        matched_policies = (
            ("boostin_high", "#c44e52"),
            ("boostin_low", "#4c72b0"),
        )
        for matched_policy, matched_colour in matched_policies:
            matched_subset = matched_effects.loc[matched_effects["policy"].eq(matched_policy)]
            for _matched_group, matched_rows in matched_subset.groupby("cell_group"):
                matched_axis.errorbar(
                    matched_rows["batch_size"],
                    matched_rows["effect"],
                    yerr=[
                        matched_rows["effect"] - matched_rows["ci_low"],
                        matched_rows["ci_high"] - matched_rows["effect"],
                    ],
                    marker="o",
                    capsize=3,
                    color=matched_colour,
                    alpha=0.7,
                    label=f"{matched_policy} vs stratified median",
                )
        matched_axis.axhline(0.0, color="0.3", linewidth=0.8, linestyle="--")
        matched_axis.set_xlabel("batch size")
        matched_axis.set_ylabel(
            "delta vs median stratified-control policy (positive = ranked arm better)"
        )
        matched_axis.set_title("Ranked arms versus label/source-stratified controls")
        matched_axis.legend(fontsize=7)
        matched_figure.tight_layout()
    matched_figure
    return (matched_effects,)


@app.cell
def _(arm_summary, effects, mo):
    policy_comparisons = effects.loc[
        effects["comparison_kind"].isin(["ranked_vs_random_median", "ranked_vs_matched_median"]),
        [
            "comparison_id",
            "comparison_kind",
            "control_policy",
            "n_control_draws",
            "control_arm_ids",
            "effect",
            "ci_low",
            "ci_high",
        ],
    ].sort_values(["comparison_kind", "comparison_id"], kind="stable")
    stratified_rows = arm_summary.loc[
        arm_summary["policy"].isin(["label_source_matched_high", "label_source_matched_low"])
    ].copy()
    planned_stratified = stratified_rows.loc[stratified_rows["status"].eq("planned")]
    infeasible_stratified = stratified_rows.loc[stratified_rows["status"].eq("infeasible")]
    diagnostics_views = [
        mo.md(
            "### Stratified-control matching quality\n"
            "These controls preserve source composition and within-source label quantile bins; "
            "they are not exact continuous-label matches, so non-zero Wasserstein distances are "
            "expected. Infeasible groups are recorded explicitly and are never replaced by random "
            "draws. The full intended source × bin composition is preserved per arm in "
            "`arm_summary.matching_plan`."
        )
    ]
    if policy_comparisons.empty:
        diagnostics_views.append(mo.md("No policy-level control comparisons were recorded."))
    else:
        diagnostics_views.append(mo.md("### Policy-level comparisons (median of all draws)"))
        diagnostics_views.append(mo.ui.table(policy_comparisons))
    if planned_stratified.empty:
        diagnostics_views.append(mo.md("No stratified-control arm was constructed in this run."))
    else:
        planned_table = (
            planned_stratified.groupby(
                ["eligible_pool", "stratum", "direction", "batch_size"],
                as_index=False,
            )
            .agg(
                n_draws=("arm_id", "size"),
                bins_used_min=("matching_bins_used", "min"),
                wasserstein_overall_mean=("label_wasserstein_overall", "mean"),
                wasserstein_overall_max=("label_wasserstein_overall", "max"),
                wasserstein_max_source=("label_wasserstein_max_source", "max"),
                matching_plan=("matching_plan", "first"),
            )
            .sort_values(["eligible_pool", "stratum", "direction", "batch_size"], kind="stable")
        )
        diagnostics_views.append(mo.ui.table(planned_table))
    if not infeasible_stratified.empty:
        infeasible_table = (
            infeasible_stratified.loc[
                :, ["eligible_pool", "stratum", "direction", "batch_size", "matching_diagnostic"]
            ]
            .drop_duplicates()
            .sort_values(["eligible_pool", "stratum", "direction", "batch_size"], kind="stable")
        )
        diagnostics_views.append(mo.md("### Infeasible stratified-control groups"))
        diagnostics_views.append(mo.ui.table(infeasible_table))
    mo.vstack(diagnostics_views)
    return


@app.cell
def _(matched_effects, plt):
    seed_figure, seed_axis = plt.subplots(figsize=(10, 4))
    for seed_policy, seed_colour in (("boostin_high", "#c44e52"), ("boostin_low", "#4c72b0")):
        seed_subset = matched_effects.loc[matched_effects["policy"].eq(seed_policy)]
        seed_axis.bar(
            seed_subset["eligible_pool"].astype(str)
            + ":"
            + seed_subset["stratum"].astype(str)
            + " k"
            + seed_subset["batch_size"].astype(str),
            seed_subset["effect"],
            color=seed_colour,
            alpha=0.7,
            label=seed_policy,
        )
    seed_axis.axhline(0.0, color="0.3", linewidth=0.8, linestyle="--")
    seed_axis.set_ylabel("delta vs stratified median")
    seed_axis.set_title("Stratified-control contrasts by pool and batch size")
    seed_axis.tick_params(axis="x", rotation=90, labelsize=6)
    seed_axis.legend(fontsize=7)
    seed_figure.tight_layout()
    seed_figure
    return


@app.cell
def _(draws, effects, mean_metrics, plt):
    bootstrap_ranked = mean_metrics.loc[
        mean_metrics["policy"].eq("boostin_high")
        & mean_metrics["cell_group"].eq("native_only:target")
    ].sort_values("batch_size")
    if bootstrap_ranked.empty:
        bootstrap_figure = None
    else:
        bootstrap_figure, bootstrap_axes = plt.subplots(
            1,
            len(bootstrap_ranked),
            figsize=(4 * len(bootstrap_ranked), 3.4),
            squeeze=False,
            sharey=True,
        )
        for bootstrap_index, bootstrap_condition in enumerate(bootstrap_ranked["condition"]):
            bootstrap_comparison = f"full_pooled_vs_{bootstrap_condition}"
            bootstrap_values = draws.loc[draws["comparison_id"].eq(bootstrap_comparison), "effect"]
            bootstrap_axis = bootstrap_axes[0][bootstrap_index]
            bootstrap_axis.hist(bootstrap_values, bins=30, color="#55a868", alpha=0.85)
            bootstrap_axis.axvline(0.0, color="0.3", linestyle="--", linewidth=0.8)
            bootstrap_row = effects.loc[effects["comparison_id"].eq(bootstrap_comparison)].iloc[0]
            bootstrap_axis.axvline(bootstrap_row["effect"], color="black", linewidth=1.0)
            bootstrap_axis.set_title(
                f"native_only boostin_high\nk={int(bootstrap_row['batch_size'])}"
            )
            bootstrap_axis.set_xlabel("delta RMSE resample")
        bootstrap_figure.suptitle(
            "Paired cluster-bootstrap distributions: full pooled vs high-influence deletion"
        )
        bootstrap_figure.tight_layout()
    bootstrap_figure
    return


@app.cell
def _(arm_summary, plt, sns):
    mixed_batches = arm_summary.loc[
        arm_summary["eligible_pool"].eq("mixed_pool")
        & arm_summary["status"].eq("planned")
        & arm_summary["policy"].isin(["boostin_high", "boostin_low"])
    ].copy()
    mixed_batches["batch_label"] = (
        mixed_batches["policy"].str.replace("boostin_", "")
        + " k"
        + mixed_batches["batch_size"].astype(str)
    )
    mixed_composition = mixed_batches.melt(
        id_vars=["batch_label"],
        value_vars=["source_biogen", "source_polaris", "source_expansionrx"],
        var_name="source",
        value_name="count",
    )
    mixed_composition["source"] = mixed_composition["source"].str.replace("source_", "")
    composition_figure, composition_axis = plt.subplots(figsize=(9, 4))
    sns.barplot(
        data=mixed_composition, x="batch_label", y="count", hue="source", ax=composition_axis
    )
    composition_axis.set_title("Source composition of mixed-pool ranked batches")
    composition_axis.tick_params(axis="x", rotation=30)
    composition_figure.tight_layout()
    composition_figure
    return


@app.cell
def _(arm_manifest, manifest, scores):
    from rdkit import Chem
    from rdkit.Chem import Draw

    structure_smiles = dict(
        zip(
            manifest["candidate_id"].astype(str),
            manifest["canonical_smiles"].astype(str),
            strict=True,
        )
    )
    structure_rank = scores.drop_duplicates("candidate_id").set_index("candidate_id")[
        "aggregate_rank"
    ]
    structure_size = int(
        arm_manifest.loc[arm_manifest["eligible_pool"].eq("mixed_pool"), "batch_size"].max()
    )
    structure_arms = {
        "high": f"mixed_pool:all:boostin_high:k{structure_size}",
        "low": f"mixed_pool:all:boostin_low:k{structure_size}",
    }
    structure_ids: list[str] = []
    structure_legends: list[str] = []
    for structure_label, structure_arm in structure_arms.items():
        structure_members = arm_manifest.loc[
            arm_manifest["arm_id"].eq(structure_arm), "candidate_id"
        ].astype(str)
        for structure_candidate in structure_members.head(6):
            structure_ids.append(structure_candidate)
            structure_legends.append(
                f"{structure_label} rank {int(structure_rank.loc[structure_candidate])}"
            )
    structure_mols = [
        Chem.MolFromSmiles(structure_smiles[candidate]) for candidate in structure_ids
    ]
    structure_image = Draw.MolsToGridImage(
        structure_mols, legends=structure_legends, molsPerRow=4, subImgSize=(230, 170)
    )
    structure_image
    return


if __name__ == "__main__":
    app.run()
