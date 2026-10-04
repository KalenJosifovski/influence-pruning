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
    # Data and partition audit

    Read-only view of one completed run. This notebook loads a finished run by path, calls
    `verify` (which is read-only), and refuses to continue if required artifacts are missing
    or the configuration hash disagrees. It never fits models, recomputes attribution,
    regenerates partitions, or recomputes bootstrap resamples.

    The notebooks are read-only with respect to raw run artifacts and model computation.
    They may export derived figures to `analysis/<run-id>/`; `results/` holds only immutable
    script-generated artifacts.
    """)
    return


@app.cell
def _(mo):
    run_input = mo.ui.text(
        value="",
        label=("Run directory (blank = latest completed run under results/boostin_pruning/)"),
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
    if facts.get("dry_run", False):
        summary_banner = mo.callout(
            mo.md(
                "**Dry run.** This run used capped limits and is a smoke test of the artifact "
                "contract, not a scientific result."
            ),
            kind="warn",
        )
    else:
        summary_banner = mo.callout(mo.md("Completed real-data run."), kind="success")
    mo.vstack(
        [
            summary_banner,
            mo.md(
                f"- directory: `{run_dir}`\n"
                f"- kind: `{facts['kind']}`; endpoint: `{facts['endpoint']}`\n"
                f"- target: `{facts['target']}`\n"
                f"- config hash: `{facts['config_hash']}`; fits: `{facts['n_fits']}`\n"
                f"- runtime: `{facts['runtime_seconds']} s`"
            ),
        ]
    )
    return facts, pd, run_dir


@app.cell
def _(mo, pd, run_dir):
    data_audit = pd.read_parquet(run_dir / "data_audit.parquet")
    exclusions = pd.read_parquet(run_dir / "exclusions.parquet")
    manifest = pd.read_parquet(run_dir / "partition_manifest.parquet")
    mo.vstack([mo.md("## Source counts before and after exclusions"), mo.ui.table(data_audit)])
    return exclusions, manifest


@app.cell
def _(exclusions, mo):
    if exclusions.empty:
        exclusion_view = mo.md("No exclusions were recorded for this run.")
    else:
        exclusion_summary = (
            exclusions.groupby(["source", "exclusion_reason"]).size().rename("count").reset_index()
        )
        exclusion_view = mo.vstack(
            [mo.md("### Exclusions by source and reason"), mo.ui.table(exclusion_summary)]
        )
    exclusion_view
    return


@app.cell
def _(exclusions):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    exclusion_figure, exclusion_axis = plt.subplots(figsize=(8, 3.5))
    if not exclusions.empty:
        exclusion_counts = (
            exclusions.groupby(["exclusion_reason", "source"]).size().reset_index(name="n")
        )
        sns.barplot(
            data=exclusion_counts, x="exclusion_reason", y="n", hue="source", ax=exclusion_axis
        )
        exclusion_axis.tick_params(axis="x", rotation=20)
    exclusion_axis.set_title("Exclusions by reason")
    exclusion_axis.set_ylabel("molecules")
    exclusion_figure
    return plt, sns


@app.cell
def _(manifest, mo):
    role_source = (
        manifest.groupby(["role", "source", "origin_class"]).size().rename("n").reset_index()
    )
    mo.vstack([mo.md("### Partition membership by role and source"), mo.ui.table(role_source)])
    return


@app.cell
def _(manifest, plt, sns):
    distribution_figure, distribution_axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.boxplot(
        data=manifest,
        x="role",
        y="model_target",
        hue="source",
        ax=distribution_axes[0],
        showfliers=False,
    )
    distribution_axes[0].set_title("Endpoint distribution by role and source")
    distribution_axes[0].set_ylabel("model target (log10 mL/min/mL)")
    sns.histplot(
        data=manifest,
        x="model_target",
        hue="source",
        multiple="stack",
        bins=30,
        ax=distribution_axes[1],
    )
    distribution_axes[1].set_title("Model-target distribution across all partitions")
    distribution_figure
    return


@app.cell
def _(facts, manifest, mo):
    split_record = facts.get("split_record", {})
    if split_record.get("test_mode") == "temporal_halves":
        allocation_lines = [
            "- test mode: `temporal_halves`",
            (
                f"- selection IDs: {split_record.get('selection_id_min')}–"
                f"{split_record.get('selection_id_max')} "
                f"({split_record.get('n_selection')} molecules)"
            ),
            (
                f"- evaluation IDs: {split_record.get('evaluation_id_min')}–"
                f"{split_record.get('evaluation_id_max')} "
                f"({split_record.get('n_evaluation')} molecules)"
            ),
            (
                "- selection earlier than evaluation: "
                f"`{split_record.get('selection_earlier_than_evaluation')}`"
            ),
        ]
    else:
        allocation_lines = [
            f"- test mode: `{split_record.get('test_mode')}`",
            f"- clusters: {split_record.get('n_clusters')} "
            f"(selection {split_record.get('n_selection_clusters')}, "
            f"evaluation {split_record.get('n_evaluation_clusters')})",
            f"- achieved selection fraction: `{split_record.get('achieved_selection_fraction')}`",
        ]
    cluster_note = (
        f"- manifest rows carrying cluster assignments: "
        f"`{int(manifest['cluster_id'].notna().sum())}`"
    )
    mo.md("### Partition allocation audit\n" + "\n".join(allocation_lines) + "\n" + cluster_note)
    return


@app.cell
def _(manifest, mo):
    training_keys = set(manifest.loc[manifest["role"].eq("full_training"), "inchikey"])
    selection_keys = set(manifest.loc[manifest["role"].eq("influence_selection"), "inchikey"])
    evaluation_keys = set(manifest.loc[manifest["role"].eq("evaluation"), "inchikey"])
    leakage_overlap = {
        "training ∩ selection": len(training_keys & selection_keys),
        "training ∩ evaluation": len(training_keys & evaluation_keys),
        "selection ∩ evaluation": len(selection_keys & evaluation_keys),
    }
    duplicated_roles = int(manifest["candidate_id"].duplicated().sum())
    mo.md(
        "### Leakage guards\n"
        + "\n".join(f"- {key}: `{value}`" for key, value in leakage_overlap.items())
        + f"\n- duplicated candidate IDs across all roles: `{duplicated_roles}`\n\n"
        "Selection and evaluation are structurally disjoint by whole-structure InChIKey and by "
        "candidate identifier; neither appears in full training."
    )
    return


@app.cell
def _(manifest, mo):
    scoring_visibility = manifest["label_visible_to_scoring"].value_counts().to_dict()
    evaluation_visibility = manifest["label_visible_to_evaluation"].value_counts().to_dict()
    mo.md(
        "### Label visibility\n"
        f"- label_visible_to_scoring: `{scoring_visibility}`\n"
        f"- label_visible_to_evaluation: `{evaluation_visibility}`\n\n"
        "Scoring sees training and selection labels only; evaluation labels are visible only to "
        "the evaluation stage."
    )
    return


@app.cell
def _(manifest, plt, run_dir, sns):
    from pathlib import Path

    composition = (
        manifest.loc[manifest["role"].eq("full_training")]
        .groupby(["source", "origin_class"])
        .size()
        .reset_index(name="n")
    )
    composition_figure, composition_axis = plt.subplots(figsize=(7, 3.5))
    sns.barplot(data=composition, x="source", y="n", hue="origin_class", ax=composition_axis)
    composition_axis.set_title("Full pooled training composition")
    composition_axis.set_ylabel("molecules")
    figures_dir = Path("analysis") / run_dir.name / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    composition_figure.tight_layout()
    composition_figure.savefig(figures_dir / "notebook01_full_pool_composition.png", dpi=200)
    composition_figure
    return


if __name__ == "__main__":
    app.run()
