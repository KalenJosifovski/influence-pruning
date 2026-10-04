"""Interactively interrogate conditional BoostIn pruning experiments."""

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="full")


@app.cell
def _():
    from textwrap import dedent
    import pathlib

    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from influence_pruning.artifacts import verify_run
    from influence_pruning.evaluate import EFFECT_COLUMNS

    return EFFECT_COLUMNS, dedent, go, make_subplots, mo, np, pathlib, pd, verify_run


@app.cell
def _(EFFECT_COLUMNS, pathlib, pd):
    run_root = pathlib.Path("results/boostin_pruning")
    all_completed_runs = sorted(
        path.parent
        for path in run_root.glob("*/run_facts.json")
        if (path.parent / "COMPLETED").is_file()
    )
    completed_runs = [
        path
        for path in all_completed_runs
        if set(EFFECT_COLUMNS).issubset(pd.read_parquet(path / "paired_effects.parquet").columns)
    ]
    if not completed_runs:
        raise FileNotFoundError(
            f"no completed runs with the current effects schema under {run_root}"
        )
    return (completed_runs,)


@app.cell
def _(completed_runs, mo):
    run_labels = {
        f"{path.name.split('_')[3].title()} — {path.name[:15]}": str(path)
        for path in completed_runs
    }
    run_selector = mo.ui.dropdown(
        options=run_labels,
        value=next(label for label in run_labels if "Expansionrx" in label),
        label="Evaluation programme",
        full_width=True,
    )
    run_selector
    return run_labels, run_selector


@app.cell
def _(pathlib, pd, run_selector, verify_run):
    run_dir = pathlib.Path(run_selector.value)
    run_facts = verify_run(run_dir)
    effects = pd.read_parquet(run_dir / "paired_effects.parquet")
    predictions = pd.read_parquet(run_dir / "predictions.parquet")
    arm_manifest = pd.read_parquet(run_dir / "arm_manifest.parquet")
    partition_manifest = pd.read_parquet(run_dir / "partition_manifest.parquet")
    scores = pd.read_parquet(run_dir / "boostin_scores.parquet").drop_duplicates("candidate_id")
    return arm_manifest, effects, partition_manifest, predictions, run_dir, run_facts, scores


@app.cell
def _(mo, partition_manifest, run_dir, run_facts):
    evaluation = partition_manifest.loc[partition_manifest["role"].eq("evaluation")]
    selection = partition_manifest.loc[partition_manifest["role"].eq("influence_selection")]
    target = str(run_facts["target"])
    test_mode = str(run_facts.get("split_record", {}).get("test_mode", "unknown"))
    caveat = (
        "This Polaris result has only 13 evaluation clusters; treat every interval as exploratory."
        if target == "polaris"
        else (
            "The displayed intervals are paired cluster-bootstrap intervals, not "
            "multiplicity-adjusted p-values."
        )
    )
    mo.vstack(
        [
            mo.md("# The Public-Data Surgeon"),
            mo.md(
                "**Can external public bioactivity data improve a local model—or only shift its "
                "calibration?** This explorer reads completed conditional BoostIn deletion "
                "experiments. It never fits, attributes, or rewrites a model."
            ),
            mo.callout(
                mo.md(
                    f"**{target.title()} as evaluation programme** · `{test_mode}` · "
                    f"selection: {len(selection):,} molecules · "
                    f"evaluation: {len(evaluation):,} molecules.  "
                    f"{caveat}"
                ),
                kind="warn" if target == "polaris" else "info",
            ),
            mo.md(f"Immutable source artifacts: `{run_dir}`."),
        ]
    )
    return evaluation, selection, target


@app.cell
def _(go, make_subplots, mo, partition_manifest):
    partition_counts = (
        partition_manifest.groupby(["role", "source"], as_index=False)
        .size()
        .rename(columns={"size": "n_molecules"})
    )
    audit_figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Programme composition", "Label distributions by role"),
    )
    audit_source_colours = {"expansionrx": "#4C78A8", "biogen": "#F58518", "polaris": "#54A24B"}
    for audit_source, source_counts in partition_counts.groupby("source", sort=False):
        audit_figure.add_trace(
            go.Bar(
                x=source_counts["role"],
                y=source_counts["n_molecules"],
                name=str(audit_source),
                marker_color=audit_source_colours.get(str(audit_source), "#777777"),
            ),
            row=1,
            col=1,
        )
    for role, role_rows in partition_manifest.groupby("role", sort=False):
        audit_figure.add_trace(
            go.Box(
                x=role_rows["source"],
                y=role_rows["model_target"],
                name=str(role).replace("_", " "),
                legendgroup=str(role),
                showlegend=False,
                boxpoints=False,
            ),
            row=1,
            col=2,
        )
    audit_figure.update_layout(
        barmode="stack",
        height=410,
        margin={"l": 50, "r": 20, "t": 65, "b": 80},
        legend={"orientation": "h", "y": -0.25},
    )
    audit_figure.update_yaxes(title_text="Molecules", row=1, col=1)
    audit_figure.update_yaxes(title_text="Log HLM clearance", row=1, col=2)
    mo.vstack(
        [
            mo.md("## 0. Audit the evidence surface"),
            mo.md(
                "The selection labels inform BoostIn; the evaluation labels are untouched until "
                "the deletion arms are assessed. External donors remain in full pooled training."
            ),
            audit_figure,
        ]
    )
    return (partition_counts,)


@app.cell
def _(effects, go, mo, np):
    baseline = effects.loc[effects["comparison_kind"].eq("baseline")].iloc[0]
    # RMSE values are read in the calibration section; this effect is already paired.
    baseline_figure = go.Figure()
    baseline_figure.add_trace(
        go.Bar(
            x=["Full pooled − local-only"],
            y=[baseline["effect"]],
            error_y={
                "type": "data",
                "symmetric": False,
                "array": [baseline["ci_high"] - baseline["effect"]],
                "arrayminus": [baseline["effect"] - baseline["ci_low"]],
            },
            marker_color="#4C78A8" if baseline["effect"] <= 0 else "#E45756",
            hovertemplate="ΔRMSE = %{y:.3f}<extra></extra>",
        )
    )
    baseline_figure.add_hline(y=0, line_dash="dash", line_color="#555")
    baseline_figure.update_layout(
        title="Does pooling help before pruning?",
        yaxis_title="RMSE(full pooled) − RMSE(local-only)",
        height=340,
        margin={"l": 50, "r": 20, "t": 55, "b": 50},
        showlegend=False,
    )
    message = (
        "Negative means the pooled model has lower error; "
        "positive means local-only has lower error."
    )
    mo.vstack([mo.md("## 1. Start with the pooling decision"), baseline_figure, mo.md(message)])
    return (baseline,)


@app.cell
def _(effects):
    ranked = effects.loc[
        effects["comparison_kind"].isin(["ranked_vs_random_median", "ranked_vs_matched_median"])
    ].copy()
    ranked["pool_label"] = (
        ranked["eligible_pool"].astype(str) + " · " + ranked["stratum"].astype(str)
    )
    pool_labels = sorted(ranked["pool_label"].unique().tolist())
    batch_sizes = sorted(ranked["batch_size"].dropna().astype(int).unique().tolist())
    return batch_sizes, pool_labels, ranked


@app.cell
def _(batch_sizes, mo, pool_labels):
    pool_selector = mo.ui.dropdown(
        options=pool_labels,
        value="native_only · target" if "native_only · target" in pool_labels else pool_labels[0],
        label="Candidate pool",
    )
    comparator_selector = mo.ui.radio(
        options={
            "Random deletion": "ranked_vs_random_median",
            "Source + label-quantile control": "ranked_vs_matched_median",
        },
        value="Random deletion",
        label="Comparator",
    )
    batch_selector = mo.ui.dropdown(
        options={f"Remove {size} molecules": size for size in batch_sizes},
        value=f"Remove {max(batch_sizes)} molecules",
        label="Inspect batch size",
    )
    mo.hstack([pool_selector, comparator_selector, batch_selector], justify="start", gap=2)
    return batch_selector, comparator_selector, pool_selector


@app.cell
def _(go, np, pool_selector, ranked):
    chosen_pool = pool_selector.value
    curve_rows = ranked.loc[ranked["pool_label"].eq(chosen_pool)].copy()
    pruning_figure = go.Figure()
    palette = {"high": "#E45756", "low": "#54A24B"}
    dash = {
        "ranked_vs_random_median": "solid",
        "ranked_vs_matched_median": "dash",
    }
    labels = {
        "ranked_vs_random_median": "vs random median",
        "ranked_vs_matched_median": "vs stratified-control median",
    }
    for comparison_kind in dash:
        for direction in ("high", "low"):
            subset = curve_rows.loc[
                curve_rows["comparison_kind"].eq(comparison_kind)
                & curve_rows["direction"].eq(direction)
            ].sort_values("batch_size")
            if subset.empty:
                continue
            pruning_figure.add_trace(
                go.Scatter(
                    x=subset["batch_size"],
                    y=subset["effect"],
                    error_y={
                        "type": "data",
                        "symmetric": False,
                        "array": subset["ci_high"] - subset["effect"],
                        "arrayminus": subset["effect"] - subset["ci_low"],
                    },
                    mode="lines+markers",
                    line={"color": palette[direction], "dash": dash[comparison_kind], "width": 3},
                    marker={"size": 9},
                    name=f"Remove {direction} score · {labels[comparison_kind]}",
                    customdata=np.stack(
                        [subset["n_control_draws"].fillna(0), subset["ci_low"], subset["ci_high"]],
                        axis=1,
                    ),
                    hovertemplate=(
                        "removed=%{x}<br>ΔRMSE=%{y:.3f}<br>95% interval=[%{customdata[1]:.3f}, "
                        "%{customdata[2]:.3f}]<br>control draws=%{customdata[0]}<extra></extra>"
                    ),
                )
            )
    pruning_figure.add_hline(y=0, line_dash="dot", line_color="#555")
    pruning_figure.update_layout(
        title=f"Precomputed deletion experiments: {chosen_pool}",
        xaxis_title="Molecules removed from full pooled training",
        yaxis_title="Control median RMSE − ranked-pruning RMSE",
        legend={"orientation": "h", "y": -0.28},
        height=490,
        margin={"l": 60, "r": 20, "t": 60, "b": 120},
    )
    pruning_figure
    return curve_rows, pruning_figure


@app.cell
def _(arm_manifest, batch_selector, comparator_selector, curve_rows, mo, pool_selector):
    selected_rows = curve_rows.loc[
        curve_rows["comparison_kind"].eq(comparator_selector.value)
        & curve_rows["batch_size"].eq(batch_selector.value)
    ]
    if selected_rows.empty:
        raise ValueError("selected pool has no completed comparison at this batch size")
    selected_effects = selected_rows.set_index("direction")
    selected_arm_ids = selected_effects["condition_b"].to_dict()
    arm_rows = arm_manifest.loc[arm_manifest["arm_id"].isin(selected_arm_ids.values())].copy()
    arm_rows["direction"] = arm_rows["arm_id"].map(
        {arm_id: direction for direction, arm_id in selected_arm_ids.items()}
    )
    evidence_table = selected_rows.loc[
        :, ["direction", "effect", "ci_low", "ci_high", "n_control_draws", "control_policy"]
    ].sort_values("direction")
    interpretation = (
        "Positive ΔRMSE means the ranked deletion outperformed the median control deletion. "
        "This is a comparison of already-fitted models, not an on-the-fly refit."
    )
    mo.vstack(
        [
            mo.md("## 2. Choose a surgical intervention"),
            mo.md(
                f"**{pool_selector.value}; remove {batch_selector.value} molecules; "
                f"{comparator_selector.value.replace('_', ' ')}.** {interpretation}"
            ),
            mo.ui.table(evidence_table, selection=None),
        ]
    )
    return arm_rows, selected_arm_ids, selected_effects


@app.cell
def _(arm_manifest, arm_rows, batch_selector, go, pool_selector):
    selected_pool, selected_stratum = pool_selector.value.split(" · ", maxsplit=1)
    candidate_arms = arm_manifest.loc[
        arm_manifest["eligible_pool"].eq(selected_pool)
        & arm_manifest["stratum"].eq(selected_stratum)
        & arm_manifest["batch_size"].eq(batch_selector.value)
    ].copy()
    arm_label_means = (
        candidate_arms.groupby(["arm_id", "policy", "direction"], dropna=False)["model_target"]
        .mean()
        .reset_index(name="removed_label_mean")
    )
    label_figure = go.Figure()
    style = {
        "boostin_high": "#E45756",
        "boostin_low": "#54A24B",
        "random": "#888888",
        "label_source_matched_high": "#F2A541",
        "label_source_matched_low": "#72B7B2",
    }
    for policy, policy_rows in arm_label_means.groupby("policy", sort=False):
        label_figure.add_trace(
            go.Box(
                y=policy_rows["removed_label_mean"],
                name=str(policy).replace("_", " "),
                marker_color=style.get(str(policy), "#777777"),
                boxpoints="all",
                jitter=0.25,
                pointpos=0,
                hovertemplate="mean removed log clearance=%{y:.3f}<extra></extra>",
            )
        )
    label_figure.update_layout(
        title="Are the selected molecules label-matched to their controls?",
        yaxis_title="Mean log HLM clearance of molecules removed per arm",
        height=420,
        margin={"l": 60, "r": 20, "t": 60, "b": 100},
    )
    label_figure
    return arm_label_means, label_figure


@app.cell
def _(go, np, predictions, selected_arm_ids):
    visible_conditions = {"full_pooled", *selected_arm_ids.values()}
    prediction_view = predictions.loc[predictions["condition"].isin(visible_conditions)].copy()
    prediction_means = (
        prediction_view.groupby(["condition", "evaluation_id"], as_index=False)
        .agg(observed=("observed", "first"), predicted=("predicted", "mean"))
        .assign(residual=lambda frame: frame["observed"] - frame["predicted"])
    )
    calibration_figure = go.Figure()
    condition_labels = {"full_pooled": "Full pooled"} | {
        arm_id: f"Remove {direction} score" for direction, arm_id in selected_arm_ids.items()
    }
    colours = {"full_pooled": "#4C78A8"} | {
        arm_id: {"high": "#E45756", "low": "#54A24B"}[direction]
        for direction, arm_id in selected_arm_ids.items()
    }
    limits = [prediction_means["observed"].min(), prediction_means["observed"].max()]
    for condition, condition_rows in prediction_means.groupby("condition", sort=False):
        calibration_figure.add_trace(
            go.Scattergl(
                x=condition_rows["observed"],
                y=condition_rows["predicted"],
                mode="markers",
                name=condition_labels[condition],
                marker={"size": 6, "opacity": 0.55, "color": colours[condition]},
                hovertemplate="observed=%{x:.3f}<br>predicted=%{y:.3f}<extra></extra>",
            )
        )
    calibration_figure.add_trace(
        go.Scatter(
            x=limits,
            y=limits,
            mode="lines",
            name="Ideal calibration",
            line={"color": "#222", "dash": "dash"},
        )
    )
    calibration_figure.update_layout(
        title="Prediction calibration on the untouched evaluation surface",
        xaxis_title="Observed log HLM clearance",
        yaxis_title="Mean prediction across model seeds",
        height=480,
        margin={"l": 60, "r": 20, "t": 60, "b": 55},
    )
    return calibration_figure, prediction_means


@app.cell
def _(go, np, prediction_means):
    decomposition = (
        prediction_means.groupby("condition")["residual"]
        .agg(
            bias="mean",
            centered_rmse=lambda series: float(np.sqrt(np.mean((series - series.mean()) ** 2))),
            rmse=lambda series: float(np.sqrt(np.mean(series**2))),
        )
        .reset_index()
    )
    decomposition_long = decomposition.melt(
        id_vars="condition",
        value_vars=["bias", "centered_rmse", "rmse"],
        var_name="component",
        value_name="value",
    )
    decomposition_figure = go.Figure()
    for component, component_rows in decomposition_long.groupby("component", sort=False):
        decomposition_figure.add_trace(
            go.Bar(
                name=component.replace("_", " "),
                x=component_rows["condition"],
                y=component_rows["value"],
            )
        )
    decomposition_figure.update_layout(
        title="Does pruning improve calibration, centred error, or both?",
        barmode="group",
        yaxis_title="Log HLM clearance error",
        height=390,
        margin={"l": 60, "r": 20, "t": 60, "b": 100},
    )
    decomposition_figure
    return decomposition


@app.cell
def _(calibration_figure, decomposition, decomposition_figure, mo):
    mo.vstack(
        [
            mo.md("## 3. Audit the mechanism"),
            mo.md(
                "An RMSE improvement can arise from a better molecular relationship, a global "
                "calibration shift, or both. Read the scatter and decomposition together."
            ),
            mo.hstack([calibration_figure, decomposition_figure], widths="equal"),
            mo.ui.table(decomposition.round(4), selection=None),
        ]
    )
    return


@app.cell
def _(mo, scores):
    source_selector = mo.ui.dropdown(
        options=["All sources", *sorted(scores["source"].unique().tolist())],
        value="All sources",
        label="Inspect score source",
    )
    source_selector
    return (source_selector,)


@app.cell
def _(go, make_subplots, np, scores, source_selector):
    score_view = (
        scores
        if source_selector.value == "All sources"
        else scores.loc[scores["source"].eq(source_selector.value)]
    )
    label_rho = score_view[["raw_score_mean", "model_target"]].corr(method="spearman").iloc[0, 1]
    similarity_rho = (
        score_view[["raw_score_mean", "max_tanimoto_to_selection"]]
        .corr(method="spearman")
        .iloc[0, 1]
    )
    score_figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Score versus measured label (Spearman ρ = {label_rho:.2f})",
            f"Score versus selection similarity (Spearman ρ = {similarity_rho:.2f})",
        ),
    )
    source_colours = {"expansionrx": "#4C78A8", "biogen": "#F58518", "polaris": "#54A24B"}
    for source, source_rows in score_view.groupby("source", sort=False):
        common = {
            "mode": "markers",
            "name": str(source),
            "marker": {
                "size": 5,
                "opacity": 0.45,
                "color": source_colours.get(str(source), "#777"),
            },
            "hovertemplate": (
                "source=" + str(source) + "<br>label=%{customdata[0]:.3f}<br>"
                "score=%{y:.4f}<br>similarity=%{customdata[1]:.3f}<extra></extra>"
            ),
            "customdata": np.stack(
                [
                    source_rows["model_target"].to_numpy(),
                    source_rows["max_tanimoto_to_selection"].to_numpy(),
                ],
                axis=1,
            ),
        }
        score_figure.add_trace(
            go.Scattergl(x=source_rows["model_target"], y=source_rows["raw_score_mean"], **common),
            row=1,
            col=1,
        )
        score_figure.add_trace(
            go.Scattergl(
                x=source_rows["max_tanimoto_to_selection"],
                y=source_rows["raw_score_mean"],
                showlegend=False,
                **common,
            ),
            row=1,
            col=2,
        )
    score_figure.update_xaxes(title_text="Measured log HLM clearance", row=1, col=1)
    score_figure.update_xaxes(
        title_text="Max Tanimoto similarity to selection surface", row=1, col=2
    )
    score_figure.update_yaxes(title_text="Mean BoostIn score", row=1, col=1)
    score_figure.update_layout(height=510, margin={"l": 60, "r": 20, "t": 80, "b": 55})
    score_figure
    return label_rho, score_view, similarity_rho


@app.cell
def _(label_rho, mo, similarity_rho):
    mo.vstack(
        [
            mo.md("## 4. What is BoostIn actually ranking?"),
            mo.callout(
                mo.md(
                    f"For this view, score–label Spearman ρ is **{label_rho:.2f}**, while "
                    f"score–selection-similarity ρ is **{similarity_rho:.2f}**. A high score–label "
                    "association means a pruning result can be partly a label-level calibration "
                    "effect, "
                    "not automatically a chemical-domain discovery."
                ),
                kind="warn",
            ),
        ]
    )
    return


@app.cell
def _(dedent, mo):
    mo.md(
        dedent(
            """
            ## Interpretation guardrails

            - **High score is beneficial under the fitted deletion objective.** If removing it harms
              error, that validates the direction of the conditional attribution—not a universal
              molecular property.
            - **Controls are source + within-source label-quantile stratified.** They are not exact
              continuous-label matches; use the removed-label plot before making a chemical claim.
            - **No adaptive refitting occurs here.** Every visible value comes from immutable
              experiment artifacts and the held-out evaluation surface.
            - **This is HLM clearance only.** Generalisation to other endpoints, model families, or
              assay-harmonised targets remains future work.
            """
        )
    )
    return


if __name__ == "__main__":
    app.run()
