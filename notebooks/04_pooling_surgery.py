# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "anywidget>=0.11",
#   "marimo>=0.25",
#   "numpy",
#   "pandas",
#   "plotly>=7",
#   "pyarrow",
#   "pyyaml",
#   "rdkit>=2024.9",
# ]
# ///
"""Pooling Under the Knife: a cross-target story of conditional influence pruning.

Read-only analysis of completed BoostIn deletion runs. The notebook recomputes paired
effect intervals from frozen per-molecule predictions to repair a row-alignment defect in
the persisted effects tables; it never fits a model, recomputes attribution, or changes a
raw artifact.

The interactive molecule grids are rendered by the custom ``MoleculeGrid`` anywidget in
``influence_pruning.molgrid``; the notebook degrades to a static fallback if anywidget is
unavailable.
"""

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="full")


@app.cell
def _():
    import contextlib
    import json
    import pathlib
    import sys
    from textwrap import dedent

    import marimo as mo

    _roots = [pathlib.Path.cwd()]
    _notebook_root = mo.notebook_dir()
    if _notebook_root is not None and _notebook_root not in _roots:
        _roots.append(_notebook_root)

    # molab bootstrap: restore the self-extracting upload bundle when running outside the
    # repository checkout (no-op locally, where the package is on the path). `publish_molab.py`
    # fills in the release URL, and the download is persistently cached so only the first molab
    # session pays for the transfer.
    MOLAB_BUNDLE_URL = (
        "https://github.com/KalenJosifovski/influence-pruning/releases/latest/download/molab_bundle.zip"
    )
    _has_local_package = any(
        (_root / "influence_pruning").is_dir() or (_root / "src" / "influence_pruning").is_dir()
        for _root in _roots
    )
    _target_zip = _roots[0] / "molab_bundle.zip"
    if MOLAB_BUNDLE_URL and not _has_local_package and not _target_zip.is_file():
        import urllib.request

        @mo.persistent_cache
        def _fetch_bundle(url: str) -> bytes:
            with urllib.request.urlopen(url, timeout=120) as response:
                return response.read()

        with contextlib.suppress(OSError):
            _target_zip.write_bytes(_fetch_bundle(MOLAB_BUNDLE_URL))

    for _root in _roots:
        _bundle_zip = _root / "molab_bundle.zip"
        if _bundle_zip.is_file() and not (_root / "src" / "influence_pruning").is_dir():
            import zipfile

            with zipfile.ZipFile(_bundle_zip) as _archive:
                _archive.extractall(_root)

    def _add_to_path(candidate):
        if (candidate / "influence_pruning").is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))

    for _root in _roots:
        _add_to_path(_root)
        _add_to_path(_root / "src")

    _package_found = _has_local_package or any(
        (_root / "influence_pruning").is_dir() or (_root / "src" / "influence_pruning").is_dir()
        for _root in _roots
    )
    if not _package_found:
        raise ModuleNotFoundError(
            "influence_pruning is not importable. On molab, upload molab_bundle.zip through "
            "the Files panel (persistent), or confirm that MOLAB_BUNDLE_URL is reachable from "
            "this session."
        )

    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from rdkit import Chem
    from rdkit.Chem import Draw, rdFingerprintGenerator

    from influence_pruning.artifacts import verify_run
    from influence_pruning.errors import ArtifactError

    return (
        ArtifactError,
        Chem,
        Draw,
        dedent,
        go,
        json,
        make_subplots,
        mo,
        np,
        pathlib,
        pd,
        px,
        rdFingerprintGenerator,
        verify_run,
    )


@app.cell
def _():
    from influence_pruning.molgrid import (
        MoleculeGrid,
        ScaffoldBars,
        molecule_records,
        scaffold_of,
        scaffold_summary,
        selection_ids,
    )

    molgrid_available = MoleculeGrid is not None
    return (
        MoleculeGrid,
        ScaffoldBars,
        molgrid_available,
        molecule_records,
        scaffold_of,
        scaffold_summary,
        selection_ids,
    )


@app.cell
def _():
    TARGET_ORDER = ["expansionrx", "biogen", "polaris"]
    TARGET_STYLE = {
        "expansionrx": {"label": "ExpansionRx", "colour": "#4C78A8"},
        "biogen": {"label": "Biogen", "colour": "#F58518"},
        "polaris": {"label": "Polaris", "colour": "#54A24B"},
    }
    SOURCE_STYLE = {
        "expansionrx": {"label": "ExpansionRx", "colour": "#4C78A8", "symbol": "circle"},
        "biogen": {"label": "Biogen", "colour": "#F58518", "symbol": "square"},
        "polaris": {"label": "Polaris", "colour": "#54A24B", "symbol": "diamond"},
    }
    POOL_LABELS = {
        "native_only": "Native target molecules",
        "mixed_pool": "Mixed pool",
        "donor_only": "External donors",
    }
    STATUS_COLOURS = {
        "interval excludes zero": "#2E7D32",
        "direction consistent": "#C97A00",
        "inconclusive": "#9E9E9E",
        "pooled better": "#4C78A8",
        "local-only better": "#E45756",
    }

    def style_figure(figure, title, height=460, **overrides):
        layout = dict(
            template="plotly_white",
            title={"text": title, "x": 0.01, "xanchor": "left", "font": {"size": 17}},
            height=height,
            margin={"l": 70, "r": 30, "t": 70, "b": 60},
            font={
                "family": "Inter, -apple-system, BlinkMacSystemFont, sans-serif",
                "size": 12.5,
            },
            hoverlabel={"font_size": 12},
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        )
        layout.update(overrides)
        figure.update_layout(**layout)
        return figure

    return (
        POOL_LABELS,
        SOURCE_STYLE,
        STATUS_COLOURS,
        TARGET_ORDER,
        TARGET_STYLE,
        style_figure,
    )


@app.cell
def _(json, mo, np, pathlib, pd):
    """Vectorised cluster bootstrap that reproduces the project's RNG stream exactly."""

    def _parse_bootstrap(field):
        parts = dict(item.split("=") for item in str(field).split(";") if "=" in item)
        return int(parts["n_resamples"]), int(parts["seed"])

    def _cluster_weights(clusters, n_resamples, seed):
        labels = np.asarray(clusters).astype(int)
        unique_ids = np.unique(labels)
        positions = np.searchsorted(unique_ids, labels)
        rng = np.random.default_rng(seed)
        probabilities = np.full(unique_ids.size, 1.0 / unique_ids.size)
        multiplicities = np.stack(
            [rng.multinomial(unique_ids.size, probabilities) for _ in range(n_resamples)]
        )
        return multiplicities[:, positions]

    def _boot_delta(observed, predictions_a, predictions_b, clusters, n_resamples, seed):
        errors_a = (observed[:, None] - predictions_a) ** 2
        errors_b = (observed[:, None] - predictions_b) ** 2

        def statistic(weights):
            total = weights.sum(axis=1, keepdims=True)
            rmse_a = np.sqrt((weights @ errors_a) / total).mean(axis=-1)
            rmse_b = np.sqrt((weights @ errors_b) / total).mean(axis=-1)
            return rmse_a - rmse_b

        point = float(statistic(np.ones((1, observed.size)))[0])
        return point, statistic(_cluster_weights(clusters, n_resamples, seed))

    def _boot_median_control(observed, ranked, controls, clusters, n_resamples, seed):
        errors_ranked = (observed[:, None] - ranked) ** 2
        n_draws, n_molecules, n_seeds = controls.shape
        error_controls = (observed[None, :, None] - controls) ** 2
        flat_controls = error_controls.transpose(1, 0, 2).reshape(n_molecules, n_draws * n_seeds)

        def statistic(weights):
            total = weights.sum(axis=1, keepdims=True)
            ranked_rmse = np.sqrt((weights @ errors_ranked) / total).mean(axis=-1)
            control_rmse = np.sqrt(
                ((weights @ flat_controls) / total).reshape(-1, n_draws, n_seeds)
            ).mean(axis=-1)
            return np.median(control_rmse, axis=-1) - ranked_rmse

        point = float(statistic(np.ones((1, observed.size)))[0])
        return point, statistic(_cluster_weights(clusters, n_resamples, seed))

    @mo.cache
    def corrected_effects(run_dir: str, stamp: float):
        """Recompute every paired effect with targets, clusters, and predictions aligned.

        The persisted effects tables mix the evaluation frame's row order with identifier-sorted
        prediction matrices whenever the evaluation partition is not already sorted. Predictions
        are frozen and row-wise correct, so the repair is a bootstrap recomputation, never a
        refit. A cached copy is reused from ``analysis/effects_recomputed`` when present.
        """
        directory = pathlib.Path(run_dir)
        analysis_root = pathlib.Path("analysis/effects_recomputed")
        if not analysis_root.parent.is_dir():
            _notebook_root = mo.notebook_dir()
            if _notebook_root is not None:
                analysis_root = _notebook_root / "analysis/effects_recomputed"
        destination = analysis_root / directory.name
        if (destination / "paired_effects.parquet").is_file() and (
            destination / "bootstrap_draws.parquet"
        ).is_file():
            effects = pd.read_parquet(destination / "paired_effects.parquet")
            draws = pd.read_parquet(destination / "bootstrap_draws.parquet")
        else:
            predictions = pd.read_parquet(directory / "predictions.parquet")
            manifest = pd.read_parquet(directory / "partition_manifest.parquet")
            persisted = pd.read_parquet(directory / "paired_effects.parquet")
            blocks = manifest.loc[manifest["role"].eq("evaluation")]
            order = blocks["candidate_id"].astype(str).to_numpy()
            observed = blocks["model_target"].to_numpy(dtype=float)
            clusters = (
                predictions.loc[predictions["condition"].eq("full_pooled")]
                .drop_duplicates("evaluation_id")
                .set_index("evaluation_id")
                .loc[order, "cluster_id"]
                .to_numpy()
            )
            seeds = sorted(predictions["seed"].unique().tolist())
            matrices = {}
            for condition, group in predictions.groupby("condition", sort=False):
                matrices[str(condition)] = (
                    group.pivot(index="evaluation_id", columns="seed", values="predicted")
                    .loc[order, seeds]
                    .to_numpy(dtype=float)
                )
            rows, draw_frames = [], []
            for _, record in persisted.iterrows():
                n_resamples, seed = _parse_bootstrap(record["settings"])
                if record["comparison_kind"] in ("baseline", "full_vs_pruned"):
                    point, values = _boot_delta(
                        observed,
                        matrices[record["condition_a"]],
                        matrices[record["condition_b"]],
                        clusters,
                        n_resamples,
                        seed,
                    )
                else:
                    controls = np.stack(
                        [matrices[control] for control in json.loads(record["control_arm_ids"])],
                        axis=0,
                    )
                    point, values = _boot_median_control(
                        observed,
                        matrices[record["condition_b"]],
                        controls,
                        clusters,
                        n_resamples,
                        seed,
                    )
                updated = record.to_dict()
                updated.update(
                    effect=point,
                    ci_low=float(np.percentile(values, 2.5)),
                    ci_high=float(np.percentile(values, 97.5)),
                )
                rows.append(updated)
                draw_frames.append(
                    pd.DataFrame(
                        {
                            "comparison_id": record["comparison_id"],
                            "resample_index": np.arange(values.size, dtype=int),
                            "effect": values,
                        }
                    )
                )
            effects = pd.DataFrame(rows).loc[:, persisted.columns]
            draws = pd.concat(draw_frames, ignore_index=True)
            try:
                destination.mkdir(parents=True, exist_ok=True)
                effects.to_parquet(destination / "paired_effects.parquet")
                draws.to_parquet(destination / "bootstrap_draws.parquet")
            except OSError:
                pass
        ranked_kinds = ("ranked_vs_random_median", "ranked_vs_matched_median")
        ranked_ids = set(
            effects.loc[effects["comparison_kind"].isin(ranked_kinds), "comparison_id"]
        )
        return effects, draws.loc[draws["comparison_id"].isin(ranked_ids)].reset_index(drop=True)

    return (corrected_effects,)


@app.cell
def _(ArtifactError, corrected_effects, mo, pathlib, pd, verify_run):
    def _load_runs():
        run_root = pathlib.Path("results/boostin_pruning")
        if not run_root.is_dir():
            _notebook_root = mo.notebook_dir()
            if _notebook_root is not None:
                run_root = _notebook_root / "results/boostin_pruning"
        completed = sorted(
            path.parent
            for path in run_root.glob("*/run_facts.json")
            if (path.parent / "COMPLETED").is_file()
        )
        if not completed:
            raise FileNotFoundError(
                "no completed runs under results/boostin_pruning; run the study first"
            )
        payloads = {}
        for directory in completed:
            try:
                facts = verify_run(directory)
            except ArtifactError:
                continue
            if facts.get("dry_run", False):
                continue
            target = str(facts["target"])
            stamp = (directory / "predictions.parquet").stat().st_mtime
            effects, draws = corrected_effects(str(directory), stamp)
            metrics = pd.read_parquet(directory / "seed_metrics.parquet")
            baseline = effects.loc[effects["comparison_kind"].eq("baseline")].iloc[0]
            payloads[target] = {
                "directory": directory,
                "facts": facts,
                "manifest": pd.read_parquet(directory / "partition_manifest.parquet"),
                "scores": pd.read_parquet(directory / "boostin_scores.parquet").drop_duplicates(
                    "candidate_id"
                ),
                "effects": effects,
                "draws": draws,
                "arm_manifest": pd.read_parquet(directory / "arm_manifest.parquet"),
                "baseline": baseline,
                "full_rmse": float(
                    metrics.loc[metrics["condition"].eq("full_pooled"), "rmse"].mean()
                ),
                "local_rmse": float(
                    metrics.loc[metrics["condition"].eq("local_only"), "rmse"].mean()
                ),
            }
        if not payloads:
            raise FileNotFoundError("only dry-run artifacts were found; no scientific run to show")
        return payloads

    runs = _load_runs()
    return (runs,)


@app.cell
def _(TARGET_ORDER, TARGET_STYLE, mo, runs):
    stat_cards = []
    for _target in TARGET_ORDER:
        if _target not in runs:
            continue
        _baseline = runs[_target]["baseline"]
        stat_cards.append(
            mo.stat(
                value=f"{_baseline['effect']:+.3f}",
                label=TARGET_STYLE[_target]["label"],
                caption=(
                    f"pooled − local · {int(_baseline['n_molecules'])} eval molecules · "
                    f"{int(_baseline['n_clusters'])} clusters"
                ),
            )
        )
    _total_fits = sum(int(p["facts"].get("n_fits", 0)) for p in runs.values())
    _total_hours = (
        sum(float(p["facts"].get("runtime_seconds", 0.0)) for p in runs.values()) / 3600.0
    )
    mo.vstack(
        [
            mo.md(
                """
                # Pooling Under the Knife

                **Can an influence ranking tell you which pooled molecules to keep?** Three
                target rotations share two external donors. A full pooled model is fitted, every
                training molecule is scored with BoostIn against a target-like selection surface,
                and ranked batches are deleted and refit. This notebook tells the resulting story
                from the frozen artifacts.
                """
            ),
            mo.hstack(stat_cards, justify="start", gap=1.5),
            mo.md(
                f"*{len(runs)} rotations · {_total_fits:,} model fits already spent · "
                f"{_total_hours:.1f} machine-hours · every number below is read-only or a "
                "derived bootstrap summary of frozen predictions.*"
            ),
            mo.callout(
                mo.md(
                    "**How to read the signs.** For the pooling bars, "
                    "`RMSE(full pooled) − RMSE(local-only)`: **negative means pooling helped**. "
                    "For the surgery panels, `RMSE(high-influence removal) − "
                    "RMSE(low-influence removal)`: **positive means the attribution direction "
                    "was correct** (removing supposedly helpful molecules hurt more than "
                    "removing supposedly harmful ones)."
                ),
                kind="info",
            ),
        ]
    )
    return


@app.cell
def _(TARGET_ORDER, TARGET_STYLE, mo, runs):
    available = [target for target in TARGET_ORDER if target in runs]
    rotation = mo.ui.dropdown(
        options={TARGET_STYLE[target]["label"]: target for target in available},
        value=TARGET_STYLE[available[0]]["label"],
        label="Deep-dive rotation",
    )
    rotation
    return (rotation,)


@app.cell
def _(dedent, mo):
    mo.md(
        dedent(
            """
            ## 1 · The verdict at a glance

            One row per claim, one column per target rotation. The intervals are percentile
            intervals from resampling whole evaluation clusters (2,000 draws) of the *same*
            evaluation set; they describe evaluation-set uncertainty, not independent replication.
            **Green** marks intervals that exclude zero; **amber** marks intervals that include
            zero but where at least four in five resamples agree on the sign; **grey** is
            inconclusive at this evaluation-set size. The matched controls preserve source
            composition and within-source label quantile bins, not continuous label values.
            """
        )
    )
    return


@app.cell
def _(POOL_LABELS, STATUS_COLOURS, TARGET_ORDER, TARGET_STYLE, mo, np, runs):
    def _contract():
        def contrast(draws, effects, pool, stratum, batch_size, kind):
            subset = effects.loc[
                effects["comparison_kind"].eq(kind)
                & effects["eligible_pool"].eq(pool)
                & effects["stratum"].eq(stratum)
                & effects["batch_size"].eq(batch_size)
            ]
            high = subset.loc[subset["direction"].eq("high"), "comparison_id"]
            low = subset.loc[subset["direction"].eq("low"), "comparison_id"]
            if high.empty or low.empty:
                return None
            pivot = draws.pivot(index="resample_index", columns="comparison_id", values="effect")
            if high.iloc[0] not in pivot.columns or low.iloc[0] not in pivot.columns:
                return None
            delta = pivot[low.iloc[0]] - pivot[high.iloc[0]]
            return {
                "median": float(delta.median()),
                "lo": float(np.percentile(delta, 2.5)),
                "hi": float(np.percentile(delta, 97.5)),
                "p": float((delta > 0).mean()),
            }

        def best(draws, effects, kind, candidates):
            sizes = sorted(
                effects.loc[effects["comparison_kind"].eq(kind), "batch_size"].dropna().unique(),
                reverse=True,
            )
            for pool, stratum in candidates:
                for size in sizes:
                    block = contrast(draws, effects, pool, stratum, size, kind)
                    if block is not None:
                        return block, pool, stratum, int(size)
            return None, None, None, None

        def status(block):
            if block is None:
                return "inconclusive"
            if block["lo"] > 0 or block["hi"] < 0:
                return "interval excludes zero"
            if block["p"] >= 0.80 or block["p"] <= 0.20:
                return "direction consistent"
            return "inconclusive"

        def evidence(block, pool, size):
            if block is None:
                return "no completed contrast"
            return (
                f"Δ {block['median']:+.3f} [{block['lo']:+.3f}, {block['hi']:+.3f}] · "
                f"{block['p'] * 100:.0f}% of resamples agree · {POOL_LABELS[pool]} k={size}"
            )

        card = {}
        for target in TARGET_ORDER:
            if target not in runs:
                continue
            payload = runs[target]
            effects, draws = payload["effects"], payload["draws"]
            baseline = payload["baseline"]
            interval = (
                f"Δ {baseline['effect']:+.3f} "
                f"[{baseline['ci_low']:+.3f}, {baseline['ci_high']:+.3f}]"
            )
            if baseline["ci_high"] < 0:
                pooling = ("pooled better", "interval excludes zero", interval)
            elif baseline["ci_low"] > 0:
                pooling = ("local-only better", "interval excludes zero", interval)
            elif baseline["effect"] < 0:
                pooling = (
                    "pooled better",
                    "inconclusive",
                    interval + " · point estimate only",
                )
            else:
                pooling = (
                    "local-only better",
                    "inconclusive",
                    interval + " · point estimate only",
                )
            native, native_pool, _, native_size = best(
                draws, effects, "ranked_vs_random_median", [("native_only", "target")]
            )
            mixed, mixed_pool, _, mixed_size = best(
                draws, effects, "ranked_vs_random_median", [("mixed_pool", "all")]
            )
            matched, matched_pool, _, matched_size = best(
                draws,
                effects,
                "ranked_vs_matched_median",
                [("native_only", "target"), ("mixed_pool", "all")],
            )
            card[target] = {
                "pooling": pooling,
                "native": (native, native_pool, native_size),
                "mixed": (mixed, mixed_pool, mixed_size),
                "matched": (matched, matched_pool, matched_size),
            }

        def badge(label, status_key, evidence_text):
            colour = STATUS_COLOURS[status_key]
            return (
                f"<div style='font-size:11.5px;font-weight:700;letter-spacing:.03em;"
                f"color:{colour};'>{label.upper()}</div>"
                f"<div style='font-size:11px;color:#444;margin-top:2px;'>"
                f"{evidence_text}</div>"
            )

        claim_rows = [
            ("Pooling vs local-only", lambda block: block["pooling"]),
            (
                "Native high-vs-low deletion contrast",
                lambda block: (
                    status(block["native"][0]),
                    status(block["native"][0]),
                    evidence(block["native"][0], block["native"][1], block["native"][2]),
                ),
            ),
            (
                "Mixed high-vs-low deletion contrast",
                lambda block: (
                    status(block["mixed"][0]),
                    status(block["mixed"][0]),
                    evidence(block["mixed"][0], block["mixed"][1], block["mixed"][2]),
                ),
            ),
            (
                "Direction under the source + label-quantile control",
                lambda block: (
                    status(block["matched"][0]),
                    status(block["matched"][0]),
                    evidence(block["matched"][0], block["matched"][1], block["matched"][2]),
                ),
            ),
        ]
        header = "".join(
            f"<th style='padding:10px 14px;text-align:left;font-size:12.5px;color:#666;"
            f"border-bottom:1px solid #DDD;'>{TARGET_STYLE[target]['label']}</th>"
            for target in card
        )
        body = ""
        for claim, extract in claim_rows:
            body += (
                f"<tr><td style='padding:11px 14px;font-size:12.5px;color:#222;"
                f"border-bottom:1px solid #EEE;'>{claim}</td>"
            )
            for target in card:
                label, status_key, evidence_text = extract(card[target])
                body += (
                    "<td style='padding:11px 14px;border-bottom:1px solid #EEE;"
                    "border-left:1px solid #F2F2F2;'>"
                    + badge(label, status_key, evidence_text)
                    + "</td>"
                )
            body += "</tr>"
        return mo.Html(
            "<div style='border:1px solid #E6E6E6;border-radius:12px;overflow:hidden;"
            "box-shadow:0 1px 4px rgba(0,0,0,0.05);'>"
            "<table style='border-collapse:collapse;width:100%;background:white;'>"
            "<thead><tr><th style='padding:10px 14px;text-align:left;font-size:12.5px;"
            "color:#888;border-bottom:1px solid #DDD;'>Claim</th>"
            + header
            + "</tr></thead><tbody>"
            + body
            + "</tbody></table></div>"
        )

    scorecard_html = _contract()
    return (scorecard_html,)


@app.cell
def _(scorecard_html):
    scorecard_html
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            "The pattern is the story: the **pooling point estimate flips sign** across rotations "
            "and every pooling interval includes zero, while the **native high-vs-low deletion "
            "contrast interval excludes zero in all three rotations**. Under the coarse "
            "source + label-quantile control, the interval excludes zero only for ExpansionRx; "
            "Biogen and Polaris remain inconclusive at their evaluation-set sizes."
        ),
        kind="success",
    )
    return


@app.cell
def _(TARGET_ORDER, pd, runs):
    def _baseline_table():
        records = []
        for target in TARGET_ORDER:
            if target not in runs:
                continue
            payload = runs[target]
            baseline = payload["baseline"]
            records.append(
                {
                    "target": target,
                    "effect": float(baseline["effect"]),
                    "ci_low": float(baseline["ci_low"]),
                    "ci_high": float(baseline["ci_high"]),
                    "full_rmse": payload["full_rmse"],
                    "local_rmse": payload["local_rmse"],
                    "n_eval": int(baseline["n_molecules"]),
                    "n_clusters": int(baseline["n_clusters"]),
                }
            )
        return pd.DataFrame(records)

    baseline_table = _baseline_table()
    return (baseline_table,)


@app.cell
def _(TARGET_STYLE, baseline_table, go, style_figure):
    def _forest():
        labels = [TARGET_STYLE[target]["label"] for target in baseline_table["target"]]
        colours = ["#4C78A8" if effect < 0 else "#E45756" for effect in baseline_table["effect"]]
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=baseline_table["effect"],
                y=labels,
                mode="markers",
                marker={
                    "size": 17,
                    "color": colours,
                    "line": {"width": 1.5, "color": "white"},
                },
                error_x={
                    "type": "data",
                    "symmetric": False,
                    "array": baseline_table["ci_high"] - baseline_table["effect"],
                    "arrayminus": baseline_table["effect"] - baseline_table["ci_low"],
                    "thickness": 2.2,
                    "width": 7,
                    "color": "#555555",
                },
                customdata=baseline_table[["full_rmse", "local_rmse", "n_eval", "n_clusters"]],
                hovertemplate=(
                    "ΔRMSE = %{x:+.4f}<br>full pooled RMSE = %{customdata[0]:.3f}<br>"
                    "local-only RMSE = %{customdata[1]:.3f}<br>%{customdata[2]} eval "
                    "molecules · %{customdata[3]} clusters<extra></extra>"
                ),
                showlegend=False,
            )
        )
        figure.add_vline(x=0, line_dash="dash", line_color="#555555")
        for index, row in baseline_table.iterrows():
            figure.add_annotation(
                x=row["effect"],
                y=labels[index],
                yshift=24,
                showarrow=False,
                text=f"RMSE {row['full_rmse']:.3f} vs {row['local_rmse']:.3f}",
                font={"size": 10.5, "color": "#666666"},
            )
        return style_figure(
            figure,
            "Does pooling beat local-only? Point estimate and 95% cluster-bootstrap interval",
            height=370,
            xaxis_title="RMSE(full pooled) − RMSE(local-only)",
            xaxis={"zeroline": False, "gridcolor": "#EEEEEE"},
            yaxis={"gridcolor": "#EEEEEE"},
        )

    forest_figure = _forest()
    forest_figure
    return (forest_figure,)


@app.cell
def _(dedent, mo):
    mo.md(
        dedent(
            """
            ## 2 · The scalpel: dose–response of a ranked deletion

            Each point is `RMSE(high-influence removal) − RMSE(low-influence removal)` under the
            same random-control comparison, recomputed with paired resample draws of the same
            evaluation clusters. Positive values (green field) mean the ranking pointed the right
            way: removing molecules it called helpful hurt the target model more than removing the
            ones it called harmful. Marker area encodes the fraction of resamples that agree on
            the sign — a stability summary for this evaluation set, not a probability that the
            claim is true. **Each panel has its own y-scale**, so the donor panels are readable
            even though their effects are an order of magnitude smaller than the native panels.
            """
        )
    )
    return


@app.cell
def _(np, pd, runs, rotation):
    def _contrast_table():
        payload = runs[rotation.value]
        effects, draws = payload["effects"], payload["draws"]
        ranked = effects.loc[effects["comparison_kind"].eq("ranked_vs_random_median")]
        pivot = draws.pivot(index="resample_index", columns="comparison_id", values="effect")
        records = []
        for (pool, stratum, batch_size), group in ranked.groupby(
            ["eligible_pool", "stratum", "batch_size"], sort=True
        ):
            high = group.loc[group["direction"].eq("high"), "comparison_id"]
            low = group.loc[group["direction"].eq("low"), "comparison_id"]
            if high.empty or low.empty:
                continue
            if high.iloc[0] not in pivot.columns or low.iloc[0] not in pivot.columns:
                continue
            delta = pivot[low.iloc[0]] - pivot[high.iloc[0]]
            records.append(
                {
                    "pool": str(pool),
                    "stratum": str(stratum),
                    "batch_size": int(batch_size),
                    "median": float(delta.median()),
                    "ci_low": float(np.percentile(delta, 2.5)),
                    "ci_high": float(np.percentile(delta, 97.5)),
                    "p_agree": float((delta > 0).mean()),
                    "n_draws": int(delta.size),
                }
            )
        return pd.DataFrame(records).sort_values(["pool", "stratum", "batch_size"], kind="stable")

    contrast_table = _contrast_table()
    return (contrast_table,)


@app.cell
def _(
    POOL_LABELS,
    TARGET_STYLE,
    contrast_table,
    go,
    make_subplots,
    np,
    rotation,
    style_figure,
):
    def _contrast_figure():
        panel_specs = [
            ("native_only", "target", POOL_LABELS["native_only"]),
            ("mixed_pool", "all", POOL_LABELS["mixed_pool"]),
        ]
        donor_strata = (
            contrast_table.loc[contrast_table["pool"].eq("donor_only"), "stratum"]
            .drop_duplicates()
            .tolist()
        )
        for donor_stratum in donor_strata:
            panel_specs.append(("donor_only", donor_stratum, f"Donor only · {donor_stratum}"))
        n_cols = 3
        n_rows = int(np.ceil(len(panel_specs) / n_cols))
        figure = make_subplots(
            rows=n_rows,
            cols=n_cols,
            shared_yaxes=False,
            subplot_titles=[spec[2] for spec in panel_specs],
            vertical_spacing=0.16,
            horizontal_spacing=0.06,
        )
        panel_colours = {
            "native_only": "#4C78A8",
            "mixed_pool": "#7A5AA6",
            "donor_only": "#F58518",
        }
        shared_bottom = min(0.0, float(contrast_table["ci_low"].min()) * 1.05)
        for index, (pool, stratum, _) in enumerate(panel_specs):
            row, col = divmod(index, n_cols)
            subset = contrast_table.loc[
                contrast_table["pool"].eq(pool) & contrast_table["stratum"].eq(stratum)
            ].sort_values("batch_size")
            colour = panel_colours[pool]
            figure.add_trace(
                go.Scatter(
                    x=subset["batch_size"],
                    y=subset["median"],
                    mode="markers+lines",
                    line={"color": colour, "width": 2, "dash": "dot"},
                    marker={
                        "size": 6 + 11 * subset["p_agree"],
                        "color": colour,
                        "line": {"width": 1, "color": "white"},
                    },
                    error_y={
                        "type": "data",
                        "symmetric": False,
                        "array": subset["ci_high"] - subset["median"],
                        "arrayminus": subset["median"] - subset["ci_low"],
                        "thickness": 1.6,
                        "width": 5,
                        "color": colour,
                    },
                    customdata=subset[["p_agree", "ci_low", "ci_high", "n_draws"]],
                    hovertemplate=(
                        "k = %{x}<br>ΔRMSE (high − low) = %{y:+.4f}<br>"
                        "95% resample interval = [%{customdata[1]:+.4f}, "
                        "%{customdata[2]:+.4f}]<br>agreement = %{customdata[0]:.1%} of "
                        "%{customdata[3]} resamples<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=row + 1,
                col=col + 1,
            )
            highs = float(subset["ci_high"].max())
            top = max(highs * 1.15, 0.002) if pool == "donor_only" else max(0.2, highs * 1.05)
            figure.update_yaxes(range=[shared_bottom, top], row=row + 1, col=col + 1)
            figure.add_hline(y=0, line_dash="dash", line_color="#888888", row=row + 1, col=col + 1)
            figure.add_hrect(
                y0=0,
                y1=top,
                fillcolor="#54A24B",
                opacity=0.05,
                line_width=0,
                row=row + 1,
                col=col + 1,
            )
        for index in range(len(panel_specs), n_rows * n_cols):
            row, col = divmod(index, n_cols)
            figure.update_xaxes(visible=False, row=row + 1, col=col + 1)
            figure.update_yaxes(visible=False, row=row + 1, col=col + 1)
        style_figure(
            figure,
            f"Surgical dose–response · {TARGET_STYLE[rotation.value]['label']} target",
            height=310 * n_rows,
            xaxis_title="Molecules removed per arm",
            hovermode="closest",
            margin={"l": 95, "r": 30, "t": 80, "b": 60},
        )
        figure.add_annotation(
            xref="paper",
            yref="paper",
            x=-0.05,
            y=0.5,
            text="ΔRMSE (high-influence removal − low-influence removal)",
            textangle=-90,
            showarrow=False,
            font={"size": 12.5, "color": "#444444"},
        )
        for annotation in figure.layout.annotations[: len(panel_specs)]:
            annotation.update(font={"size": 13})
        return figure

    contrast_figure = _contrast_figure()
    contrast_figure
    return (contrast_figure,)


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            "**What to notice.** Native target molecules show a dose–response: the gap grows "
            "with the deletion size. The mixed pool inherits the signal. Donor-only panels are "
            "where the chemistry gets interesting — the direction is weak in aggregate, and for "
            "**Polaris donors it inverts under both other targets**: pruning the molecules the "
            "ranking most wants to keep actually improves the target model. External data are "
            "not interchangeable."
        ),
        kind="warn",
    )
    return


@app.cell
def _(dedent, mo):
    mo.md(
        dedent(
            """
            ## 3 · The confound ledger
    
            BoostIn influence is dominated by the measured label, so a naive “high-influence
            removal” is mostly a label-level intervention. The project's stratified controls exist
            to strip that out; the score–label correlation is the reason they are mandatory. The
            selected-run scatter has marginal violins by source.
            """
        )
    )
    return


@app.cell
def _(pd, runs):
    def _confound_table():
        records = []
        for target, payload in runs.items():
            scores = payload["scores"]
            for source, group in scores.groupby("source", sort=True):
                if len(group) < 10:
                    continue
                label_rho = (
                    group[["raw_score_mean", "model_target"]].corr(method="spearman").iloc[0, 1]
                )
                similarity_rho = (
                    group[["raw_score_mean", "max_tanimoto_to_selection"]]
                    .corr(method="spearman")
                    .iloc[0, 1]
                )
                records.append(
                    {
                        "target": target,
                        "source": str(source),
                        "label_rho": float(label_rho),
                        "similarity_rho": float(similarity_rho),
                        "n": int(len(group)),
                    }
                )
        return pd.DataFrame(records)

    confound_table = _confound_table()
    return (confound_table,)


@app.cell
def _(SOURCE_STYLE, TARGET_STYLE, confound_table, go, make_subplots, style_figure):
    def _confound_figure():
        targets = [target for target in TARGET_STYLE if target in set(confound_table["target"])]
        sources = [source for source in SOURCE_STYLE if source in set(confound_table["source"])]
        figure = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                "Influence vs measured label",
                "Influence vs similarity to the selection surface",
            ),
            horizontal_spacing=0.12,
        )
        for col, metric in ((1, "label_rho"), (2, "similarity_rho")):
            z_values, text_values = [], []
            for target in targets:
                row_z, row_text = [], []
                for source in sources:
                    match = confound_table.loc[
                        confound_table["target"].eq(target) & confound_table["source"].eq(source)
                    ]
                    value = float(match[metric].iloc[0]) if len(match) else float("nan")
                    row_z.append(value)
                    row_text.append(f"{value:+.2f}" if len(match) else "—")
                z_values.append(row_z)
                text_values.append(row_text)
            figure.add_trace(
                go.Heatmap(
                    z=z_values,
                    x=[SOURCE_STYLE[source]["label"] for source in sources],
                    y=[TARGET_STYLE[target]["label"] for target in targets],
                    text=text_values,
                    texttemplate="%{text}",
                    textfont={"size": 15},
                    colorscale="RdBu_r",
                    zmin=-1,
                    zmax=1,
                    showscale=col == 2,
                    colorbar={"title": "Spearman ρ", "thickness": 14},
                    hovertemplate="%{y} · %{x}: ρ = %{z:+.3f}<extra></extra>",
                ),
                row=1,
                col=col,
            )
        return style_figure(
            figure,
            "The confound ledger: BoostIn ranks labels, not chemical neighbourhoods",
            height=360,
            xaxis_title="Training source",
            yaxis_title="Target rotation",
        )

    confound_figure = _confound_figure()
    confound_figure
    return (confound_figure,)


@app.cell
def _(SOURCE_STYLE, TARGET_STYLE, px, rotation, runs):
    def _confound_scatter():
        payload = runs[rotation.value]
        scores = payload["scores"].copy()
        scores["source_label"] = scores["source"].map(
            {source: style["label"] for source, style in SOURCE_STYLE.items()}
        )
        figure = px.scatter(
            scores,
            x="model_target",
            y="raw_score_mean",
            color="source_label",
            color_discrete_map={style["label"]: style["colour"] for style in SOURCE_STYLE.values()},
            marginal_x="violin",
            marginal_y="violin",
            opacity=0.55,
            hover_data={"max_tanimoto_to_selection": ":.3f", "source": True},
            labels={
                "model_target": "Measured log HLM clearance",
                "raw_score_mean": "BoostIn influence (positive = beneficial)",
                "source_label": "Source",
            },
        )
        figure.update_traces(marker={"size": 5})
        figure.update_layout(
            template="plotly_white",
            title={
                "text": f"Every training molecule, ranked against the target loss · "
                f"{TARGET_STYLE[rotation.value]['label']} rotation",
                "x": 0.01,
                "xanchor": "left",
                "font": {"size": 17},
            },
            height=560,
            font={"family": "Inter, sans-serif", "size": 12.5},
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
            margin={"l": 70, "r": 30, "t": 95, "b": 60},
        )
        return figure

    confound_scatter = _confound_scatter()
    confound_scatter
    return (confound_scatter,)


@app.cell
def _(dedent, mo):
    mo.md(
        dedent(
            """
            ### Does matching actually neutralise the label?
    
            The stratified controls preserve source composition and within-source label quantile
            bins. If the ranking were only a label proxy, matched high and matched low deletions
            would land on the same label distribution and erase the surgery effect. At the largest
            batch size, the ranked arms separate sharply while the matched arms close the gap.
            """
        )
    )
    return


@app.cell
def _(px, rotation, runs, style_figure):
    def _matching_figure():
        payload = runs[rotation.value]
        arm_manifest = payload["arm_manifest"]
        kmax = int(arm_manifest["batch_size"].max())
        subset = arm_manifest.loc[arm_manifest["batch_size"].eq(kmax)]
        per_arm = (
            subset.groupby(["arm_id", "policy", "stratum"], dropna=False)["model_target"]
            .mean()
            .reset_index(name="removed_label_mean")
        )
        policy_labels = {
            "boostin_high": "Ranked high",
            "boostin_low": "Ranked low",
            "label_source_matched_high": "Matched high",
            "label_source_matched_low": "Matched low",
            "random": "Random",
        }
        policy_colours = {
            "Ranked high": "#E45756",
            "Ranked low": "#2E8B57",
            "Matched high": "#F2A541",
            "Matched low": "#72B7B2",
            "Random": "#999999",
        }
        per_arm["policy_label"] = per_arm["policy"].map(policy_labels)
        figure = px.box(
            per_arm,
            x="policy_label",
            y="removed_label_mean",
            color="policy_label",
            color_discrete_map=policy_colours,
            points="all",
            hover_data=["stratum", "arm_id"],
            labels={
                "policy_label": "Deletion policy",
                "removed_label_mean": "Mean measured label of removed molecules",
            },
            category_orders={
                "policy_label": [
                    "Ranked high",
                    "Matched high",
                    "Matched low",
                    "Ranked low",
                    "Random",
                ]
            },
        )
        figure.update_traces(marker={"size": 6}, jitter=0.35)
        figure.update_layout(showlegend=False)
        return style_figure(
            figure,
            f"Did matching work? Removed-label distributions at k = {kmax}",
            height=430,
            yaxis={"gridcolor": "#EEEEEE"},
            xaxis={"gridcolor": "#EEEEEE"},
        )

    matching_figure = _matching_figure()
    matching_figure
    return (matching_figure,)


@app.cell
def _(dedent, mo):
    mo.md(
        dedent(
            """
            ## 4 · Does the attribution score track the observed deletion effect?

            Every deletion arm is a real fitted model. Plotting the mean removed influence against
            the observed ΔRMSE tests whether the attribution score orders the refit outcomes
            without another fit. The fitted trend and the printed ρ cover the ranked arms only —
            the population the score is meant to order. Random and matched controls are drawn for
            reference, since they are comparison policies rather than score-ordered arms.
            """
        )
    )
    return


@app.cell
def _(TARGET_STYLE, go, np, rotation, runs, style_figure):
    def _predicted_figure():
        payload = runs[rotation.value]
        arm_manifest = payload["arm_manifest"]
        effects = payload["effects"]
        mean_removed = (
            arm_manifest.groupby("arm_id", as_index=False)["raw_score_mean"]
            .mean()
            .rename(columns={"raw_score_mean": "removed_score"})
        )

        def family(policy):
            if str(policy).startswith("boostin"):
                return "Ranked (high + low)"
            if policy == "random":
                return "Random control"
            return "Matched control"

        merged = (
            effects.loc[effects["comparison_kind"].eq("full_vs_pruned")]
            .merge(mean_removed, left_on="condition_b", right_on="arm_id", how="inner")
            .assign(policy_family=lambda frame: frame["policy"].map(family))
        )
        family_colours = {
            "Ranked (high + low)": "#6A3D9A",
            "Random control": "#999999",
            "Matched control": "#F2A541",
        }
        figure = go.Figure()
        for family_name, colour in family_colours.items():
            subset = merged.loc[merged["policy_family"].eq(family_name)]
            figure.add_trace(
                go.Scattergl(
                    x=subset["removed_score"],
                    y=subset["effect"],
                    mode="markers",
                    name=family_name,
                    marker={"size": 8, "color": colour, "opacity": 0.55},
                    customdata=subset[["eligible_pool", "stratum", "batch_size", "direction"]],
                    hovertemplate=(
                        "removed score = %{x:+.4f}<br>ΔRMSE = %{y:+.4f}<br>"
                        "%{customdata[0]} · %{customdata[1]} · k=%{customdata[2]} · "
                        "%{customdata[3]}<extra></extra>"
                    ),
                )
            )
        ranked = merged.loc[merged["policy_family"].eq("Ranked (high + low)")]
        slope, intercept = np.polyfit(ranked["removed_score"], ranked["effect"], 1)
        x_line = np.linspace(ranked["removed_score"].min(), ranked["removed_score"].max(), 50)
        figure.add_trace(
            go.Scatter(
                x=x_line,
                y=slope * x_line + intercept,
                mode="lines",
                name="Ranked trend",
                line={"color": "#6A3D9A", "width": 3},
                hoverinfo="skip",
            )
        )
        figure.add_hline(y=0, line_dash="dash", line_color="#555555")
        figure.add_vline(x=0, line_dash="dot", line_color="#BBBBBB")
        rho_ranked = ranked[["removed_score", "effect"]].corr(method="spearman").iloc[0, 1]
        rho_all = merged[["removed_score", "effect"]].corr(method="spearman").iloc[0, 1]
        style_figure(
            figure,
            f"Attribution score vs observed deletion effect · "
            f"{TARGET_STYLE[rotation.value]['label']}",
            height=540,
            xaxis_title="Mean removed BoostIn influence (positive = molecules the ranking "
            "wants kept)",
            yaxis_title="ΔRMSE after refit (positive = pruning improved)",
            xaxis={"zeroline": False, "gridcolor": "#EEEEEE"},
            yaxis={"gridcolor": "#EEEEEE"},
        )
        figure.add_annotation(
            x=0.99,
            y=0.02,
            xref="paper",
            yref="paper",
            xanchor="right",
            align="right",
            showarrow=False,
            text=(
                f"Spearman ρ over ranked arms = {rho_ranked:+.2f}<br>"
                f"all arms for reference = {rho_all:+.2f}"
            ),
            font={"size": 12, "color": "#444444"},
            bgcolor="rgba(255,255,255,0.85)",
        )
        return figure

    predicted_figure = _predicted_figure()
    predicted_figure
    return (predicted_figure,)


@app.cell
def _(dedent, mo):
    mo.md(
        dedent(
            """
            ## 5 · What changed inside the model

            An RMSE change can come from two very different places, and this panel separates
            them. Write the residual on the held-out surface as `r = observed − predicted`:

            - **Bias** = `mean(r)`. A positive bias means the model under-predicts on average;
              a constant shift could remove it, so this is *calibration* error.
            - **Centred RMSE** = `sqrt(mean((r − mean(r))²))`. This is the error remaining after
              removing a constant calibration shift. It still contains noise, heteroscedasticity,
              and any non-constant assay mismatch, so read it as "error beyond a constant offset",
              not as a pure structure–activity signal.
            - The three bars are linked exactly: **RMSE² = Bias² + Centred RMSE²**, so the first
              two decompose the third.

            Read each group of bars left to right. If an intervention moves mostly the bias bar,
            it behaves like a label-level calibration shift; if it moves mostly the centred bar,
            it changed the error structure beyond a constant offset. In these rotations the
            failure modes differ: **Polaris** local-only is badly biased and pooling repairs the
            bias, while high-influence removal throws that repair away; **Biogen** separates
            through centred error instead; **ExpansionRx** sits in between. The colour scheme
            encodes the design — blue/grey for the baselines, warm tones for high-influence
            removal, green tones for low-influence removal, with darker shades for the native
            pool and lighter shades for the mixed pool.
            """
        )
    )
    return


@app.cell
def _(TARGET_STYLE, go, np, pd, rotation, runs, style_figure):
    def _calibration_figure():
        payload = runs[rotation.value]
        predictions = pd.read_parquet(payload["directory"] / "predictions.parquet")
        effects = payload["effects"]
        ranked = effects.loc[effects["comparison_kind"].eq("ranked_vs_random_median")]
        kmax = ranked["batch_size"].max()
        conditions = [("full_pooled", "Full pooled"), ("local_only", "Local only")]
        for pool, stratum, label in (
            ("native_only", "target", "Native"),
            ("mixed_pool", "all", "Mixed"),
        ):
            for direction in ("high", "low"):
                match = ranked.loc[
                    ranked["eligible_pool"].eq(pool)
                    & ranked["stratum"].eq(stratum)
                    & ranked["batch_size"].eq(kmax)
                    & ranked["direction"].eq(direction)
                ]
                if not match.empty:
                    conditions.append(
                        (match.iloc[0]["condition_b"], f"{label} {direction} removal")
                    )
        condition_names = [condition for condition, _ in conditions]
        labels = {condition: label for condition, label in conditions}
        view = predictions.loc[predictions["condition"].isin(condition_names)]
        aggregated = (
            view.groupby(["condition", "evaluation_id"], as_index=False)["predicted"]
            .mean()
            .merge(
                view.drop_duplicates("evaluation_id")[["evaluation_id", "observed"]],
                on="evaluation_id",
                how="left",
            )
            .assign(residual=lambda frame: frame["observed"] - frame["predicted"])
        )
        decomposition = (
            aggregated.groupby("condition")["residual"]
            .agg(
                bias="mean",
                centred=lambda series: float(np.sqrt(np.mean((series - series.mean()) ** 2))),
                rmse=lambda series: float(np.sqrt(np.mean(series**2))),
            )
            .reset_index()
        )
        colours = {
            "Full pooled": "#2E6E9E",
            "Local only": "#8C8C8C",
            "Native high removal": "#C0392B",
            "Mixed high removal": "#E08A7F",
            "Native low removal": "#1E8449",
            "Mixed low removal": "#7DCEA0",
        }
        figure = go.Figure()
        for _, row in decomposition.iterrows():
            label = labels[row["condition"]]
            figure.add_trace(
                go.Bar(
                    name=label,
                    x=["Bias", "Centred RMSE", "RMSE"],
                    y=[row["bias"], row["centred"], row["rmse"]],
                    marker_color=colours.get(label, "#777777"),
                    hovertemplate="%{x}: %{y:.4f}<extra>" + label + "</extra>",
                )
            )
        return style_figure(
            figure,
            f"What changed: calibration shift or molecular relationship? · "
            f"{TARGET_STYLE[rotation.value]['label']}",
            height=470,
            barmode="group",
            yaxis_title="Log HLM clearance error",
            yaxis={"gridcolor": "#EEEEEE"},
            xaxis={"gridcolor": "#EEEEEE"},
        )

    calibration_figure = _calibration_figure()
    calibration_figure
    return (calibration_figure,)


@app.cell
def _(dedent, mo):
    mo.md(
        dedent(
            """
            ## 6 · Where is the chemistry?

            The score–similarity correlation is essentially zero in every rotation: the ranking
            has little association with the maximum-Tanimoto-to-selection summary used here. That
            is all it shows — other chemical summaries could behave differently. The PCA map is a
            descriptive view of fingerprint space, not a falsification test; the absence of
            obvious clusters would not rule out chemical structure. Held-out evaluation molecules
            are outlined in black.
            """
        )
    )
    return


@app.cell
def _(
    Chem,
    SOURCE_STYLE,
    TARGET_STYLE,
    go,
    np,
    rdFingerprintGenerator,
    rotation,
    runs,
    style_figure,
):
    def _chem_figure():
        payload = runs[rotation.value]
        manifest = payload["manifest"]
        scores = payload["scores"]
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        training = manifest.loc[manifest["role"].eq("full_training")].merge(
            scores[["candidate_id", "raw_score_mean", "max_tanimoto_to_selection"]],
            on="candidate_id",
            how="left",
        )
        evaluation = manifest.loc[manifest["role"].eq("evaluation")].copy()

        def fingerprints(smiles_series):
            rows = []
            for smiles in smiles_series.astype(str):
                molecule = Chem.MolFromSmiles(smiles)
                if molecule is None:
                    rows.append(np.zeros(2048, dtype=np.float32))
                else:
                    rows.append(generator.GetFingerprintAsNumPy(molecule).astype(np.float32))
            return np.stack(rows)

        matrix = fingerprints(training["canonical_smiles"])
        mean = matrix.mean(axis=0)
        centred = matrix - mean
        eigenvalues, eigenvectors = np.linalg.eigh(centred.T @ centred)
        order = np.argsort(eigenvalues)[::-1][:2]
        embedding = centred @ eigenvectors[:, order]
        variance_share = eigenvalues[order] / eigenvalues.sum() * 100
        evaluation_embedding = (fingerprints(evaluation["canonical_smiles"]) - mean) @ eigenvectors[
            :, order
        ]
        score_values = training["raw_score_mean"].to_numpy(dtype=float)
        finite = score_values[np.isfinite(score_values)]
        limit = float(np.quantile(np.abs(finite), 0.98)) if finite.size else 1.0
        figure = go.Figure()
        selection_lookup = {}
        for source, style in SOURCE_STYLE.items():
            subset = training["source"].eq(source).to_numpy()
            source_rows = training.loc[subset]
            selection_lookup[len(figure.data)] = source_rows["candidate_id"].astype(str).tolist()
            figure.add_trace(
                go.Scattergl(
                    x=embedding[subset, 0],
                    y=embedding[subset, 1],
                    mode="markers",
                    name=style["label"],
                    marker={
                        "size": 5,
                        "color": score_values[subset],
                        "colorscale": "RdBu_r",
                        "cmin": -limit,
                        "cmax": limit,
                        "symbol": style["symbol"],
                        "opacity": 0.65,
                        "colorbar": {
                            "title": "BoostIn<br>influence",
                            "thickness": 14,
                            "len": 0.75,
                        },
                        "showscale": source == "expansionrx",
                    },
                    customdata=source_rows[
                        [
                            "candidate_id",
                            "model_target",
                            "max_tanimoto_to_selection",
                            "raw_score_mean",
                        ]
                    ],
                    hovertemplate=(
                        "source = " + style["label"] + "<br>label = %{customdata[1]:.2f}<br>"
                        "score = %{customdata[3]:+.4f}<br>max Tanimoto to selection = "
                        "%{customdata[2]:.3f}<extra></extra>"
                    ),
                )
            )
        selection_lookup[len(figure.data)] = evaluation["candidate_id"].astype(str).tolist()
        figure.add_trace(
            go.Scattergl(
                x=evaluation_embedding[:, 0],
                y=evaluation_embedding[:, 1],
                mode="markers",
                name="Evaluation molecules",
                marker={
                    "size": 6,
                    "color": "rgba(0,0,0,0)",
                    "line": {"width": 1.2, "color": "#333333"},
                    "symbol": "diamond-open",
                },
                customdata=evaluation[["candidate_id", "original_id"]],
                hovertemplate=("held-out evaluation molecule = %{customdata[1]}<extra></extra>"),
            )
        )
        style_figure(
            figure,
            f"Chemical space of the pooled training set · "
            f"{TARGET_STYLE[rotation.value]['label']} rotation",
            height=640,
            xaxis_title=f"PC1 ({variance_share[0]:.1f}% variance)",
            yaxis_title=f"PC2 ({variance_share[1]:.1f}% variance)",
            xaxis={"showgrid": False, "zeroline": False, "visible": False},
            yaxis={"showgrid": False, "zeroline": False, "visible": False},
        )
        figure.add_annotation(
            x=0.99,
            y=0.02,
            xref="paper",
            yref="paper",
            xanchor="right",
            showarrow=False,
            text="Morgan r=2, 2048 bits · training points coloured by influence · "
            "held-out evaluation outlined · box-select or lasso to inspect molecules",
            font={"size": 11, "color": "#666666"},
            bgcolor="rgba(255,255,255,0.85)",
        )
        return figure, selection_lookup

    chem_figure, chem_selection_lookup = _chem_figure()
    return (chem_figure, chem_selection_lookup)


@app.cell
def _(chem_figure, mo):
    pca_plot = mo.ui.plotly(chem_figure)
    pca_plot
    return (pca_plot,)


@app.cell
def _(
    MoleculeGrid,
    chem_selection_lookup,
    mo,
    molgrid_available,
    molecule_records,
    pca_plot,
    rotation,
    runs,
    selection_ids,
):
    brushed_ids = selection_ids(pca_plot.value, chem_selection_lookup)
    payload = runs[rotation.value]
    brushed_records = (
        molecule_records(payload["manifest"], payload["scores"], brushed_ids, cap=200)
        if brushed_ids
        else []
    )
    if not molgrid_available:
        brushed_view = mo.callout(
            mo.md("Install `anywidget` to inspect brushed molecules interactively."),
            kind="warn",
        )
    elif not brushed_records:
        brushed_view = mo.callout(
            mo.md(
                "**Pick the box-select or lasso tool in the map's mode bar**, then drag "
                "around points to load those molecules below."
            ),
            kind="info",
        )
    else:
        brushed_view = mo.vstack(
            [
                mo.md(
                    f"**Brushed molecules** · showing {len(brushed_records)} of "
                    f"{len(brushed_ids)} selected"
                ),
                mo.ui.anywidget(
                    MoleculeGrid(molecules=brushed_records, title="Chemical-space selection")
                ),
            ]
        )
    brushed_view
    return


@app.cell
def _(dedent, mo):
    mo.md(
        dedent(
            """
            ### Scaffold view

            Aggregating at the Bemis–Murcko level asks whether the signal is chemically
            interpretable. Click a bar in the scaffold selector to load that scaffold's molecules
            into the interactive grid. The grid supports search, sorting, and click selection;
            highlighted atoms mark the scaffold.
            """
        )
    )
    return


@app.cell
def _(TARGET_STYLE, go, rotation, runs, scaffold_summary, style_figure):
    def _scaffold_view():
        payload = runs[rotation.value]
        table = scaffold_summary(payload["manifest"], payload["scores"])
        seen = {}
        labels = []
        for value in table["short"]:
            seen[value] = seen.get(value, 0) + 1
            suffix = "" if seen[value] == 1 else f" ({seen[value]})"
            labels.append(f"{value}{suffix}")
        table = table.assign(label=labels)
        figure = go.Figure()
        for direction, colour in (("Harmful", "#E45756"), ("Helpful", "#4C78A8")):
            subset = table.loc[table["direction"].eq(direction)].sort_values("mean_score")
            figure.add_trace(
                go.Bar(
                    x=subset["mean_score"],
                    y=subset["label"],
                    orientation="h",
                    name=direction,
                    marker_color=colour,
                    customdata=subset[["n", "scaffold", "label"]],
                    hovertemplate=(
                        "mean influence = %{x:+.4f}<br>molecules = %{customdata[0]}<br>"
                        "scaffold = %{customdata[1]}<extra></extra>"
                    ),
                )
            )
        style_figure(
            figure,
            f"Scaffold-level influence · {TARGET_STYLE[rotation.value]['label']} rotation",
            height=620,
            xaxis_title="Mean BoostIn influence (positive = ranked helpful)",
            yaxis={"gridcolor": "#EEEEEE", "tickfont": {"size": 10}},
            xaxis={"gridcolor": "#EEEEEE", "zeroline": True, "zerolinecolor": "#888888"},
        )
        bar_records = (
            table.sort_values("mean_score", ascending=False)
            .loc[:, ["scaffold", "label", "n", "mean_score", "direction"]]
            .to_dict("records")
        )
        return table, figure, bar_records

    scaffold_table, scaffold_figure, scaffold_bar_records = _scaffold_view()
    return (scaffold_bar_records, scaffold_figure, scaffold_table)


@app.cell
def _(
    ScaffoldBars,
    TARGET_STYLE,
    mo,
    molgrid_available,
    rotation,
    scaffold_bar_records,
    scaffold_figure,
    scaffold_table,
):
    scaffold_default_row = scaffold_table.nlargest(1, "mean_score").iloc[0]
    scaffold_default = str(scaffold_default_row["scaffold"])
    if molgrid_available:
        scaffold_control = mo.ui.anywidget(
            ScaffoldBars(
                scaffolds=scaffold_bar_records,
                selected=scaffold_default,
                title=f"Scaffold influence · {TARGET_STYLE[rotation.value]['label']}",
            )
        )
        scaffold_selector = scaffold_control
    else:
        scaffold_control = mo.ui.dropdown(
            options=dict(
                zip(
                    scaffold_table["label"],
                    scaffold_table["scaffold"],
                    strict=True,
                )
            ),
            value=str(scaffold_default_row["label"]),
            label="Scaffold selector",
            full_width=True,
        )
        scaffold_selector = mo.vstack([scaffold_figure, scaffold_control])
    scaffold_selector
    return (scaffold_control,)


@app.cell
def _(molecule_records, rotation, runs, scaffold_control, scaffold_of, scaffold_table):
    def _scaffold_records():
        payload = runs[rotation.value]
        raw_selection = scaffold_control.value
        if isinstance(raw_selection, dict):
            selected = str(raw_selection.get("selected") or "")
        else:
            selected = str(raw_selection or "")
        if selected not in set(scaffold_table["scaffold"]):
            selected = str(scaffold_table.nlargest(1, "mean_score").iloc[0]["scaffold"])
        manifest = payload["manifest"]
        training = manifest.loc[manifest["role"].eq("full_training")].copy()
        training["scaffold"] = training["canonical_smiles"].map(scaffold_of)
        member_ids = (
            training.loc[training["scaffold"].eq(selected), "candidate_id"].astype(str).tolist()
        )
        scores = payload["scores"]
        score_order = scores.drop_duplicates("candidate_id").set_index("candidate_id")[
            "raw_score_mean"
        ]
        member_ids = sorted(
            member_ids,
            key=lambda candidate: score_order.get(candidate, float("-inf")),
            reverse=True,
        )
        records = molecule_records(
            manifest, scores, member_ids, cap=200, highlight_scaffold=selected
        )
        return selected, records

    selected_scaffold, scaffold_records = _scaffold_records()
    return (scaffold_records, selected_scaffold)


@app.cell
def _(
    Chem,
    Draw,
    MoleculeGrid,
    mo,
    molgrid_available,
    scaffold_records,
    selected_scaffold,
):
    if molgrid_available:
        scaffold_grid = mo.ui.anywidget(
            MoleculeGrid(
                molecules=scaffold_records,
                title=(
                    f"Scaffold explorer · {len(scaffold_records)} molecules · "
                    f"{selected_scaffold[:44]}"
                ),
            )
        )
        scaffold_view = scaffold_grid
    else:
        scaffold_grid = None
        molecules = [
            molecule
            for molecule in (
                Chem.MolFromSmiles(record["smiles"]) for record in scaffold_records[:6]
            )
            if molecule is not None
        ]
        image = (
            Draw.MolsToGridImage(molecules, molsPerRow=3, subImgSize=(300, 220))
            if molecules
            else None
        )
        fallback = [
            mo.callout(
                mo.md(
                    "Install `anywidget` for the interactive molecule grid; showing a "
                    "static fallback instead."
                ),
                kind="warn",
            )
        ]
        if image is not None:
            fallback.append(mo.hstack([image], justify="center"))
        scaffold_view = mo.vstack(fallback)
    scaffold_view
    return (scaffold_grid,)


@app.cell
def _(mo, pd, rotation, runs, scaffold_grid):
    def _selection_detail():
        if scaffold_grid is None:
            return mo.md("")
        value = scaffold_grid.value if isinstance(scaffold_grid.value, dict) else {}
        selected = [str(candidate) for candidate in value.get("selected", [])]
        if not selected:
            return mo.callout(
                mo.md(
                    "Click molecules in the grid to inspect their measured labels, "
                    "influences, and similarities."
                ),
                kind="info",
            )
        payload = runs[rotation.value]
        manifest = payload["manifest"].drop_duplicates("candidate_id").set_index("candidate_id")
        scores = payload["scores"].drop_duplicates("candidate_id").set_index("candidate_id")
        rows = []
        for candidate in selected:
            if candidate not in manifest.index:
                continue
            row = manifest.loc[candidate]
            score_row = scores.loc[candidate] if candidate in scores.index else None
            rows.append(
                {
                    "molecule": str(row.get("original_id", candidate)),
                    "source": str(row.get("source", "")),
                    "role": str(row.get("role", "")),
                    "label": row.get("model_target"),
                    "influence": None if score_row is None else score_row.get("raw_score_mean"),
                    "max tanimoto": None
                    if score_row is None
                    else score_row.get("max_tanimoto_to_selection"),
                    "inchikey": str(row.get("inchikey", "")),
                }
            )
        return mo.vstack(
            [
                mo.md(f"**Selected molecules** · {len(rows)}"),
                mo.ui.table(pd.DataFrame(rows), selection=None),
            ]
        )

    _selection_detail()
    return


@app.cell
def _(mo):
    mo.vstack(
        [
            mo.md("## 7 · The bottom line"),
            mo.md(
                """
                - **Pooling is not one decision.** Across the three rotations the point estimate
                  flips sign; no pooling interval excludes zero, so the practical reading is that
                  a pooled donor corpus needs a per-target verdict, not a global one.
                - **The native deletion direction is consistent.** The high-vs-low contrast
                  interval excludes zero in all three rotations and grows with the deletion size.
                  Under the coarse source + label-quantile control the interval excludes zero only
                  in ExpansionRx; Biogen and Polaris are inconclusive at their evaluation-set
                  sizes, and those controls are not continuous-label matches.
                - **The ranking is dominated by label level.** Score–label Spearman is +0.85 for
                  ExpansionRx, +0.87 for Polaris, and −0.52 for Biogen; score–similarity with the
                  maximum-Tanimoto summary is ~0 everywhere. The stratified control is the only
                  comparison that tests the ranking beyond label extremity, and it is coarse.
                - **Donors are not interchangeable.** Polaris donors invert the direction for
                  both other targets: the ranking flags molecules that actively hurt.
                - **Magnitudes scale with the surgical budget.** Donor effects are a few percent
                  of RMSE; native deletions at k = 100–150 reach double-digit percentages of RMSE,
                  but cost large parts of the training set. This is ranking evidence, not a
                  solved business case.
                """
            ),
            mo.accordion(
                {
                    "Why are the persisted effect tables recomputed here?": mo.md(
                        "The frozen `paired_effects.parquet` for the Biogen and Polaris rotations "
                        "aligns the evaluation frame's row order with identifier-sorted prediction "
                        "matrices, so targets and clusters are shuffled relative to predictions. "
                        "The evaluator is fixed (`evaluate.py`), and this notebook repairs the "
                        "derived tables from the frozen per-molecule predictions without refitting "
                        "a single model. The point estimate for the Biogen pooling contrast flips "
                        "sign when repaired, which is exactly why the notebook must not trust the "
                        "old intervals."
                    ),
                    "Why are the intervals so wide for Polaris?": mo.md(
                        "The Polaris rotation has 49 evaluation molecules in 13 Butina clusters. "
                        "Cluster-bootstrap intervals over 13 clusters are exploratory by "
                        "construction; the notebook labels them as such rather than hiding them."
                    ),
                    "What is the selection surface?": mo.md(
                        "Attribution is scored against a target-like selection surface disjoint "
                        "from the evaluation molecules. The evaluation surface is unreachable by "
                        "construction, so every number here is a held-out measurement."
                    ),
                    "Scope and honesty": mo.md(
                        "The ExpansionRx test partition has informed prior exploration, so "
                        "results are temporally ordered exploratory analyses, not prospective "
                        "confirmation. Dry runs are excluded. No addition-utility calibration "
                        "is used: direction is assessed only through conditional deletion."
                    ),
                    "What the matched control does and does not show": mo.md(
                        "The stratified controls preserve source composition and within-source "
                        "label quantile bins; they are not continuous-label matches, and the "
                        "removed-label panel shows the ranked and matched arms can still sit at "
                        "different label levels. The supported statement is that the direction "
                        "is not fully explained by this coarse stratification, not that label "
                        "level has been neutralised."
                    ),
                }
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
