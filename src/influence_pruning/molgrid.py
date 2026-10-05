"""Reactive molecule explorer for the read-only analysis notebooks.

The module is a single script that holds the custom anywidget used by
``notebooks/04_pooling_surgery.py``:

* :class:`MoleculeGrid` renders RDKit SVGs in a searchable, sortable, clickable grid and
  syncs the selected candidate identifiers back to Python.
* The helpers (:func:`draw_svg`, :func:`scaffold_of`, :func:`scaffold_summary`,
  :func:`molecule_records`) do the chemistry-safe data preparation so notebook cells stay
  thin and the same logic is unit-tested.

Nothing in this module reads or writes run artifacts; the notebook passes in frames that it
has already loaded.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.Scaffolds import MurckoScaffold

try:
    import anywidget
    import traitlets
except ImportError:  # pragma: no cover - the widget is optional outside the dev environment
    anywidget = None
    traitlets = None

__all__ = [
    "MoleculeGrid",
    "ScaffoldBars",
    "Speaker",
    "draw_svg",
    "molecule_records",
    "scaffold_of",
    "scaffold_summary",
    "selection_ids",
]

_GRID_ESM = """
const escapeHtml = (value) =>
  String(value ?? "").replace(/[&<>"']/g, (character) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[character]);

const asNumber = (value, digits = 2) =>
  value === null || value === undefined || Number.isNaN(value)
    ? "—"
    : Number(value).toFixed(digits);

const asSigned = (value, digits = 4) =>
  value === null || value === undefined || Number.isNaN(value)
    ? "—"
    : (value >= 0 ? "+" : "") + Number(value).toFixed(digits);

function render({ model, el }) {
  el.classList.add("molgrid");

  const header = document.createElement("div");
  header.className = "molgrid-header";
  const title = document.createElement("div");
  title.className = "molgrid-title";
  const count = document.createElement("div");
  count.className = "molgrid-count";
  header.append(title, count);

  const toolbar = document.createElement("div");
  toolbar.className = "molgrid-toolbar";
  const search = document.createElement("input");
  search.type = "search";
  search.placeholder = "Search name, source, scaffold, or id";
  search.className = "molgrid-search";
  const sort = document.createElement("select");
  for (const [value, label] of [
    ["score_desc", "Influence ↓"],
    ["score_asc", "Influence ↑"],
    ["label_desc", "Label ↓"],
    ["label_asc", "Label ↑"],
    ["similarity_desc", "Similarity ↓"],
    ["name", "Name"],
  ]) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = label;
    sort.append(option);
  }
  sort.value = "score_desc";
  const clear = document.createElement("button");
  clear.textContent = "Clear selection";
  clear.className = "molgrid-clear";
  toolbar.append(search, sort, clear);

  const grid = document.createElement("div");
  grid.className = "molgrid-grid";
  el.append(header, toolbar, grid);

  const rows = () => model.get("molecules") || [];

  function visibleRows() {
    const query = search.value.trim().toLowerCase();
    let rowsToShow = rows().filter((molecule) => {
      if (!query) {
        return true;
      }
      return [molecule.name, molecule.id, molecule.source, molecule.scaffold].some(
        (field) => String(field ?? "").toLowerCase().includes(query),
      );
    });
    const sortedBy = (key, direction) => (left, right) => {
      const leftValue =
        left[key] === null || left[key] === undefined ? -Infinity : left[key];
      const rightValue =
        right[key] === null || right[key] === undefined ? -Infinity : right[key];
      return direction * (leftValue - rightValue);
    };
    switch (sort.value) {
      case "score_asc":
        rowsToShow.sort(sortedBy("score", 1));
        break;
      case "label_desc":
        rowsToShow.sort(sortedBy("label", -1));
        break;
      case "label_asc":
        rowsToShow.sort(sortedBy("label", 1));
        break;
      case "similarity_desc":
        rowsToShow.sort(sortedBy("similarity", -1));
        break;
      case "name":
        rowsToShow.sort((left, right) =>
          String(left.name).localeCompare(String(right.name)),
        );
        break;
      default:
        rowsToShow.sort(sortedBy("score", -1));
    }
    return rowsToShow;
  }

  function draw() {
    const selected = new Set(model.get("selected") || []);
    const rowsToShow = visibleRows();
    title.textContent = model.get("title") || "Molecules";
    count.textContent =
      `${rowsToShow.length} shown · ${rows().length} total · ` +
      `${selected.size} selected`;
    grid.innerHTML = "";
    for (const molecule of rowsToShow) {
      const card = document.createElement("div");
      card.className = "molgrid-card" + (selected.has(molecule.id) ? " selected" : "");
      card.dataset.id = molecule.id;
      card.innerHTML =
        `<div class="molgrid-figure">${molecule.svg}</div>` +
        `<div class="molgrid-meta">` +
        `<div class="molgrid-name">${escapeHtml(molecule.name)}</div>` +
        `<div class="molgrid-source">${escapeHtml(molecule.source)} · ` +
        `${escapeHtml(molecule.role)}</div>` +
        `<div class="molgrid-stats">score ${asSigned(molecule.score)} · ` +
        `label ${asNumber(molecule.label)}</div>` +
        `</div>`;
      card.addEventListener("click", () => toggle(molecule.id));
      grid.append(card);
    }
  }

  function toggle(id) {
    const selected = model.get("selected") || [];
    const next = selected.includes(id)
      ? selected.filter((candidate) => candidate !== id)
      : [...selected, id];
    model.set("selected", next);
    model.save_changes();
  }

  clear.addEventListener("click", () => {
    model.set("selected", []);
    model.save_changes();
  });
  search.addEventListener("input", draw);
  sort.addEventListener("change", draw);
  model.on("change:molecules", draw);
  model.on("change:title", draw);
  model.on("change:selected", draw);

  draw();
}

export default { render };
"""

_GRID_CSS = """
.molgrid { font-family: Inter, -apple-system, BlinkMacSystemFont, sans-serif; }
.molgrid-title { font-size: 14px; font-weight: 600; color: #222222; }
.molgrid-count { font-size: 11.5px; color: #777777; margin-top: 2px; }
.molgrid-toolbar { display: flex; gap: 8px; margin: 10px 0 12px; flex-wrap: wrap; }
.molgrid-search {
  flex: 1 1 220px; padding: 6px 8px; border: 1px solid #dcdcdc;
  border-radius: 6px; font-size: 12px;
}
.molgrid-toolbar select, .molgrid-clear {
  padding: 6px 8px; border: 1px solid #dcdcdc; border-radius: 6px;
  background: white; font-size: 12px; cursor: pointer;
}
.molgrid-grid {
  display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 10px;
}
.molgrid-card {
  border: 1px solid #e8e8e8; border-radius: 10px; padding: 6px; background: white;
  cursor: pointer; transition: border-color .12s ease, box-shadow .12s ease;
}
.molgrid-card:hover { border-color: #b9b9b9; }
.molgrid-card.selected {
  border-color: #6A3D9A; box-shadow: 0 0 0 2px rgba(106, 61, 154, 0.22);
  background: #faf7ff;
}
.molgrid-figure svg { width: 100%; height: auto; display: block; }
.molgrid-name {
  font-size: 11.5px; color: #333333; margin-top: 4px;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.molgrid-source { font-size: 10.5px; color: #777777; }
.molgrid-stats { font-size: 10.5px; color: #555555; }
"""

_BARS_ESM = """
const escapeHtml = (value) =>
  String(value ?? "").replace(/[&<>"']/g, (character) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[character]);

function render({ model, el }) {
  el.classList.add("scaffoldbars");
  const title = document.createElement("div");
  title.className = "scaffoldbars-title";
  const hint = document.createElement("div");
  hint.className = "scaffoldbars-hint";
  const list = document.createElement("div");
  list.className = "scaffoldbars-list";
  el.append(title, hint, list);

  function draw() {
    const scaffolds = model.get("scaffolds") || [];
    const selected = model.get("selected");
    title.textContent = model.get("title") || "Scaffold influence";
    hint.textContent = "Click a bar to load its molecules into the grid below";
    list.innerHTML = "";
    const maxAbs = Math.max(
      ...scaffolds.map((record) => Math.abs(record.mean_score || 0)),
      1e-9,
    );
    for (const record of scaffolds) {
      const value = record.mean_score || 0;
      const width = Math.max(2, (Math.abs(value) / maxAbs) * 100);
      const row = document.createElement("div");
      row.className =
        "scaffoldbars-row" + (record.scaffold === selected ? " selected" : "");
      row.innerHTML =
        `<div class="scaffoldbars-head">` +
        `<span class="scaffoldbars-label">${escapeHtml(record.label)}</span>` +
        `<span class="scaffoldbars-meta">n=${record.n} · ` +
        `${value >= 0 ? "+" : ""}${value.toFixed(4)}</span>` +
        `</div>` +
        `<div class="scaffoldbars-track">` +
        `<div class="scaffoldbars-fill ${value >= 0 ? "positive" : "negative"}" ` +
        `style="width:${width}%"></div>` +
        `</div>`;
      row.addEventListener("click", () => {
        model.set("selected", record.scaffold);
        model.save_changes();
      });
      list.append(row);
    }
  }

  model.on("change:scaffolds", draw);
  model.on("change:selected", draw);
  model.on("change:title", draw);
  draw();
}

export default { render };
"""

_BARS_CSS = """
.scaffoldbars { font-family: Inter, -apple-system, BlinkMacSystemFont, sans-serif; }
.scaffoldbars-title { font-size: 14px; font-weight: 600; color: #222222; }
.scaffoldbars-hint { font-size: 11.5px; color: #777777; margin: 2px 0 10px; }
.scaffoldbars-list {
  display: flex; flex-direction: column; gap: 6px; max-height: 560px;
  overflow-y: auto; padding-right: 4px;
}
.scaffoldbars-row {
  border: 1px solid #ececec; border-radius: 8px; padding: 6px 10px;
  cursor: pointer; background: white;
  transition: border-color .12s ease, box-shadow .12s ease;
}
.scaffoldbars-row:hover { border-color: #b9b9b9; }
.scaffoldbars-row.selected {
  border-color: #6A3D9A; box-shadow: 0 0 0 2px rgba(106, 61, 154, 0.22);
  background: #faf7ff;
}
.scaffoldbars-head {
  display: flex; justify-content: space-between; gap: 10px;
  font-size: 11.5px; color: #333333;
}
.scaffoldbars-label { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.scaffoldbars-meta { color: #777777; flex-shrink: 0; }
.scaffoldbars-track {
  height: 7px; background: #f3f3f3; border-radius: 4px; margin-top: 5px;
  overflow: hidden;
}
.scaffoldbars-fill { height: 100%; border-radius: 4px; }
.scaffoldbars-fill.positive { background: #4C78A8; }
.scaffoldbars-fill.negative { background: #E45756; }
"""

_SPEAKER_ESM = """
function render({ model, el }) {
  el.classList.add("speaker");
  const supported = "speechSynthesis" in window;

  const controls = document.createElement("div");
  controls.className = "speaker-controls";
  const button = document.createElement("button");
  button.className = "speaker-button";
  const voice = document.createElement("select");
  voice.className = "speaker-voice";
  voice.setAttribute("aria-label", "Voice");
  const rate = document.createElement("input");
  rate.className = "speaker-rate";
  rate.type = "range";
  rate.min = "0.5";
  rate.max = "1.8";
  rate.step = "0.1";
  rate.value = "1";
  rate.setAttribute("aria-label", "Speed");
  const rateLabel = document.createElement("span");
  rateLabel.className = "speaker-rate-label";
  const status = document.createElement("div");
  status.className = "speaker-status";
  controls.append(button, voice, rate, rateLabel);
  el.append(controls, status);

  let phase = "idle";
  let userSelectedVoice = false;

  const preferredLanguage = () => model.get("lang") || "en-GB";

  const normalizeLanguage = (value) =>
    String(value || "")
      .toLowerCase()
      .replace(/_/g, "-");

  const voiceRank = (entry) => {
    const lang = normalizeLanguage(entry.lang);
    const preferred = normalizeLanguage(preferredLanguage());
    if (lang === preferred) {
      return 0;
    }
    if (lang.startsWith("en-gb")) {
      return 1;
    }
    if (lang.startsWith("en")) {
      return 2;
    }
    return 3;
  };

  function updateButton() {
    if (!supported) {
      button.textContent = model.get("label") || "Listen";
      return;
    }
    button.textContent =
      phase === "speaking"
        ? "Pause"
        : phase === "paused"
          ? "Resume"
          : model.get("label") || "Listen";
  }

  function setPhase(next) {
    phase = next;
    updateButton();
  }

  function populateVoices() {
    if (!supported) {
      return;
    }
    const voices = window.speechSynthesis.getVoices() || [];
    const previous = voice.value;
    const sorted = [...voices].sort(
      (left, right) =>
        voiceRank(left) - voiceRank(right) || left.name.localeCompare(right.name),
    );
    voice.innerHTML = "";
    for (const entry of sorted) {
      const option = document.createElement("option");
      option.value = entry.name;
      option.textContent = `${entry.name} · ${entry.lang}`;
      voice.append(option);
    }
    if (previous && voices.some((entry) => entry.name === previous)) {
      voice.value = previous;
    } else if (!userSelectedVoice && sorted.length) {
      const preferred = sorted.find((entry) => voiceRank(entry) <= 2) || sorted[0];
      voice.value = preferred.name;
    }
    voice.disabled = voices.length === 0;
  }

  function speak() {
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(model.get("text") || "");
    const voices = window.speechSynthesis.getVoices() || [];
    const chosen = voices.find((entry) => entry.name === voice.value);
    if (chosen) {
      utterance.voice = chosen;
    } else {
      utterance.lang = preferredLanguage();
    }
    utterance.rate = Number(rate.value) || 1;
    utterance.onend = () => setPhase("idle");
    utterance.onerror = () => setPhase("idle");
    window.speechSynthesis.speak(utterance);
    setPhase("speaking");
  }

  button.addEventListener("click", () => {
    if (!supported) {
      return;
    }
    if (phase === "idle") {
      speak();
    } else if (phase === "speaking") {
      window.speechSynthesis.pause();
      setPhase("paused");
    } else {
      window.speechSynthesis.resume();
      setPhase("speaking");
    }
  });

  voice.addEventListener("change", () => {
    userSelectedVoice = true;
  });
  rate.addEventListener("input", () => {
    rateLabel.textContent = `${Number(rate.value).toFixed(1)}×`;
  });
  rateLabel.textContent = "1.0×";

  if (!supported) {
    button.disabled = true;
    voice.disabled = true;
    rate.disabled = true;
    status.textContent = "This browser does not expose the Web Speech API.";
  } else {
    status.textContent = "Browser speech synthesis · voice and speed stay on your machine";
    populateVoices();
    window.speechSynthesis.onvoiceschanged = populateVoices;
  }
  updateButton();

  model.on("change:label", updateButton);
  model.on("change:text", () => {
    if (phase !== "idle") {
      window.speechSynthesis.cancel();
      setPhase("idle");
    }
  });
}

export default { render };
"""

_SPEAKER_CSS = """
.speaker { font-family: Inter, -apple-system, BlinkMacSystemFont, sans-serif; }
.speaker-controls { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.speaker-button {
  padding: 8px 14px; border-radius: 8px; border: 1px solid #6A3D9A;
  background: #6A3D9A; color: white; font-size: 13px; font-weight: 600;
  cursor: pointer;
}
.speaker-button:hover { background: #5a3285; }
.speaker-button:disabled { background: #cccccc; border-color: #cccccc; cursor: not-allowed; }
.speaker-voice {
  max-width: 240px; padding: 6px 8px; border: 1px solid #dcdcdc;
  border-radius: 6px; background: white; font-size: 12px;
}
.speaker-rate { width: 110px; }
.speaker-rate-label { font-size: 12px; color: #555555; min-width: 34px; }
.speaker-status { font-size: 11.5px; color: #777777; margin-top: 6px; }
"""

_PLACEHOLDER_SVG = (
    "<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
    "viewBox='0 0 {width} {height}'>"
    "<rect width='100%' height='100%' rx='6' fill='#fafafa'/>"
    "<text x='50%' y='50%' text-anchor='middle' fill='#999999' font-size='12' "
    "font-family='sans-serif'>{message}</text></svg>"
)


if anywidget is not None:

    class MoleculeGrid(anywidget.AnyWidget):  # type: ignore[misc]
        """Searchable, sortable SVG molecule grid with click selection.

        :param molecules: list of record dicts (see :func:`molecule_records`).
        :param selected: candidate identifiers currently selected.
        :param title: heading shown above the grid.
        """

        _esm = _GRID_ESM
        _css = _GRID_CSS

        molecules = traitlets.List([]).tag(sync=True)
        selected = traitlets.List([]).tag(sync=True)
        title = traitlets.Unicode("Molecules").tag(sync=True)

    class ScaffoldBars(anywidget.AnyWidget):  # type: ignore[misc]
        """Clickable horizontal bars ranking scaffolds by mean influence.

        :param scaffolds: list of records with ``scaffold``, ``label``, ``n``,
            ``mean_score``, and ``direction``.
        :param selected: scaffold SMILES currently selected.
        :param title: heading shown above the bars.
        """

        _esm = _BARS_ESM
        _css = _BARS_CSS

        scaffolds = traitlets.List([]).tag(sync=True)
        selected = traitlets.Unicode("").tag(sync=True)
        title = traitlets.Unicode("Scaffold influence").tag(sync=True)

    class Speaker(anywidget.AnyWidget):  # type: ignore[misc]
        """Browser text-to-speech control for a passage of the notebook.

        Uses the Web Speech API, so no API key, audio asset, or server-side model is
        involved; the selected voice and speed stay on the viewer's machine.

        :param text: passage to read aloud.
        :param label: idle button label, for example "Hear the verdict".
        :param lang: BCP-47 language tag used to pick a default voice, e.g. ``en-GB``.
        """

        _esm = _SPEAKER_ESM
        _css = _SPEAKER_CSS

        text = traitlets.Unicode("").tag(sync=True)
        label = traitlets.Unicode("Listen").tag(sync=True)
        lang = traitlets.Unicode("en-GB").tag(sync=True)

else:  # pragma: no cover - exercised only when anywidget is absent
    MoleculeGrid = None  # type: ignore[assignment,misc]
    ScaffoldBars = None  # type: ignore[assignment,misc]
    Speaker = None  # type: ignore[assignment,misc]


def scaffold_of(smiles: str) -> str:
    """Return the Bemis-Murcko scaffold SMILES, or an empty string when unparsable."""
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)
    except Exception:
        return ""


def scaffold_atoms(smiles: str, scaffold: str) -> tuple[int, ...]:
    """Return the atom indices of `scaffold` inside `smiles`, or an empty tuple."""
    molecule = Chem.MolFromSmiles(str(smiles))
    scaffold_molecule = Chem.MolFromSmiles(str(scaffold))
    if molecule is None or scaffold_molecule is None:
        return ()
    match = molecule.GetSubstructMatch(scaffold_molecule)
    return tuple(match)


def draw_svg(
    smiles: str,
    *,
    width: int = 210,
    height: int = 160,
    highlight_atoms: Sequence[int] | None = None,
) -> str:
    """Render one molecule to an inline SVG string.

    Invalid structures return a labelled placeholder instead of raising, so one bad record
    can never break the grid.
    """
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        return _PLACEHOLDER_SVG.format(width=width, height=height, message="invalid structure")
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    drawer.drawOptions().fixedBondLength = 28
    if highlight_atoms:
        rdMolDraw2D.PrepareAndDrawMolecule(drawer, molecule, highlightAtoms=list(highlight_atoms))
    else:
        drawer.DrawMolecule(molecule)
    drawer.FinishDrawing()
    return (
        drawer.GetDrawingText().replace("<?xml version='1.0' encoding='iso-8859-1'?>", "").strip()
    )


def molecule_records(
    manifest: pd.DataFrame,
    scores: pd.DataFrame | None,
    candidate_ids: Iterable[str],
    *,
    cap: int = 200,
    highlight_scaffold: str | None = None,
) -> list[dict]:
    """Build grid records for `candidate_ids`, preserving order and capping the payload.

    :param manifest: partition manifest with one row per molecule.
    :param scores: scored molecules with ``raw_score_mean`` and
        ``max_tanimoto_to_selection``; evaluation molecules are simply absent.
    :param candidate_ids: identifiers to render, deduplicated in first-seen order.
    :param cap: maximum number of records returned.
    :param highlight_scaffold: optional scaffold SMILES highlighted in every depiction.
    :returns: list of record dicts consumed by :class:`MoleculeGrid`.
    """
    frame = manifest.drop_duplicates("candidate_id").set_index("candidate_id")
    score_frame = (
        scores.drop_duplicates("candidate_id").set_index("candidate_id")
        if scores is not None
        else None
    )
    records: list[dict] = []
    for candidate in list(dict.fromkeys(str(value) for value in candidate_ids))[:cap]:
        if candidate not in frame.index:
            continue
        row = frame.loc[candidate]
        smiles = str(row["canonical_smiles"])
        highlight = scaffold_atoms(smiles, highlight_scaffold) if highlight_scaffold else None
        score = None
        similarity = None
        if score_frame is not None and candidate in score_frame.index:
            score_row = score_frame.loc[candidate]
            score = _finite_or_none(score_row.get("raw_score_mean"))
            similarity = _finite_or_none(score_row.get("max_tanimoto_to_selection"))
        records.append(
            {
                "id": candidate,
                "smiles": smiles,
                "svg": draw_svg(smiles, highlight_atoms=highlight),
                "name": str(row.get("original_id", candidate)),
                "source": str(row.get("source", "")),
                "role": str(row.get("role", "")),
                "label": _finite_or_none(row.get("model_target")),
                "score": score,
                "similarity": similarity,
                "scaffold": scaffold_of(smiles),
            }
        )
    return records


def scaffold_summary(
    manifest: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    min_count: int = 5,
    top: int = 8,
) -> pd.DataFrame:
    """Rank Bemis-Murcko scaffolds of the pooled training set by mean influence.

    :param manifest: partition manifest with one row per molecule.
    :param scores: scored molecules with ``raw_score_mean``.
    :param min_count: minimum molecules per scaffold to be ranked.
    :param top: number of scaffolds kept per direction.
    :returns: columns ``scaffold, n, mean_score, direction, short``.
    """
    training = manifest.loc[manifest["role"].eq("full_training")].merge(
        scores[["candidate_id", "raw_score_mean"]], on="candidate_id", how="left"
    )
    with_scaffold = training.assign(scaffold=training["canonical_smiles"].map(scaffold_of))
    grouped = (
        with_scaffold.loc[with_scaffold["scaffold"].ne("")]
        .groupby("scaffold")
        .agg(n=("raw_score_mean", "size"), mean_score=("raw_score_mean", "mean"))
        .query("n >= @min_count")
        .reset_index()
    )
    combined = pd.concat(
        [
            grouped.nsmallest(top, "mean_score").assign(direction="Harmful"),
            grouped.nlargest(top, "mean_score").assign(direction="Helpful"),
        ],
        ignore_index=True,
    )
    combined["short"] = combined["scaffold"].str.slice(0, 40)
    return combined


def selection_ids(
    selection: object,
    lookup: dict[int, Sequence[str]],
) -> list[str]:
    """Resolve a marimo plotly selection payload into candidate identifiers.

    marimo can hand back the selection as a list of point dicts (clicks) or as a dict with a
    ``points`` list (box/lasso extraction). Click points may carry ``customdata``, while
    box/lasso-extracted points only carry ``curveNumber`` and ``pointIndex``, which are
    resolved through `lookup` (trace index → identifiers in trace row order).

    :param selection: raw ``mo.ui.plotly`` value.
    :param lookup: trace curve number to its row-ordered candidate identifiers.
    :returns: deduplicated identifiers in first-seen order.
    """
    if isinstance(selection, list):
        points = [point for point in selection if isinstance(point, dict)]
    elif isinstance(selection, dict):
        points = selection.get("points") or []
    else:
        points = []
    found: list[str] = []
    for point in points:
        customdata = point.get("customdata")
        if isinstance(customdata, (list, tuple)) and customdata:
            found.append(str(customdata[0]))
            continue
        if isinstance(customdata, dict) and customdata.get("candidate_id"):
            found.append(str(customdata["candidate_id"]))
            continue
        if point.get("candidate_id"):
            found.append(str(point["candidate_id"]))
            continue
        curve = point.get("curveNumber")
        index = point.get("pointNumber", point.get("pointIndex"))
        if curve is not None and index is not None:
            trace_ids = lookup.get(int(curve), [])
            if 0 <= int(index) < len(trace_ids):
                found.append(str(trace_ids[int(index)]))
    return list(dict.fromkeys(found))


def _finite_or_none(value: object) -> float | None:
    """Return a JSON-safe float, or None for missing values."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return number
