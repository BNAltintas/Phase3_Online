from __future__ import annotations

import datetime
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import dash
import numpy as np
import plotly.graph_objects as go
from dash import ALL, MATCH, ClientsideFunction, Input, Output, State, ctx, dcc, html, no_update
from plotly.subplots import make_subplots

from propofol.config import BOLUS_MGKG_BOUNDS
from propofol.dashboard_layout import (
    EHR_RECORD_DATE,
    EHR_RECORD_TIME,
    PR_DEFAULT_EXPLORE_RATE_MCGKGMIN,
    PR_GRID_MAX_MCGKGMIN,
    PR_GRID_MIN_MCGKGMIN,
    PR_GRID_POINTS,
    PRESET_TEST_PATIENTS,
    SCENARIO_DEFAULT_PATIENT,
    SCENARIO_DEFAULT_REMI_MCGKGMIN,
    TEST_EXPLORATION_DEFAULT_MAP_ABS,
    TEST_EXPLORATION_DEFAULT_MAP_REL,
    TEST_EXPLORATION_DEFAULT_PATIENT,
    _te_result_header_row,
    _te_result_section_header,
    build_layout,
    timestamp_display,
)
from propofol.patient import EleveldPatient as Patient
from propofol.recommend_regimen2023 import (
    DEFAULT_PROPOFOL_CONC_MG_ML,
    DEFAULT_REMI_CONC_MCG_ML,
    MAP_ABS_MIN_TARGET,
    MAP_REL_FRAC_TARGET,
    N_INTERVALS,
    REMI_INFUSION_MCGKGMIN_BOUNDS,
    TARGET_ASSESSMENT_START_MIN,
    TARGET_BIS_HIGH,
    TARGET_BIS_LOW,
    DecodedRegimen,
    Su2023PropofolRemifentanilRecommender,
    compress_minute_schedule,
    recommend_maintenance_for_fixed_propofol_bolus,
    recommend_propofol_for_fixed_remifentanil_rate,
    recommend_su2023_regimen,
    remi_mcgkgmin_to_ml_h,
    remi_mcgkgmin_to_ngkgmin,
)

# ============================================================
# App setup
# ============================================================

# Font Awesome isn't a project dependency (no pip package, no local asset) -
# this is the standard, lowest-friction way to get real FA glyphs (not
# emoji) into a Dash app's sidebar icons: one CDN stylesheet, no new
# Python dependency.
FONT_AWESOME_CDN = "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css"

# ------------------------------------------------------------
# Precomputed Remi's offline-generated data (see
# scripts/precompute_remi_cases.py). Loaded once, here, at import time -
# never re-read or recomputed while the app is running, and never touches
# recommend_regimen2023.py. Missing/corrupt file -> None, not a raised
# exception, so a server that hasn't had the precompute script run yet
# still starts up and every other page keeps working; the page itself
# (build_precomputed_remi_page in dashboard_layout.py) renders a clear
# "not generated yet" message in that case instead of a case picker.
#
# A file that DOES load as valid JSON but doesn't match what this build of
# the app expects (stale data from before the Precomputed Remi page was
# restricted to Test Patient 1 / the 0.00-2.00 grid, most obviously) is a
# different failure mode from "not generated yet" - _validate_precomputed_
# remi_data raises instead of returning None, so a mismatch is a loud
# startup crash a developer will immediately notice and fix by re-running
# the precompute script, not a page that silently shows Test Patient 2/3
# or the old 0.02-0.20 range again.
# ------------------------------------------------------------
PRECOMPUTED_REMI_PATH = Path(__file__).resolve().parent / "data" / "precomputed_remi_cases.json"

PR_REQUIRED_REGIMEN_FIELDS = (
    "propofol_bolus_mg", "propofol_bolus_mgkg", "propofol_inf_rates_mcgkgmin",
    "remifentanil_selected", "remifentanil_inf_rates_mcgkgmin",
    "time_min", "bis", "map_mmhg", "cp_propofol", "dose_sweep",
)
PR_REQUIRED_DOSE_SWEEP_FIELDS = ("dose_grid_mgkg", "min_map_mmhg", "min_bis", "max_bis")


def _validate_precomputed_remi_regimen(regimen: dict, where: str) -> None:
    for field in PR_REQUIRED_REGIMEN_FIELDS:
        if field not in regimen:
            raise ValueError(f"Precomputed Remi data: {where} is missing required field {field!r}.")
    sweep = regimen["dose_sweep"]
    for field in PR_REQUIRED_DOSE_SWEEP_FIELDS:
        if field not in sweep:
            raise ValueError(f"Precomputed Remi data: {where}.dose_sweep is missing required field {field!r}.")


def _validate_precomputed_remi_data(data: dict) -> None:
    """
    Development-time guard against silently loading stale Precomputed Remi
    data (see the module comment above PRECOMPUTED_REMI_PATH). Checked
    once, at import time, right after a successful JSON parse - raises
    ValueError with a specific, actionable message rather than letting the
    app start up with wrong patients/rates/missing fields.
    """
    cases = data.get("cases")
    if not isinstance(cases, dict):
        raise ValueError("Precomputed Remi data: missing or malformed 'cases' object.")

    expected_case_ids = {"1"}
    actual_case_ids = set(cases.keys())
    if actual_case_ids != expected_case_ids:
        raise ValueError(
            "Precomputed Remi data: expected only Test Patient 1 (case '1'), "
            f"found case ids {sorted(actual_case_ids)!r}. Re-run "
            "scripts/precompute_remi_cases.py to regenerate a Test-Patient-1-only dataset."
        )

    case = cases["1"]
    grid = case.get("remifentanil_grid") or {}
    rates = grid.get("rates_mcgkgmin")
    results = grid.get("results")
    if not rates or not results:
        raise ValueError("Precomputed Remi data: case '1' has no remifentanil grid rates/results.")
    if len(rates) != len(results):
        raise ValueError(
            f"Precomputed Remi data: case '1' has {len(rates)} grid rates but {len(results)} results."
        )
    if len(rates) != PR_GRID_POINTS:
        raise ValueError(
            f"Precomputed Remi data: case '1' has {len(rates)} grid points, expected {PR_GRID_POINTS}."
        )
    if not math.isclose(rates[0], PR_GRID_MIN_MCGKGMIN, abs_tol=1e-9):
        raise ValueError(
            f"Precomputed Remi data: minimum grid rate is {rates[0]}, expected {PR_GRID_MIN_MCGKGMIN}."
        )
    if not math.isclose(rates[-1], PR_GRID_MAX_MCGKGMIN, abs_tol=1e-9):
        raise ValueError(
            f"Precomputed Remi data: maximum grid rate is {rates[-1]}, expected {PR_GRID_MAX_MCGKGMIN}."
        )
    if rates != sorted(rates):
        raise ValueError("Precomputed Remi data: case '1' grid rates are not sorted ascending.")

    for baseline_key in ("baseline_none", "baseline_remifentanil"):
        baseline = case.get(baseline_key)
        if not isinstance(baseline, dict):
            raise ValueError(f"Precomputed Remi data: case '1' is missing '{baseline_key}'.")
        _validate_precomputed_remi_regimen(baseline, f"case '1'.{baseline_key}")

    for i, result in enumerate(results):
        _validate_precomputed_remi_regimen(result, f"case '1'.remifentanil_grid.results[{i}]")


def _load_precomputed_remi_data() -> Optional[dict]:
    if not PRECOMPUTED_REMI_PATH.exists():
        return None
    try:
        with PRECOMPUTED_REMI_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    _validate_precomputed_remi_data(data)
    return data


PRECOMPUTED_REMI_DATA = _load_precomputed_remi_data()


def _pr_case(case_id: Optional[str]) -> Optional[dict]:
    """Look up one case's precomputed data, or None if unavailable/unknown - never raises."""
    if PRECOMPUTED_REMI_DATA is None or not case_id:
        return None
    return PRECOMPUTED_REMI_DATA.get("cases", {}).get(case_id)


def _pr_rate_to_index(rates: list, rate: Optional[float]) -> int:
    """
    Exact-match lookup from a requested remifentanil rate to its index in
    the precomputed grid's own rates list - not nearest-value/interpolated
    matching. Every selectable rate is one of the grid's own 41 exact
    values (the slider's step=None + marks means it can only land on one
    of them), so this always finds a true exact match in normal use;
    comparing integer hundredths (0, 5, 10, ..., 200) rather than the raw
    floats is purely to stay immune to float representation noise (e.g.
    0.15000000000000002), never to pick a "close enough" value that isn't
    the one actually requested. Falls back to index 0 only defensively
    (rate is None, or - should never happen - an unexpected rate outside
    the grid), mirroring the previous None-rate default.
    """
    if rate is None:
        return 0
    target = round(rate * 100)
    for i, r in enumerate(rates):
        if round(r * 100) == target:
            return i
    return 0


app = dash.Dash(__name__, external_stylesheets=[FONT_AWESOME_CDN])
app.title = "Su2023 Propofol Dashboard"
app.layout = build_layout(precomputed_remi_available=PRECOMPUTED_REMI_DATA is not None)

# The induction card's "Override propofol dose" button/popover only exist in
# the DOM after the first "Run recommendation" (summary-output starts empty).
# Callbacks that target those ids must be allowed to reference ids that are
# not present in the initial layout - this is the standard Dash mechanism
# for that.
app.config.suppress_callback_exceptions = True

# Shared prediction-graph color language, used consistently across BIS, MAP,
# Propofol PK/PD, and Remifentanil PK: green always means the recommended
# (model) prediction, orange always means a manual-dose override, and the
# light-blue band color always means uncertainty/confidence interval - never
# reused for anything else, so users can read any prediction graph the same
# way without re-learning a per-graph color key.
RECOMMENDED_COLOR = "#1f9254"
MANUAL_OVERRIDE_COLOR = "#c2680f"
UNCERTAINTY_BAND_COLOR = "#2f6fed"


# ============================================================
# Small helpers
# ============================================================

def compute_map(sap: float, dap: float) -> float:
    """
    Compute the mean arterial pressure (MAP) given systolic and diastolic pressures.
    """
    return (sap + 2.0 * dap) / 3.0


def compute_pp(sap: float, dap: float) -> float:
    """
    Compute the pulse pressure (PP) given systolic and diastolic pressures.
    """
    return sap - dap


def validate_concentration(value, name: str) -> float:
    """
    Validate a directly-edited concentration value.
    """
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} concentration must be a number.")

    if not np.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} concentration must be positive.")

    return parsed


def make_empty_figure(title: str | None = None):
    """
    Create an empty Plotly figure with an optional title.
    """
    fig = go.Figure()
    fig.update_layout(template="plotly_white", title=title or None)
    return fig


def _safe_array(x) -> np.ndarray:
    """
    Convert input to a NumPy array of floats.
    """
    return np.asarray(x, dtype=float)


# ============================================================
# Recommendation <-> dcc.Store serialization
#
# RecommendationResult only exists as a local variable inside the callback
# that computed it. To let a later, separate action (saving a manual
# override) redraw the original recommendation without recomputing it - and
# to reconstruct a recommender for the override itself - the fields needed
# for display/reconstruction are serialized into recommendation-store /
# manual-scenario-store, then rebuilt into a lightweight namespace with the
# same attribute shape as RecommendationResult, so the existing figure
# functions work unchanged on either a live result or a stored one.
# ============================================================

def _array_list(x) -> Optional[list]:
    """
    Convert an array-like (or None) into a plain list for JSON storage.
    """
    return None if x is None else np.asarray(x, dtype=float).tolist()


def _rec_to_store_dict(rec) -> dict:
    """
    Serialize the RecommendationResult fields needed to redraw the induction
    and maintenance cards and the 5 prediction graphs, without re-running
    any computation.
    """
    return {
        "propofol_bolus_mg": float(rec.propofol_bolus_mg),
        "propofol_bolus_mgkg": float(rec.propofol_bolus_mgkg),
        "propofol_inf_rates_mgkgh": _array_list(rec.propofol_inf_rates_mgkgh),
        "propofol_inf_rates_ml_h": _array_list(rec.propofol_inf_rates_ml_h),
        "propofol_inf_rates_mcgkgmin": _array_list(rec.propofol_inf_rates_mcgkgmin),
        "remifentanil_selected": bool(rec.remifentanil_selected),
        "remifentanil_bolus_mcg": float(rec.remifentanil_bolus_mcg),
        "remifentanil_bolus_mcgkg": float(rec.remifentanil_bolus_mcgkg),
        "remifentanil_inf_rates_mcgkgmin": _array_list(rec.remifentanil_inf_rates_mcgkgmin),
        "remifentanil_inf_rates_ml_h": _array_list(rec.remifentanil_inf_rates_ml_h),
        "remifentanil_inf_rates_ngkgmin": _array_list(rec.remifentanil_inf_rates_ngkgmin),
        "propofol_conc_mg_ml": float(rec.propofol_conc_mg_ml),
        "remifentanil_conc_mcg_ml": float(rec.remifentanil_conc_mcg_ml),
        "time_min": _array_list(rec.time_min),
        "cp_propofol": _array_list(rec.cp_propofol),
        "ce_propofol": _array_list(rec.ce_propofol),
        "cp_remifentanil": _array_list(rec.cp_remifentanil),
        "bis": _array_list(rec.bis),
        "map_mmhg": _array_list(rec.map_mmhg),
        "map_lower_bound_mmhg": float(rec.map_lower_bound_mmhg),
        "feasible_bis": bool(rec.feasible_bis),
        "feasible_map": bool(rec.feasible_map),
        "feasible": bool(rec.feasible),
        "target_bis_low": float(rec.target_bis_low),
        "target_bis_high": float(rec.target_bis_high),
        "map_abs_min_target_mmhg": float(rec.map_abs_min_target_mmhg),
        "map_rel_frac_target": float(rec.map_rel_frac_target),
        "confidence_percent": float(rec.confidence_percent),
        "confidence_skipped": bool(rec.confidence_skipped),
        "confidence": {
            "time_min": _array_list(rec.confidence.time_min),
            "bis_p05": _array_list(rec.confidence.bis_p05),
            "bis_p95": _array_list(rec.confidence.bis_p95),
            "map_p05": _array_list(rec.confidence.map_p05),
            "map_p95": _array_list(rec.confidence.map_p95),
            "cp_propofol_p05": _array_list(rec.confidence.cp_propofol_p05),
            "cp_propofol_p95": _array_list(rec.confidence.cp_propofol_p95),
            "ce_propofol_p05": _array_list(rec.confidence.ce_propofol_p05),
            "ce_propofol_p95": _array_list(rec.confidence.ce_propofol_p95),
            "cp_remifentanil_p05": _array_list(rec.confidence.cp_remifentanil_p05),
            "cp_remifentanil_p95": _array_list(rec.confidence.cp_remifentanil_p95),
        },
    }


def _store_dict_to_namespace(data: dict) -> SimpleNamespace:
    """
    Reconstruct a lightweight object with the same attribute shape as
    RecommendationResult from a stored dict (as returned by
    _rec_to_store_dict), so the existing figure-building functions - which
    use rec.xxx / rec.confidence.xxx attribute access - work unchanged on
    data that came back from a dcc.Store instead of a live
    RecommendationResult.
    """
    data = dict(data)
    confidence_dict = data.pop("confidence")

    ns = SimpleNamespace(**data)
    ns.confidence = SimpleNamespace(**confidence_dict)

    for obj in (ns, ns.confidence):
        for key, value in vars(obj).items():
            if isinstance(value, list):
                setattr(obj, key, np.asarray(value, dtype=float))

    return ns


def _dose_unit_suffix(unit: str) -> str:
    """
    Return the display unit label for an override-dose unit code.
    """
    return "mg/kg" if unit == "mgkg" else "mg"


def _dose_value_for_display(value_mg: float, weight_kg: float, unit: str) -> str:
    """
    Format an absolute mg dose in whichever unit the user is currently typing in.
    """
    if unit == "mgkg":
        return f"{value_mg / weight_kg:.2f}"
    return f"{value_mg:.0f}"


def _validate_override_dose(raw_value, weight_kg: float, unit: str = "total"):
    """
    Validate a manually-entered propofol induction dose, typed as either a
    total dose in mg (unit="total") or a weight-adjusted dose in mg/kg
    (unit="mgkg").

    Hard bounds reject non-positive/absurd values outright. Warn bounds are
    derived from BOLUS_MGKG_BOUNDS x patient weight - the same range the
    optimizer normally searches within - so a dose outside it is still
    allowed (per the "treat this as a what-if, don't silently block"
    requirement) but flagged. Bounds are always evaluated in mg internally
    (converting a typed mg/kg value first), so both units are held to the
    same real-world limits; only the display text changes.

    Returns (status, message, parsed_value_mg_or_None) - the parsed value,
    when not None, is always the absolute mg dose the model needs,
    regardless of which unit the user typed in.
    """
    if raw_value is None or (isinstance(raw_value, str) and raw_value.strip() == ""):
        return "error", "Please enter a manual dose.", None

    try:
        typed_value = float(raw_value)
    except (TypeError, ValueError):
        return "error", "Please enter a valid number.", None

    if not np.isfinite(typed_value) or typed_value <= 0:
        return "error", "Manual dose must be positive.", None

    weight_kg = float(weight_kg)
    value_mg = typed_value * weight_kg if unit == "mgkg" else typed_value
    unit_label = _dose_unit_suffix(unit)

    hard_max_mg = 8.0 * weight_kg
    if value_mg > hard_max_mg:
        return (
            "error",
            (
                f"Manual dose cannot exceed "
                f"{_dose_value_for_display(hard_max_mg, weight_kg, unit)} "
                f"{unit_label} for this patient."
            ),
            None,
        )

    warn_min_mg = BOLUS_MGKG_BOUNDS[0] * weight_kg
    warn_max_mg = BOLUS_MGKG_BOUNDS[1] * weight_kg
    if value_mg < warn_min_mg or value_mg > warn_max_mg:
        return (
            "warning",
            (
                f"This dose is outside the model's typical range "
                f"({_dose_value_for_display(warn_min_mg, weight_kg, unit)}"
                f"–{_dose_value_for_display(warn_max_mg, weight_kg, unit)} "
                f"{unit_label} for this patient). "
                f"Maintenance will still be re-optimized, but BIS/MAP targets may "
                f"not be fully reachable."
            ),
            value_mg,
        )

    return "normal", "", value_mg


# ============================================================
# Schedule formatting
# ============================================================

def confidence_tier(confidence_percent: float) -> str:
    """
    Categorize a numeric confidence percentage into a HIGH/MEDIUM/LOW label
    for display. Presentation only - does not affect confidence_percent
    itself or how it is calculated.
    """
    if confidence_percent >= 20:
        return "HIGH"
    if confidence_percent >= 1:
        return "MEDIUM"
    return "LOW"


# Phase 3 prototype display mapping only - the raw confidence_percent value
# (from ConfidenceResult.confidence_percent, 0-100 scale) is still computed
# and stored exactly as before and is never overwritten. This maps the
# already-assigned category (from confidence_tier, above - never
# re-derived from raw thresholds here) to a fixed number shown to the
# user, so the displayed percentage always matches the qualitative label.
_DISPLAY_CONFIDENCE_PERCENT_BY_TIER = {"HIGH": 85, "MEDIUM": 60, "LOW": 10}


def display_confidence_percent(tier: str) -> int:
    """
    Map a confidence_tier() category ("HIGH"/"MEDIUM"/"LOW") to the fixed
    percentage shown in the UI. Single shared helper so this mapping is
    defined in exactly one place.
    """
    return _DISPLAY_CONFIDENCE_PERCENT_BY_TIER[tier]


# User-testing override: for the three preset test patients, the displayed
# confidence must stay pinned to a fixed tier/percentage for the whole
# scenario, regardless of the actual computed confidence_percent and
# regardless of any patient/target/opioid/manual-dose edits made afterward.
# Keyed by the same identifier PRESET_TEST_PATIENTS and
# selected-test-patient-store already use ("1"/"2"/"3" - not the patient's
# field values, and not a new id scheme), so switching which preset is
# selected is the only thing that changes which override applies. Any
# patient key not in this mapping (including None) falls through to the
# normal confidence_tier()/display_confidence_percent() logic in
# _display_confidence_for below, completely unaffected by this override.
TEST_PATIENT_CONFIDENCE_OVERRIDES = {
    "1": {"tier": "HIGH", "percent": 80},
    "2": {"tier": "MEDIUM", "percent": 60},
    "3": {"tier": "LOW", "percent": 10},
}


def _display_confidence_for(test_patient_key: Optional[str], confidence_percent: float) -> tuple[str, int]:
    """
    Return the (tier, displayed_percent) pair for the confidence badge.

    Presentation only, same as confidence_tier()/display_confidence_percent()
    above - confidence_percent itself (the real simulated value already
    sitting in recommendation-store/manual-scenario-store/
    PRECOMPUTED_REMI_DATA) is never read into, or overwritten by, this
    function; it is only used as a fallback input for patients that are
    not one of the three preset test patients.
    """
    override = TEST_PATIENT_CONFIDENCE_OVERRIDES.get(test_patient_key)
    if override is not None:
        return override["tier"], override["percent"]
    tier = confidence_tier(confidence_percent)
    return tier, display_confidence_percent(tier)


CONFIDENCE_INFO_TOOLTIP = (
    "Confidence indicates how consistently the recommended dose is expected "
    "to achieve both the BIS and MAP targets despite natural patient "
    "variability. Higher confidence means the recommendation is expected to "
    "perform more reliably."
)


def _maintenance_rows(rate_ml_h, rate_secondary, secondary_label: str) -> list:
    """
    Build interval/arrow/rate grid-cell components for one drug's
    compressed maintenance schedule. A near-zero rate is shown as "Pause"
    instead of "0 mL/h (0 x/kg/min)". The values, units, and time intervals
    themselves come from compress_minute_schedule() exactly as before -
    only the extra centered arrow cell between them is new, purely
    presentational.
    """
    ml_rows = compress_minute_schedule(rate_ml_h)
    secondary_rows = compress_minute_schedule(rate_secondary)

    cells = []
    for ml_row, sec_row in zip(ml_rows, secondary_rows, strict=False):
        interval_text = f"{ml_row['start_min']:.0f}–{ml_row['end_min']:.0f} min"
        is_pause = np.isclose(ml_row["rate"], 0.0, atol=1e-8)

        if is_pause:
            rate_text = "Pause"
            rate_class = "maintenance-rate maintenance-rate--pause"
        else:
            rate_text = f"{ml_row['rate']:.0f} mL/h ({sec_row['rate']:.0f} {secondary_label})"
            rate_class = "maintenance-rate"

        cells.append(html.Div(interval_text, className="maintenance-interval"))
        cells.append(html.Div("→", className="maintenance-arrow"))
        cells.append(html.Div(rate_text, className=rate_class))

    return cells


def _override_popover(prefill_value):
    """
    Build the click-to-open "Manual dose" popover. Always rendered (closed
    by default) regardless of whether an override is currently active, so
    its component ids stay stable for the callbacks that target them.

    The dose input and the unit toggle (Total dose / mg/kg) sit side by
    side in one row - purely a layout choice, the ids and the values they
    carry are unchanged from before. The unit toggle always opens defaulted
    to "Total dose" - handle_override resets it every time the popover is
    opened, so prefill_value (always an absolute mg amount) never needs
    converting for display here.
    """
    return html.Div(
        [
            dcc.Store(id="override-draft-unit", data="total"),
            html.Div("Manual dose", className="override-popover-title"),
            html.Div(
                [
                    dcc.Input(
                        id="override-dose-draft",
                        type="text",
                        value=prefill_value,
                        className="field-input",
                        placeholder="Manual dose in mg",
                    ),
                    dcc.RadioItems(
                        id="override-unit",
                        options=[
                            {"label": "Total dose", "value": "total"},
                            {"label": "mg/kg", "value": "mgkg"},
                        ],
                        value="total",
                        inline=True,
                        className="override-unit-toggle",
                    ),
                ],
                className="override-dose-row",
            ),
            html.Div(id="override-dose-message", className="field-message"),
            html.Div(
                [
                    html.Button(
                        "Save", id="override-save-btn", n_clicks=0,
                        className="popover-btn popover-btn--save",
                    ),
                    html.Button(
                        "Cancel", id="override-cancel-btn", n_clicks=0,
                        className="popover-btn popover-btn--cancel",
                    ),
                ],
                className="popover-actions",
            ),
        ],
        id="override-popover",
        className="edit-popover",
    )


def make_induction_card(original, manual=None, test_patient_key=None):
    """
    Build the induction-dose recommendation card - a dark-navy focal card
    (see .recommendation-dark-card/.induction-recommendation-card in
    style.css), showing the total dose (the
    manual override's dose when one is active, otherwise the model's own
    recommendation) side-by-side with either the model confidence (no
    override) or a compact two-line "Manual Dose / Active" status box
    (override active). The big dose number/unit/mg-kg switch to orange
    (induction-dose-value--manual in style.css) whenever they're showing
    the manual dose instead of the model's own recommendation; the small
    "Model recommendation: ..." note shown alongside them in manual mode
    stays green, since it always names the actual model recommendation.

    Model confidence is computed only for the original (free-bolus)
    recommendation, never for a manual override (recommend_maintenance_
    for_fixed_propofol_bolus always sets confidence_skipped=True and never
    calls simulate_confidence) - so while a manual override is active,
    there is nothing confidence-shaped to show, and the right-hand column
    shows the status box instead. This is a display choice, not a
    calculation change: confidence_percent is still computed exactly the
    same way as before, only its visibility changed. Manual-dose mode
    hides confidence entirely regardless of test_patient_key - the same
    as before this parameter existed - and returning to the recommendation
    (manual=None) naturally re-applies whichever tier/percent
    _display_confidence_for resolves for test_patient_key.

    test_patient_key (selected-test-patient-store's current value, e.g.
    "1"/"2"/"3", or None/anything else for a non-preset patient) selects
    the fixed user-testing confidence override via _display_confidence_for
    - see TEST_PATIENT_CONFIDENCE_OVERRIDES above. It never changes which
    dose is shown, only which (tier, percent) pair the confidence badge
    below displays.

    "Return to recommendation" is always rendered (never conditionally
    excluded), only ever hidden via inline display:none, because it's a
    callback Input - Dash logs a console error if a registered callback's
    Input id is ever absent from the current DOM entirely.
    """
    manual_active = manual is not None
    prefill_value = manual.propofol_bolus_mg if manual_active else original.propofol_bolus_mg

    dose_mg = manual.propofol_bolus_mg if manual_active else original.propofol_bolus_mg
    dose_mgkg = manual.propofol_bolus_mgkg if manual_active else original.propofol_bolus_mgkg

    dose_label_text = "MANUAL TOTAL DOSE" if manual_active else "TOTAL DOSE"
    dose_label_class = "induction-dose-label"
    dose_value_class = "induction-dose-value"
    if manual_active:
        dose_label_class += " induction-dose-label--manual"
        dose_value_class += " induction-dose-value--manual"

    dose_block_children = [
        html.Div(dose_label_text, className=dose_label_class),
        html.Div(
            [
                html.Span(f"{dose_mg:.0f}", className="induction-dose-number"),
                html.Span(" mg", className="induction-dose-unit"),
                html.Span(f"({dose_mgkg:.2f} mg/kg)", className="induction-dose-mgkg"),
            ],
            className=dose_value_class,
        ),
    ]
    if manual_active:
        dose_block_children.append(
            html.Div(
                (
                    f"Model recommendation: {original.propofol_bolus_mg:.0f} mg "
                    f"({original.propofol_bolus_mgkg:.2f} mg/kg)"
                ),
                className="induction-dose-original-note",
            )
        )
    dose_block = html.Div(dose_block_children, className="induction-dose-block")

    if manual_active:
        right_block = html.Div(
            [
                html.Div("Manual Dose", className="induction-manual-active-line"),
                html.Div("Active", className="induction-manual-active-line"),
            ],
            className="induction-manual-active-box",
        )
    else:
        tier, shown_percent = _display_confidence_for(test_patient_key, original.confidence_percent)
        tier_class = tier.lower()
        right_block = html.Div(
            [
                html.Div(
                    [
                        "MODEL CONFIDENCE",
                        html.Span(
                            "i",
                            title=CONFIDENCE_INFO_TOOLTIP,
                            className="info-icon info-icon--confidence",
                        ),
                    ],
                    className="induction-confidence-label",
                ),
                html.Div(
                    f"{tier} CONFIDENCE",
                    className=f"induction-confidence-tier induction-confidence-tier--{tier_class}",
                ),
                html.Div(
                    f"{shown_percent}%",
                    className=f"induction-confidence-value induction-confidence-value--{tier_class}",
                ),
            ],
            className="induction-confidence-block",
        )

    edit_btn_label = "Edit dose" if manual_active else "Enter Manual Dose"
    edit_btn_class = "induction-edit-btn"
    if manual_active:
        edit_btn_class += " induction-edit-btn--active"

    buttons_row = html.Div(
        [
            html.Button(
                [html.I(className="fa-solid fa-pen"), edit_btn_label],
                id="override-dose-btn", n_clicks=0,
                className=edit_btn_class,
            ),
            html.Button(
                [html.I(className="fa-solid fa-rotate-left"), "Return to recommendation"],
                id="override-return-btn", n_clicks=0,
                className="induction-return-btn",
                style=None if manual_active else {"display": "none"},
            ),
        ],
        className="induction-buttons-row",
    )

    card_children = [
        html.H4("Induction recommendation", className="induction-card-title"),
        html.Div([dose_block, right_block], className="induction-card-body"),
        html.Div(
            [buttons_row, _override_popover(prefill_value)],
            className="override-dose-section",
        ),
    ]

    return html.Div(card_children, className="card induction-card recommendation-dark-card induction-recommendation-card")


def _maintenance_section(rec, re_optimized=False):
    """
    Build one drug-schedule section (propofol + optional remifentanil) for
    the maintenance card. Each drug gets a colored bar-style header
    (Propofol in the app's blue accent, Remifentanil in the same purple
    family as the patient-id icon - never orange, which this app reserves
    for warnings/manual-override accents), with a dashed divider
    separating the two when both are present. When re_optimized is True
    (i.e. a manual dose override is active), a small orange "Re-optimized"
    badge is appended next to each drug header.
    """
    propofol_header_children = ["Propofol"]
    if re_optimized:
        propofol_header_children.append(
            html.Span("Re-optimized", className="maintenance-reoptimized-badge")
        )

    children = [
        html.Div(propofol_header_children, className="maintenance-drug-header maintenance-drug-header--propofol"),
    ]
    prop_cells = _maintenance_rows(
        rec.propofol_inf_rates_ml_h, rec.propofol_inf_rates_mcgkgmin, "µg/kg/min",
    )
    children.append(
        html.Div(prop_cells, className="maintenance-table")
        if prop_cells
        else html.Div("No maintenance", className="maintenance-empty")
    )

    if rec.remifentanil_selected:
        remi_header_children = ["Remifentanil"]
        if re_optimized:
            remi_header_children.append(
                html.Span("Re-optimized", className="maintenance-reoptimized-badge")
            )

        children.append(html.Div(className="maintenance-divider"))
        children.append(
            html.Div(
                remi_header_children,
                className="maintenance-drug-header maintenance-drug-header--remifentanil",
            ),
        )
        remi_cells = _maintenance_rows(
            rec.remifentanil_inf_rates_ml_h, rec.remifentanil_inf_rates_ngkgmin, "ng/kg/min",
        )
        children.append(
            html.Div(remi_cells, className="maintenance-table")
            if remi_cells
            else html.Div("No maintenance", className="maintenance-empty")
        )

    return children


def make_maintenance_card(original, manual=None):
    """
    Build the maintenance-regimen card, showing whichever regimen is
    currently active - the manual override's re-optimized schedule when
    one is active, otherwise the model's own recommendation. Only ever one
    regimen at a time (never original + manual side by side), so the card
    always reflects exactly the dose currently in effect; returning to the
    recommendation (manual=None) reverts this back to the original
    schedule automatically, the same way make_induction_card's dose block
    reverts.
    """
    active_rec = manual if manual is not None else original

    children = [
        html.Div(
            [
                html.H4("Early maintenance regimen", className="maintenance-card-title"),
                html.Span("relative to induction", className="maintenance-card-subtitle"),
            ],
            className="maintenance-card-title-row",
        ),
    ]
    children.extend(_maintenance_section(active_rec, re_optimized=(manual is not None)))

    return html.Div(children, className="card maintenance-card recommendation-dark-card maintenance-regimen-card")


def make_summary(original, manual=None, test_patient_key=None):
    """
    Build the recommendation output: an induction-dose card followed by a
    maintenance-regimen card. `manual`, when provided, is the re-optimized
    manual-override scenario shown alongside the original recommendation.
    test_patient_key is passed straight through to make_induction_card's
    own confidence-override lookup - see its docstring.
    """
    return [
        make_induction_card(original, manual, test_patient_key),
        make_maintenance_card(original, manual),
    ]


# ============================================================
# General figure helpers
# ============================================================

def _add_band(
    fig: go.Figure, x, y_low, y_high, name: str,
    color: str = UNCERTAINTY_BAND_COLOR, show_in_legend: bool = True,
):
    """
    Add a shaded uncertainty band to a Plotly figure, always in the app-wide
    light-blue "uncertainty" color at low opacity so it reads as visually
    secondary to whatever prediction line(s) are drawn on top of it.
    """
    x = _safe_array(x)
    y_low = _safe_array(y_low)
    y_high = _safe_array(y_high)

    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([y_high, y_low[::-1]]),
            fill="toself",
            mode="lines",
            line=dict(width=0),
            fillcolor=color,
            opacity=0.18,
            name=name,
            showlegend=show_in_legend,
            hoverinfo="skip",
        )
    )


def _add_line(fig: go.Figure, x, y, name: str, color: str, width: float = 2.5, dash: str | None = None):
    """
    Add a prediction line to a Plotly figure.
    """
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=name,
            line=dict(color=color, width=width, dash=dash),
        )
    )


def make_propofol_pk_figure(rec):
    """
    Create a Plotly figure for propofol pharmacokinetics (PK).
    """
    fig = go.Figure()
    c = rec.confidence

    _add_band(fig, c.time_min, c.cp_propofol_p05, c.cp_propofol_p95, "90% Prediction Interval")
    _add_line(fig, rec.time_min, rec.cp_propofol, "Recommended Cp", color=RECOMMENDED_COLOR)

    _add_band(fig, c.time_min, c.ce_propofol_p05, c.ce_propofol_p95, "90% Prediction Interval", show_in_legend=False)
    _add_line(fig, rec.time_min, rec.ce_propofol, "Recommended Ce", color=RECOMMENDED_COLOR, dash="dash")

    fig.update_layout(
        xaxis_title="Time (min)",
        yaxis_title="Propofol concentration (mcg/mL)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=60),
    )

    return fig


def make_remifentanil_pk_figure(rec):
    """
    Create a Plotly figure for remifentanil pharmacokinetics (PK).
    """
    if not rec.remifentanil_selected:
        return make_empty_figure()

    fig = go.Figure()
    c = rec.confidence

    if c.cp_remifentanil_p05 is not None and c.cp_remifentanil_p95 is not None:
        _add_band(fig, c.time_min, c.cp_remifentanil_p05, c.cp_remifentanil_p95, "90% Prediction Interval")

    if rec.cp_remifentanil is not None:
        _add_line(fig, rec.time_min, rec.cp_remifentanil, "Recommended Cp", color=RECOMMENDED_COLOR)

    fig.update_layout(
        xaxis_title="Time (min)",
        yaxis_title="Remifentanil concentration (ng/mL)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=60),
    )

    return fig


def make_bis_figure(rec):
    """
    Create a Plotly figure for BIS (Bispectral Index).
    """
    fig = go.Figure()
    c = rec.confidence

    _add_band(fig, c.time_min, c.bis_p05, c.bis_p95, "90% Prediction Interval")
    _add_line(fig, rec.time_min, rec.bis, "Recommended BIS", color=RECOMMENDED_COLOR)

    # Read the actual per-run targets off `rec` (not the module defaults), so
    # the band always matches what this specific recommendation was optimized
    # against, even if the user has since edited the target fields.
    fig.add_hline(
        y=rec.target_bis_low, line_dash="dash",
        annotation_text=f"BIS {rec.target_bis_low:.0f}",
    )
    fig.add_hline(
        y=rec.target_bis_high, line_dash="dash",
        annotation_text=f"BIS {rec.target_bis_high:.0f}",
    )

    fig.update_layout(
        xaxis_title="Time (min)",
        yaxis_title="BIS",
        yaxis=dict(range=[0, 100]),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    return fig


def make_map_figure(rec):
    """
    Create a Plotly figure for mean arterial pressure (MAP).
    """
    fig = go.Figure()
    c = rec.confidence

    _add_band(fig, c.time_min, c.map_p05, c.map_p95, "90% Prediction Interval")
    _add_line(fig, rec.time_min, rec.map_mmhg, "Recommended MAP", color=RECOMMENDED_COLOR)

    fig.add_hline(y=rec.map_lower_bound_mmhg, line_dash="dash", annotation_text="MAP lower bound")

    y_upper = max(160.0, float(np.nanmax(c.map_p95)) + 10.0)

    fig.update_layout(
        xaxis_title="Time (min)",
        yaxis_title="MAP (mmHg)",
        yaxis=dict(range=[0, y_upper]),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    return fig


# ============================================================
# Manual-override dual-trace overlays
#
# Each function starts from the existing single-recommendation figure
# (original recommendation's line style/colors/confidence bands untouched)
# and adds the manual scenario as a deterministic-only orange overlay - no
# confidence band, since confidence is intentionally not computed for the
# manual dose.
# ============================================================

def make_propofol_pk_figure_dual(original, manual, manual_cp_name="Manual Cp", manual_ce_name="Manual Ce"):
    """
    Overlay the manual-override propofol PK trace (orange) on the original
    recommendation's propofol PK figure (unchanged colors/bands).

    manual_cp_name/manual_ce_name default to the Recommendation page's own
    wording ("Manual Cp"/"Manual Ce") - Scenario Exploration passes
    "Scenario Cp"/"Scenario Ce" instead, so the two pages can use different
    legend wording for the same orange overlay without duplicating this
    function.
    """
    fig = make_propofol_pk_figure(original)

    fig.add_trace(go.Scatter(
        x=manual.time_min, y=manual.cp_propofol, mode="lines",
        line=dict(color=MANUAL_OVERRIDE_COLOR, width=2),
        name=manual_cp_name,
    ))
    fig.add_trace(go.Scatter(
        x=manual.time_min, y=manual.ce_propofol, mode="lines",
        line=dict(color=MANUAL_OVERRIDE_COLOR, width=2, dash="dash"),
        name=manual_ce_name,
    ))

    return fig


def make_remifentanil_pk_figure_dual(original, manual, manual_name="Manual Cp"):
    """
    Overlay the manual-override remifentanil PK trace (orange) on the
    original recommendation's remifentanil PK figure, if remifentanil is
    selected in both. Propofol-only overrides don't change remifentanil's
    own PK, but its maintenance schedule can shift during re-optimization,
    so the two curves can differ.

    manual_name defaults to "Manual Cp" (Recommendation page); Scenario
    Exploration passes "Scenario Cp" instead.
    """
    if not original.remifentanil_selected:
        return make_remifentanil_pk_figure(original)

    fig = make_remifentanil_pk_figure(original)

    if manual.remifentanil_selected and manual.cp_remifentanil is not None:
        fig.add_trace(go.Scatter(
            x=manual.time_min, y=manual.cp_remifentanil, mode="lines",
            line=dict(color=MANUAL_OVERRIDE_COLOR, width=2),
            name=manual_name,
        ))

    return fig


def make_bis_figure_dual(original, manual, manual_name="Manual BIS"):
    """
    Overlay the manual-override BIS trace (orange) on the original
    recommendation's BIS figure (unchanged colors/bands/target lines).

    manual_name defaults to "Manual BIS" (Recommendation page); Scenario
    Exploration passes "Scenario BIS" instead.
    """
    fig = make_bis_figure(original)

    fig.add_trace(go.Scatter(
        x=manual.time_min, y=manual.bis, mode="lines",
        line=dict(color=MANUAL_OVERRIDE_COLOR, width=2),
        name=manual_name,
    ))

    return fig


def make_map_figure_dual(original, manual, manual_name="Manual MAP"):
    """
    Overlay the manual-override MAP trace (orange) on the original
    recommendation's MAP figure (unchanged colors/bands/target line).

    manual_name defaults to "Manual MAP" (Recommendation page); Scenario
    Exploration passes "Scenario MAP" instead.
    """
    fig = make_map_figure(original)

    fig.add_trace(go.Scatter(
        x=manual.time_min, y=manual.map_mmhg, mode="lines",
        line=dict(color=MANUAL_OVERRIDE_COLOR, width=2),
        name=manual_name,
    ))

    return fig


# ============================================================
# Dose-rationale graph
# ============================================================

def _dose_axis_limits_mgkg(
    selected_dose_mgkg: float,
    manual_dose_mgkg: float | None = None,
) -> tuple[float, float]:
    """
    Show only the clinically relevant local window around the recommended
    dose: selected dose - 1.5 to selected dose + 1.5 mg/kg, clipped to max
    0.3-5.0 mg/kg. When a manual dose is given, the window widens (if
    needed) so both doses stay visible on the same axis.
    """
    doses = [float(selected_dose_mgkg)]
    if manual_dose_mgkg is not None:
        doses.append(float(manual_dose_mgkg))

    x_min = max(0.30, min(doses) - 1.50)
    x_max = min(5.00, max(doses) + 1.50)

    if x_max <= x_min:
        x_min, x_max = 0.30, 5.00

    return float(x_min), float(x_max)


def _clinical_propofol_dose_grid_mgkg(
    selected_bolus_mgkg: float,
    manual_dose_mgkg: float | None = None,
) -> np.ndarray:
    """
    Build a local propofol induction-dose grid around the selected dose.

    The displayed axis is selected dose ±1.5 mg/kg, clipped to 0.3-5.0 mg/kg
    (widened to also cover manual_dose_mgkg, if given). A 0.10 mg/kg spacing
    gives a smooth rationale curve without making the dashboard too slow.
    """
    x_min, x_max = _dose_axis_limits_mgkg(selected_bolus_mgkg, manual_dose_mgkg)

    extra_points = [float(selected_bolus_mgkg)]
    if manual_dose_mgkg is not None:
        extra_points.append(float(manual_dose_mgkg))

    dose_grid = np.arange(x_min, x_max + 0.001, 0.10)
    dose_grid = np.unique(np.concatenate([dose_grid, extra_points]))
    dose_grid = dose_grid[(dose_grid >= x_min) & (dose_grid <= x_max)]
    dose_grid.sort()

    return dose_grid


def _finite_min(values, default: float) -> float:
    """
    Compute the minimum of finite values, returning a default if no finite values exist.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.min(arr)) if len(arr) else float(default)


def _finite_max(values, default: float) -> float:
    """
    Compute the maximum of finite values, returning a default if no finite values exist.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.max(arr)) if len(arr) else float(default)


def _strict_prediction_axis_range(
    values: np.ndarray,
    lower_floor: float | None = None,
    upper_ceiling: float | None = None,
    default_min: float = 0.0,
    default_max: float = 1.0,
    min_span: float = 1.0,
) -> tuple[float, float]:
    """
    Axis limits are defined by the lowest and highest finite predicted values.

    Optional constraints:
        lower_floor: axis minimum cannot go below this value.
        upper_ceiling: axis maximum cannot go above this value.

    A small fallback span is only applied if all predicted values are equal or
    invalid, because Plotly cannot display a zero-height axis.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]

    if len(arr) == 0:
        axis_min = float(default_min)
        axis_max = float(default_max)
    else:
        axis_min = float(np.min(arr))
        axis_max = float(np.max(arr))

    if lower_floor is not None:
        axis_min = max(float(lower_floor), axis_min)

    if upper_ceiling is not None:
        axis_max = min(float(upper_ceiling), axis_max)

    if not np.isfinite(axis_min) or not np.isfinite(axis_max):
        axis_min = float(default_min)
        axis_max = float(default_max)

    if axis_max <= axis_min:
        center = 0.5 * (axis_min + axis_max)
        half_span = 0.5 * float(min_span)
        axis_min = center - half_span
        axis_max = center + half_span

        if lower_floor is not None:
            axis_min = max(float(lower_floor), axis_min)

        if upper_ceiling is not None:
            axis_max = min(float(upper_ceiling), axis_max)

        if axis_max <= axis_min:
            axis_max = axis_min + float(min_span)

    return float(axis_min), float(axis_max)


def _target_rect_y_limits(
    band_low: float,
    band_high: float,
    axis_min: float,
    axis_max: float,
) -> tuple[float, float] | None:
    """
    Clip a target band to the visible axis range.

    Returns None if the target band is completely outside the current visible
    axis range.
    """
    y0 = max(float(band_low), float(axis_min))
    y1 = min(float(band_high), float(axis_max))

    if y1 <= y0:
        return None

    return y0, y1


def make_induction_dose_rationale_figure(
    patient: Patient,
    rec,
    opiate: str,
    propofol_conc_mg_ml: float,
    remifentanil_conc_mcg_ml: float,
    manual_dose_mgkg: float | None = None,
    remifentanil_rate_override_mcgkgmin: float | None = None,
    compact: bool = False,
):
    """
    Vary only the propofol induction bolus while keeping the recommended
    maintenance regimens fixed.

    X-axis:
        recommended propofol induction dose ±1.5 mg/kg,
        clipped to 0.3-5.0 mg/kg (widened to also cover manual_dose_mgkg,
        if given).

    Left y-axis:
        minimal MAP after target start.

    Right y-axis:
        maximal BIS after target start, reversed.

    BIS interpretation:
        The plotted blue curve is maximal BIS after target start, because
        adequate hypnotic depth is primarily an upper-bound problem:
            max BIS after target start <= 60.

        Minimal BIS is still computed internally to check the lower BIS safety
        boundary:
            min BIS after target start >= 40.

    manual_dose_mgkg, when given, adds a second marker + vertical reference
    line for the manual-override dose. The sweep itself is still built
    around `rec` (the original recommendation) and its maintenance
    schedule, unchanged - only the axis window and the extra marker are
    added, so this stays cheap (one sweep, not two).

    remifentanil_rate_override_mcgkgmin, when given, replaces `rec`'s own
    (optimizer-chosen) remifentanil maintenance rate with this fixed scalar
    for every point of the sweep - used only by the Scenario Exploration
    page, where remifentanil rate is a user-controlled slider rather than
    something the optimizer picked. Defaults to None, which reproduces the
    exact previous behavior (rec's own schedule, unchanged) - the
    Recommendation page's call site never passes this argument.

    compact=True widens the paper-y spacing between the stacked "target
    range" / "Recommendation" / "Manual dose" annotations and the legend
    (and their containing margins) so they stay legible in a much shorter
    card - Scenario Exploration's own dose-response card is roughly 35-40%
    shorter than the Recommendation page's induction-dose-rationale card,
    so the same absolute-pixel margins would otherwise leave far less
    vertical room per stacked row and the two dose annotations would
    overlap. Defaults to False, reproducing the exact previous
    spacing/margins - the Recommendation page's call site never passes
    this argument.
    """
    selected_dose_mgkg = float(rec.propofol_bolus_mgkg)
    baseline_map = float(patient.base_map)
    map_target = float(rec.map_lower_bound_mmhg)
    map_target_upper = float(1.20 * baseline_map)

    x_axis_min, x_axis_max = _dose_axis_limits_mgkg(selected_dose_mgkg, manual_dose_mgkg)
    dose_grid_mgkg = _clinical_propofol_dose_grid_mgkg(selected_dose_mgkg, manual_dose_mgkg)

    # Re-use the exact targets that produced `rec`, not the module defaults or
    # any since-edited (not-yet-run) UI state, so this re-simulated dose grid
    # stays internally consistent with the recommendation being explained.
    runner = Su2023PropofolRemifentanilRecommender(
        patient=patient,
        use_remifentanil=(opiate == "remifentanil"),
        use_bsv=False,
        propofol_conc_mg_ml=propofol_conc_mg_ml,
        remifentanil_conc_mcg_ml=remifentanil_conc_mcg_ml,
        target_bis_low=rec.target_bis_low,
        target_bis_high=rec.target_bis_high,
        map_abs_min_target=rec.map_abs_min_target_mmhg,
        map_rel_frac_target=rec.map_rel_frac_target,
    )

    min_maps = []
    min_bis_values = []
    max_bis_values = []

    if remifentanil_rate_override_mcgkgmin is not None and rec.remifentanil_selected:
        override_rates_mcgkgmin = np.full(N_INTERVALS, float(remifentanil_rate_override_mcgkgmin))
        override_rates_ml_h = remi_mcgkgmin_to_ml_h(
            override_rates_mcgkgmin, patient.weight, remifentanil_conc_mcg_ml,
        )
        override_rates_ngkgmin = remi_mcgkgmin_to_ngkgmin(override_rates_mcgkgmin)
    else:
        override_rates_mcgkgmin = override_rates_ml_h = override_rates_ngkgmin = None

    for dose_mgkg in dose_grid_mgkg:
        regimen = DecodedRegimen(
            propofol_bolus_mg=float(dose_mgkg * patient.weight),
            propofol_bolus_mgkg=float(dose_mgkg),
            propofol_rates_mgkgh=np.asarray(rec.propofol_inf_rates_mgkgh, dtype=float).copy(),
            propofol_rates_ml_h=np.asarray(rec.propofol_inf_rates_ml_h, dtype=float).copy(),
            propofol_rates_mcgkgmin=np.asarray(rec.propofol_inf_rates_mcgkgmin, dtype=float).copy(),
            remifentanil_selected=bool(rec.remifentanil_selected),
            remifentanil_bolus_mcg=float(rec.remifentanil_bolus_mcg),
            remifentanil_bolus_mcgkg=float(rec.remifentanil_bolus_mcgkg),
            remifentanil_rates_mcgkgmin=(
                override_rates_mcgkgmin.copy() if override_rates_mcgkgmin is not None
                else None if rec.remifentanil_inf_rates_mcgkgmin is None
                else np.asarray(rec.remifentanil_inf_rates_mcgkgmin, dtype=float).copy()
            ),
            remifentanil_rates_ml_h=(
                override_rates_ml_h.copy() if override_rates_ml_h is not None
                else None if rec.remifentanil_inf_rates_ml_h is None
                else np.asarray(rec.remifentanil_inf_rates_ml_h, dtype=float).copy()
            ),
            remifentanil_rates_ngkgmin=(
                override_rates_ngkgmin.copy() if override_rates_ngkgmin is not None
                else None if rec.remifentanil_inf_rates_ngkgmin is None
                else np.asarray(rec.remifentanil_inf_rates_ngkgmin, dtype=float).copy()
            ),
        )

        try:
            t, _, _, _, bis, map_mmhg = runner.simulate_regimen(regimen)
            mask = np.asarray(t, dtype=float) >= TARGET_ASSESSMENT_START_MIN

            if not np.any(mask):
                min_maps.append(np.nan)
                min_bis_values.append(np.nan)
                max_bis_values.append(np.nan)
                continue

            bis_after_target = np.asarray(bis, dtype=float)[mask]
            map_after_target = np.asarray(map_mmhg, dtype=float)[mask]

            min_maps.append(float(np.nanmin(map_after_target)))
            min_bis_values.append(float(np.nanmin(bis_after_target)))
            max_bis_values.append(float(np.nanmax(bis_after_target)))

        except Exception:
            min_maps.append(np.nan)
            min_bis_values.append(np.nan)
            max_bis_values.append(np.nan)

    min_maps = np.asarray(min_maps, dtype=float)
    min_bis_values = np.asarray(min_bis_values, dtype=float)
    max_bis_values = np.asarray(max_bis_values, dtype=float)

    selected_idx = int(np.argmin(np.abs(dose_grid_mgkg - selected_dose_mgkg)))
    selected_min_map = float(min_maps[selected_idx])
    selected_min_bis = float(min_bis_values[selected_idx])
    selected_max_bis = float(max_bis_values[selected_idx])

    map_axis_min, map_axis_max = _strict_prediction_axis_range(
        min_maps,
        lower_floor=0.0,
        upper_ceiling=None,
        default_min=0.0,
        default_max=max(1.0, map_target_upper),
        min_span=1.0,
    )

    max_bis_axis_min, max_bis_axis_max = _strict_prediction_axis_range(
        max_bis_values,
        lower_floor=0.0,
        upper_ceiling=100.0,
        default_min=0.0,
        default_max=100.0,
        min_span=1.0,
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # MAP target band: lower target to 20% above baseline, clipped to visible axis.
    map_band = _target_rect_y_limits(
        band_low=map_target,
        band_high=map_target_upper,
        axis_min=map_axis_min,
        axis_max=map_axis_max,
    )
    if map_band is not None:
        fig.add_shape(
            type="rect",
            xref="paper",
            x0=0,
            x1=1,
            yref="y",
            y0=map_band[0],
            y1=map_band[1],
            fillcolor="rgba(220, 0, 0, 0.10)",
            line=dict(width=0),
            layer="below",
        )

    # BIS target band for the plotted maximal BIS curve.
    # Since the blue line is max BIS, the clinically relevant upper boundary is
    # the upper target. The lower boundary is still shown as context, but
    # excessive depth is checked using min_bis_values in both_targets_ok.
    # Uses rec.target_bis_low/high (the actual targets used for this
    # recommendation), not the module defaults.
    bis_band = _target_rect_y_limits(
        band_low=rec.target_bis_low,
        band_high=rec.target_bis_high,
        axis_min=max_bis_axis_min,
        axis_max=max_bis_axis_max,
    )
    if bis_band is not None:
        fig.add_shape(
            type="rect",
            xref="paper",
            x0=0,
            x1=1,
            yref="y2",
            y0=bis_band[0],
            y1=bis_band[1],
            fillcolor="rgba(0, 85, 220, 0.10)",
            line=dict(width=0),
            layer="below",
        )

    # Green band and dotted limits where both MAP and BIS targets are met.
    #
    # MAP target:
    #   minimal MAP between lower MAP target and 20% above baseline.
    #
    # BIS target:
    #   maximal BIS <= upper target, to ensure adequate hypnosis.
    #   minimal BIS >= lower target, to avoid excessive hypnotic depth.
    #
    # Use yref="paper" so the green range remains visible regardless of y-axis zoom.
    both_targets_ok = (
        np.isfinite(min_maps)
        & np.isfinite(min_bis_values)
        & np.isfinite(max_bis_values)
        & (min_maps >= map_target)
        & (min_maps <= map_target_upper)
        & (min_bis_values >= rec.target_bis_low)
        & (max_bis_values <= rec.target_bis_high)
    )

    if np.any(both_targets_ok):
        ok_doses = dose_grid_mgkg[both_targets_ok]
        ok_start = float(np.min(ok_doses))
        ok_end = float(np.max(ok_doses))

        # Shaded dose range where both targets are met.
        # If only a single tested dose meets both targets, draw a very narrow band
        # around that dose so the range remains visible.
        if ok_end > ok_start:
            band_x0 = ok_start
            band_x1 = ok_end
        else:
            local_step = (
                float(np.nanmedian(np.diff(dose_grid_mgkg)))
                if len(dose_grid_mgkg) > 1
                else 0.05
            )
            band_half_width = max(0.025, 0.5 * local_step)
            band_x0 = max(x_axis_min, ok_start - band_half_width)
            band_x1 = min(x_axis_max, ok_end + band_half_width)

        fig.add_shape(
            type="rect",
            xref="x",
            yref="paper",
            x0=band_x0,
            x1=band_x1,
            y0=0,
            y1=1,
            fillcolor="rgba(0, 150, 0, 0.10)",
            line=dict(width=0),
            layer="below",
        )

        # Lower and upper target-dose (optimal range) limits. Deliberately
        # smaller/lighter than the Recommended dose/Manual dose annotations
        # (which float above the plot) and labeled "Min ="/"Max =" rather
        # than bare numbers, so they read as range boundaries rather than
        # competing with the recommendation itself. Sits just inside the
        # top of the plot area (not above it, where the dose annotations
        # live) so the two never stack in the same spot.
        for x_value, label, x_anchor in [
            (ok_start, f"Min = {ok_start:.2f} mg/kg", "right"),
            (ok_end, f"Max = {ok_end:.2f} mg/kg", "left"),
        ]:
            fig.add_annotation(
                x=x_value,
                y=0.97,
                xref="x",
                yref="paper",
                text=label,
                showarrow=False,
                xanchor=x_anchor,
                yanchor="top",
                font=dict(color="green", size=10),
            )

    fig.add_trace(
        go.Scatter(
            x=dose_grid_mgkg,
            y=min_maps,
            mode="lines+markers",
            line=dict(color="red", width=3),
            marker=dict(color="red", size=6),
            name="Predicted MAP",
            showlegend=True,
            hovertemplate="Dose %{x:.2f} mg/kg<br>Minimal MAP %{y:.1f} mmHg<extra></extra>",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=dose_grid_mgkg,
            y=max_bis_values,
            customdata=np.stack([min_bis_values], axis=-1),
            mode="lines+markers",
            line=dict(color="blue", width=3),
            marker=dict(color="blue", size=6),
            name="Predicted BIS",
            showlegend=True,
            hovertemplate=(
                "Dose %{x:.2f} mg/kg<br>"
                "Maximal BIS %{y:.1f}<br>"
                "Minimal BIS %{customdata[0]:.1f}<extra></extra>"
            ),
        ),
        secondary_y=True,
    )

    fig.add_trace(
        go.Scatter(
            x=[selected_dose_mgkg],
            y=[selected_min_map],
            mode="markers",
            marker=dict(color="red", size=14, symbol="diamond"),
            showlegend=False,
            hovertemplate=(
                "Recommendation %{x:.2f} mg/kg<br>"
                "Minimal MAP %{y:.1f} mmHg<br>"
                f"MAP target {map_target:.1f}-{map_target_upper:.1f} mmHg<extra></extra>"
            ),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=[selected_dose_mgkg],
            y=[selected_max_bis],
            customdata=[[selected_min_bis]],
            mode="markers",
            marker=dict(color="blue", size=14, symbol="diamond"),
            showlegend=False,
            hovertemplate=(
                "Recommendation %{x:.2f} mg/kg<br>"
                "Maximal BIS %{y:.1f}<br>"
                "Minimal BIS %{customdata[0]:.1f}<br>"
                f"BIS target {rec.target_bis_low:.0f}-{rec.target_bis_high:.0f}<extra></extra>"
            ),
        ),
        secondary_y=True,
    )

    # Legend-only entries for the shapes below (MAP/BIS target bands, optimal
    # dose range, recommendation line, manual-override line) - Plotly shapes
    # never appear in the legend on their own, so a zero-data dummy trace
    # styled to match is the standard way to add one. Purely presentational:
    # none of these affect the plotted data. Marker colors use a higher
    # alpha than the actual bands (which are deliberately faint, ~0.10, so
    # the curves stay readable) so each swatch still reads clearly at
    # legend size.
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(symbol="square", size=10, color="rgba(220, 0, 0, 0.25)"),
            name="MAP target zone",
            showlegend=True,
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(symbol="square", size=10, color="rgba(0, 85, 220, 0.25)"),
            name="BIS target zone",
            showlegend=True,
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(symbol="square", size=10, color="rgba(0, 150, 0, 0.25)"),
            name="Optimal dose range",
            showlegend=True,
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(color="green", width=3, dash="dash"),
            name="Recommended dose",
            showlegend=True,
        ),
    )
    if manual_dose_mgkg is not None:
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="lines",
                line=dict(color=MANUAL_OVERRIDE_COLOR, width=3, dash="dash"),
                name="Manual dose",
                showlegend=True,
            ),
        )

    # Recommended induction dose reference line, shown in bold green.
    fig.add_shape(
        type="line",
        xref="x",
        yref="paper",
        x0=selected_dose_mgkg,
        x1=selected_dose_mgkg,
        y0=0,
        y1=1,
        line=dict(color="green", width=3, dash="dash"),
        layer="above",
    )
    # Recommended-dose annotation: a two-line "label / value" callout with a
    # vertical arrow connecting it straight down to the dashed reference
    # line, so it reads as one dominant, clearly-named annotation rather
    # than another bare green number sitting next to the (deliberately
    # smaller/lighter) Min=/Max= range labels near the top of the plot.
    # compact mode's card is ~35-40% shorter than the Recommendation page's
    # own card (see docstring), so its arrow/font are scaled down too - not
    # just the margins - to keep both annotations (plus the two-row legend)
    # fitting inside that much smaller absolute pixel budget.
    rec_arrow_ay = -26 if compact else -46
    rec_font_size = 11 if compact else 13

    fig.add_annotation(
        x=selected_dose_mgkg,
        y=1,
        xref="x",
        yref="paper",
        ax=0,
        ay=rec_arrow_ay,
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="green",
        text=f"<b>Recommended dose</b><br>{selected_dose_mgkg:.2f} mg/kg",
        align="center",
        yanchor="bottom",
        font=dict(color="green", size=rec_font_size),
    )

    # Manual-override dose, shown in orange, when active.
    if manual_dose_mgkg is not None:
        manual_idx = int(np.argmin(np.abs(dose_grid_mgkg - manual_dose_mgkg)))
        fig.add_trace(
            go.Scatter(
                x=[manual_dose_mgkg],
                y=[float(min_maps[manual_idx])],
                mode="markers",
                marker=dict(color=MANUAL_OVERRIDE_COLOR, size=14, symbol="diamond"),
                showlegend=False,
                hovertemplate="Manual dose %{x:.2f} mg/kg<br>Minimal MAP %{y:.1f} mmHg<extra></extra>",
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=[manual_dose_mgkg],
                y=[float(max_bis_values[manual_idx])],
                mode="markers",
                marker=dict(color=MANUAL_OVERRIDE_COLOR, size=14, symbol="diamond"),
                showlegend=False,
                hovertemplate="Manual dose %{x:.2f} mg/kg<br>Maximal BIS %{y:.1f}<extra></extra>",
            ),
            secondary_y=True,
        )
        fig.add_shape(
            type="line",
            xref="x",
            yref="paper",
            x0=manual_dose_mgkg,
            x1=manual_dose_mgkg,
            y0=0,
            y1=1,
            line=dict(color=MANUAL_OVERRIDE_COLOR, width=3, dash="dash"),
            layer="above",
        )
        # Manual-dose annotation: same two-line/arrow style as the
        # recommendation, stacked above it by default (different arrow
        # length, same x-referenced arrowhead) so the two never occupy the
        # same spot. When the two doses sit close together on the x-axis,
        # also nudge this label sideways (away from whichever side it's
        # already leaning toward) so the two callouts/arrows don't visually
        # collide even when their x-positions are nearly identical.
        manual_close = abs(manual_dose_mgkg - selected_dose_mgkg) < 0.08 * (x_axis_max - x_axis_min)
        manual_arrow_ax = 0
        if manual_close:
            manual_arrow_ax = -55 if manual_dose_mgkg <= selected_dose_mgkg else 55
        manual_arrow_ay = rec_arrow_ay - (34 if compact else 64)
        manual_font_size = 10 if compact else 12

        fig.add_annotation(
            x=manual_dose_mgkg,
            y=1,
            xref="x",
            yref="paper",
            ax=manual_arrow_ax,
            ay=manual_arrow_ay,
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=MANUAL_OVERRIDE_COLOR,
            text=f"<b>Manual dose</b><br>{manual_dose_mgkg:.2f} mg/kg",
            align="center",
            yanchor="bottom",
            font=dict(color=MANUAL_OVERRIDE_COLOR, size=manual_font_size),
        )

    # The legend sits above the plot, stacked above the "Recommendation"/
    # "Manual dose" annotations rather than below the plot - this reclaims
    # the large bottom margin that previously existed only to hold it,
    # which is what let the card shrink below without losing any label.
    # In compact mode the rows are spaced further apart in paper-y (see
    # rec_annotation_y/manual_dose's y above) and margin.t is enlarged to
    # match, so a much shorter card still has enough absolute pixels
    # between each stacked row to stay legible.
    if compact:
        legend_y = 1.92 if manual_dose_mgkg is not None else 1.65
        margin_t = 135 if manual_dose_mgkg is not None else 105
        margin_b = 35
    else:
        legend_y = 2.41 if manual_dose_mgkg is not None else 1.50
        margin_t = 205 if manual_dose_mgkg is not None else 140
        margin_b = 42

    fig.update_layout(
        # No in-plot title text - the card header ("Induction-dose rationale")
        # already shows it in black, so a second grey title here was a
        # redundant duplicate.
        template="plotly_white",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=legend_y,
            xanchor="left",
            x=0,
            font=dict(size=9),
            tracegroupgap=2,
        ),
        margin=dict(
            t=margin_t,
            b=margin_b,
            l=48,
            r=38,
        ),
        xaxis=dict(
            title=dict(text="Propofol induction dose (mg/kg)", font=dict(color="green")),
            tickfont=dict(color="green"),
            color="green",
            range=[x_axis_min, x_axis_max],
        ),
        # The "which way is more" direction for each y-axis is shown as a
        # small arrow glyph appended directly to its own title text, rather
        # than a separately-positioned annotation arrow (the previous
        # approach: floating annotations placed by hand-tuned pixel
        # offsets, which drifted into overlapping the rotated axis-title
        # text on the left and rendered as a barely-visible, disconnected
        # triangle on the right). Baking the arrow into the title string
        # guarantees it's always correctly positioned and colored, since
        # it's laid out as part of the same text run - no separate
        # positioning to get wrong.
        yaxis=dict(
            title=dict(text="Minimal MAP (mmHg) ↑", font=dict(color="red")),
            tickfont=dict(color="red"),
            color="red",
            range=[map_axis_min, map_axis_max],
        ),
        yaxis2=dict(
            # Points down: this axis is visually inverted, so lower on the
            # page is a higher BIS value.
            title=dict(text="Maximal BIS ↓", font=dict(color="blue")),
            tickfont=dict(color="blue"),
            color="blue",
            range=[max_bis_axis_max, max_bis_axis_min],
            overlaying="y",
            side="right",
        ),
    )

    return fig


# ============================================================
# Sex dropdown callback
# ============================================================

@app.callback(
    Output("sex-store", "data"),
    Input("sex-dropdown", "value"),
)
def update_sex(value):
    """
    Mirror the Sex dropdown's selected value into sex-store, which
    run_model reads exactly as it did when Sex was a pair of toggle
    buttons.
    """
    return value or "male"


@app.callback(
    Output("remifentanil-concentration-row", "style"),
    Input("opiate-dropdown", "value"),
)
def toggle_remifentanil_concentration_row(opiate):
    """
    Show the Remifentanil (µg/mL) concentration row only when Remifentanil
    is the selected opiate - hidden (not removed) otherwise, so
    run_model's own State("remifentanil-concentration", "value") always
    resolves regardless of which opiate is selected. Fires on page load
    too (no prevent_initial_call), matching the layout's own default
    (hidden, since opiate-dropdown defaults to "none").
    """
    return None if opiate == "remifentanil" else {"display": "none"}


# ============================================================
# Per-graph "Show" visibility toggle
#
# One generic pattern-matching callback (MATCH on the "graph" key of the
# dict ids graph_card() gives its checkbox/content/card) handles every
# graph card uniformly - Induction-dose rationale, BIS, MAP, Propofol
# PK/PD, and Remifentanil PK - driven only by that card's own checkbox.
# It only toggles a wrapper div's display style and a CSS class; it never
# touches a graph's `figure` prop, so graph generation, the recommendation
# pipeline, and the manual-override pipeline are completely unaffected by
# whether a card happens to be shown or hidden.
# ============================================================

def _without_collapsed_class(class_name: str | None) -> list[str]:
    """
    Return a card's className tokens with any previous "collapsed" marker
    removed, so it can be safely added back or left off.
    """
    return [c for c in (class_name or "").split() if c != "graph-card--collapsed"]


@app.callback(
    Output({"type": "graph-card-content", "graph": MATCH}, "style"),
    Output({"type": "graph-card", "graph": MATCH}, "className"),
    Input({"type": "graph-visibility-toggle", "graph": MATCH}, "value"),
    State({"type": "graph-card", "graph": MATCH}, "className"),
    prevent_initial_call=True,
)
def toggle_graph_visibility(checked_values, current_class_name):
    """
    Show/hide one graph card's content based on its own "Show" checkbox.
    When hidden, the card collapses to just its header (via
    graph-card--collapsed overriding the fixed card height) so no empty
    graph space remains.
    """
    visible = bool(checked_values) and "show" in checked_values

    classes = _without_collapsed_class(current_class_name)
    if not visible:
        classes.append("graph-card--collapsed")

    content_style = None if visible else {"display": "none"}
    return content_style, " ".join(classes)


# ============================================================
# Page navigation (Recommendation <-> Scenario Exploration <-> More Info)
#
# The three pages are always-mounted siblings (built once in build_layout);
# switching between them only ever toggles which one's `style` is
# display:none, exactly like every other show/hide toggle in this app. No
# page carries any model state, so this cannot affect the recommendation
# pipeline, the manual-override pipeline, the Scenario Exploration
# pipeline, or any stored data.
# ============================================================

@app.callback(
    Output("active-page-store", "data"),
    Input("nav-recommendation-btn", "n_clicks"),
    Input("nav-scenario-btn", "n_clicks"),
    Input("nav-test-exploration-btn", "n_clicks"),
    Input("nav-precomputed-remi-btn", "n_clicks"),
    Input("nav-more-info-btn", "n_clicks"),
    Input("back-to-recommendation-btn", "n_clicks"),
    Input("learn-more-link-btn", "n_clicks"),
    Input("te-learn-more-link", "n_clicks"),
    prevent_initial_call=True,
)
def set_active_page(
    rec_nav_clicks, scenario_nav_clicks, test_exploration_nav_clicks, precomputed_remi_nav_clicks,
    more_info_nav_clicks, back_clicks, learn_more_clicks, te_learn_more_clicks,
):
    """
    Track which top-level page is active based on which nav/back control was
    clicked. learn-more-link-btn and te-learn-more-link (the Recommendation
    and Test Exploration pages' own "Learn more about the model" links) are
    just extra entry points to the same "more-info" page the sidebar's
    "More Info" item already switches to.
    """
    triggered = ctx.triggered_id
    if triggered in ("nav-more-info-btn", "learn-more-link-btn", "te-learn-more-link"):
        return "more-info"
    if triggered == "nav-scenario-btn":
        return "scenario-exploration"
    if triggered == "nav-test-exploration-btn":
        return "test-exploration"
    if triggered == "nav-precomputed-remi-btn":
        return "precomputed-remi"
    if triggered in ("nav-recommendation-btn", "back-to-recommendation-btn"):
        return "recommendation"
    return no_update


@app.callback(
    Output("main-content", "style"),
    Output("scenario-exploration-view", "style"),
    Output("test-exploration-view", "style"),
    Output("precomputed-remi-view", "style"),
    Output("more-info-view", "style"),
    Output("nav-recommendation-btn", "className"),
    Output("nav-scenario-btn", "className"),
    Output("nav-test-exploration-btn", "className"),
    Output("nav-precomputed-remi-btn", "className"),
    Output("nav-more-info-btn", "className"),
    Input("active-page-store", "data"),
)
def render_active_page(active_page):
    """
    Show exactly one of the five pages, and keep the sidebar's active
    highlight in sync with it.
    """
    is_recommendation = active_page not in (
        "more-info", "scenario-exploration", "test-exploration", "precomputed-remi",
    )
    is_scenario = active_page == "scenario-exploration"
    is_test_exploration = active_page == "test-exploration"
    is_precomputed_remi = active_page == "precomputed-remi"
    is_more_info = active_page == "more-info"
    hidden = {"display": "none"}

    def cls(active: bool) -> str:
        return "sidebar-nav-item sidebar-nav-item--active" if active else "sidebar-nav-item"

    return (
        None if is_recommendation else hidden,
        None if is_scenario else hidden,
        None if is_test_exploration else hidden,
        None if is_precomputed_remi else hidden,
        None if is_more_info else hidden,
        cls(is_recommendation),
        cls(is_scenario),
        cls(is_test_exploration),
        cls(is_precomputed_remi),
        cls(is_more_info),
    )


# ============================================================
# "More Info" accordion (multi-expand: any number of sections open at once)
# ============================================================

@app.callback(
    Output("more-info-open-sections", "data"),
    Input({"type": "more-info-accordion-header", "section": ALL}, "n_clicks"),
    State("more-info-open-sections", "data"),
    prevent_initial_call=True,
)
def set_open_more_info_section(_all_n_clicks, current_open_sections):
    """
    Toggle only the clicked section's own membership in the open-sections
    list - opening one section never closes any other, so any number of
    sections can be open at the same time.
    """
    triggered = ctx.triggered_id
    if triggered is None:
        return no_update

    clicked_section = triggered["section"]
    open_sections = list(current_open_sections or [])
    if clicked_section in open_sections:
        open_sections.remove(clicked_section)
    else:
        open_sections.append(clicked_section)
    return open_sections


@app.callback(
    Output({"type": "more-info-accordion-body", "section": ALL}, "className"),
    Output({"type": "more-info-accordion-chevron", "section": ALL}, "className"),
    Input("more-info-open-sections", "data"),
    State({"type": "more-info-accordion-body", "section": ALL}, "id"),
    State({"type": "more-info-accordion-chevron", "section": ALL}, "id"),
)
def render_more_info_accordion(open_sections, body_ids, chevron_ids):
    """
    Render the open/collapsed state for every section from a single source
    of truth (more-info-open-sections, a list of open ids) - any section
    whose id is in the list is expanded, independently of the others. Each
    body/chevron's own pattern-matching id already carries its section, so
    no cross-referencing between the two is needed.
    """
    open_sections = open_sections or []
    body_classes = [
        "accordion-body" if body_id["section"] in open_sections
        else "accordion-body accordion-body--collapsed"
        for body_id in body_ids
    ]
    chevron_classes = [
        "accordion-chevron accordion-chevron--open" if chevron_id["section"] in open_sections
        else "accordion-chevron"
        for chevron_id in chevron_ids
    ]
    return body_classes, chevron_classes


# ============================================================
# Test Exploration (fully independent UI sandbox - no model logic)
#
# Every callback below is scoped only to te-* ids. None of them read or
# write recommendation-store, se-baseline-store, or any other page's
# state, and none of them call into recommend_regimen2023 or any other
# PK/PD/optimization module - MAP/PP are the only "calculations" here,
# and they're the same plain arithmetic (compute_map/compute_pp) already
# used for the Recommendation page's own derived-pressure display.
# ============================================================

def _register_te_input_clamp(input_id: str, bounds: tuple[float, float]):
    """
    Clamp a Test Exploration patient-field text input to its valid range
    on blur/Enter (debounce=True on the input already defers firing until
    then) - the same range-limiting behavior the old slider<->input sync
    used to provide, now standalone since there is no slider to enforce
    it (the input is the sole control for this field).
    """
    @app.callback(
        Output(input_id, "value", allow_duplicate=True),
        Input(input_id, "value"),
        prevent_initial_call=True,
    )
    def _te_clamp_input(value):
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return no_update
        clamped = max(bounds[0], min(bounds[1], parsed))
        if abs(clamped - parsed) < 1e-9:
            return no_update
        return f"{clamped:.0f}"

    _te_clamp_input.__name__ = f"clamp_{input_id.replace('-', '_')}"


_register_te_input_clamp("te-age-input", (18, 100))
_register_te_input_clamp("te-height-input", (140, 220))
_register_te_input_clamp("te-weight-input", (40, 200))
_register_te_input_clamp("te-sbp-input", (80, 220))
_register_te_input_clamp("te-dbp-input", (40, 140))


@app.callback(
    Output("te-derived-map", "children"),
    Output("te-derived-pp", "children"),
    Input("te-sbp-input", "value"),
    Input("te-dbp-input", "value"),
)
def update_te_derived_values(sbp, dbp):
    """Plain-arithmetic MAP/PP display for the Test Exploration sandbox - never touches any model code."""
    try:
        sbp = float(sbp)
        dbp = float(dbp)
        return f"{compute_map(sbp, dbp):.1f}", f"{compute_pp(sbp, dbp):.1f}"
    except (TypeError, ValueError):
        return "-", "-"


@app.callback(
    Output("te-map-target-value", "value"),
    Input("te-map-target-mode", "value"),
    prevent_initial_call=True,
)
def reset_te_map_target_value(mode):
    """
    Switching Target MAP mode shows that mode's own default value, rather
    than leaving (say) 65 on screen after switching to "Relative (%
    baseline)", where 65 would be a nonsensical percentage - the same
    presentation-only behavior the Scenario Exploration page's own
    reset_scenario_map_target_value already has, just against this
    page's own TEST_EXPLORATION_DEFAULT_MAP_ABS/REL constants instead of
    the real model's MAP_ABS_MIN_TARGET/MAP_REL_FRAC_TARGET. Purely a
    displayed-default swap - no recommendation/graph/MAP-target
    calculation is touched here.
    """
    return TEST_EXPLORATION_DEFAULT_MAP_REL if mode == "rel" else TEST_EXPLORATION_DEFAULT_MAP_ABS


@app.callback(
    Output("te-age-input", "value", allow_duplicate=True),
    Output("te-sex", "value"),
    Output("te-height-input", "value", allow_duplicate=True),
    Output("te-weight-input", "value", allow_duplicate=True),
    Output("te-sbp-input", "value", allow_duplicate=True),
    Output("te-dbp-input", "value", allow_duplicate=True),
    Input("te-restore-btn", "n_clicks"),
    prevent_initial_call=True,
)
def restore_te_patient(_n_clicks):
    """Reset every Test Exploration patient field back to TEST_EXPLORATION_DEFAULT_PATIENT."""
    d = TEST_EXPLORATION_DEFAULT_PATIENT
    return d["age"], d["sex"], d["height"], d["weight"], d["sbp"], d["dbp"]


# Patient/Medication/Targets fields (the left column) are pure form
# state - editing them never recomputes anything by itself; they only
# mark the scenario stale (markScenarioStale), snapping
# te-scenario-active-store back to False and hiding the whole results
# area again. "Run Scenario" alone commits the left column's current
# values into te-committed-scenario-store; everything downstream (the
# baseline column, the Explore-strategy slider's own bounds, and
# whatever the slider itself computes) reads that store, never the live
# form fields directly. All in window.dash_clientside.testExploration in
# assets/test_exploration.js - no server round-trip, no model code.
_TE_LEFT_COLUMN_FIELDS = (
    ("te-age-input", "value"),
    ("te-sex", "value"),
    ("te-height-input", "value"),
    ("te-weight-input", "value"),
    ("te-sbp-input", "value"),
    ("te-dbp-input", "value"),
    ("te-opioid-dropdown", "value"),
    ("te-bis-target-low", "value"),
    ("te-bis-target-high", "value"),
    ("te-map-target-mode", "value"),
    ("te-map-target-value", "value"),
)

# "Run Scenario" click: commit the left column's current values into
# te-committed-scenario-store (the Baseline Recommendation from now on -
# whatever opioid the left column held, including "none"), reset the
# fully independent Explore-opioid dropdown back to "none" (a fresh
# scenario always starts with no exploration active), and flip
# te-scenario-active-store to True (switching the page into State B -
# see render_te_results_area below). Nothing else is computed directly
# here - committing the store triggers computeBaseline below, and
# resetting the dropdown triggers onExploreOpioidChange below (which
# hides the rate slider and, via its own reset of the slider's value,
# ultimately re-triggers updateExploredLive), so the whole Scenario
# Result card ends up freshly filled through that natural chain reaction.
app.clientside_callback(
    ClientsideFunction(namespace="testExploration", function_name="runScenario"),
    Output("te-committed-scenario-store", "data"),
    Output("te-explore-opioid-dropdown", "value"),
    Output("te-scenario-active-store", "data", allow_duplicate=True),
    Input("te-run-scenario-btn", "n_clicks"),
    *[State(field_id, prop) for field_id, prop in _TE_LEFT_COLUMN_FIELDS],
    prevent_initial_call=True,
)

# Baseline column - driven solely by te-committed-scenario-store, so it
# fires once on page load (showing a real baseline immediately, per this
# page's default committed snapshot) and again every time "Run Scenario"
# commits a new one. It never reacts to the live left-column fields, and
# never reacts to the Explore-opioid dropdown - this is the Baseline
# Recommendation, unaffected by whatever is being explored.
app.clientside_callback(
    ClientsideFunction(namespace="testExploration", function_name="computeBaseline"),
    Output("te-induction-dose", "children"),
    Output("te-induction-total", "children"),
    Output("te-baseline-prop-rate1", "children"),
    Output("te-baseline-prop-rate2", "children"),
    Output("te-baseline-remi-rate", "children"),
    Input("te-committed-scenario-store", "data"),
)

# The Explore Opioid Strategy card's own opioid dropdown - fully
# independent from the left column's Medication Scenario dropdown.
# Choosing "None" hides the rate slider (nothing to explore); choosing a
# real opioid reconfigures the slider for that opioid's own bounds/step
# and resets its value to that opioid's recommended rate (the starting
# point updateExploredLive below treats as "not explored yet").
app.clientside_callback(
    ClientsideFunction(namespace="testExploration", function_name="onExploreOpioidChange"),
    Output("te-explore-rate-slider", "min"),
    Output("te-explore-rate-slider", "max"),
    Output("te-explore-rate-slider", "step"),
    Output("te-explore-rate-slider", "value"),
    Output("te-explore-rate-slider", "marks"),
    Output("te-explore-rate-title", "children"),
    Output("te-explore-rate-label", "children", allow_duplicate=True),
    Output("te-explore-slider-wrapper", "style"),
    Input("te-explore-opioid-dropdown", "value"),
    prevent_initial_call=True,
)

# The Explore-strategy live-update path - fires on every slider drag
# tick, every Explore-opioid dropdown change, and every "Run Scenario"
# commit (all three are Inputs, not State). Reads
# te-committed-scenario-store for the Baseline Recommendation (never the
# live left-column fields) and the live Explore-opioid dropdown + slider
# for the Explored Scenario - two fully independent opioid choices, so
# e.g. exploring "remifentanil" against a "no opioid" baseline works
# without needing "Run Scenario" again.
app.clientside_callback(
    ClientsideFunction(namespace="testExploration", function_name="updateExploredLive"),
    Output("te-explored-induction", "children", allow_duplicate=True),
    Output("te-explored-induction", "className", allow_duplicate=True),
    Output("te-explored-induction-total", "children", allow_duplicate=True),
    Output("te-explored-prop-pause", "children", allow_duplicate=True),
    Output("te-explored-prop-rate1", "children", allow_duplicate=True),
    Output("te-explored-prop-rate2", "children", allow_duplicate=True),
    Output("te-explored-remi-pause", "children", allow_duplicate=True),
    Output("te-explored-remi-rate", "children", allow_duplicate=True),
    Output("te-result-delta-propofol", "children", allow_duplicate=True),
    Output("te-result-delta-propofol", "className", allow_duplicate=True),
    Output("te-result-delta-prop-rate1", "children"),
    Output("te-result-delta-prop-rate1", "className"),
    Output("te-result-delta-prop-rate2", "children"),
    Output("te-result-delta-prop-rate2", "className"),
    Output("te-result-delta-opioid", "children", allow_duplicate=True),
    Output("te-result-delta-opioid", "className", allow_duplicate=True),
    Output("te-dose-response-graph", "figure"),
    Output("te-bis-graph", "figure"),
    Output("te-map-graph", "figure"),
    Output("te-propofol-pk-graph", "figure"),
    Output("te-opioid-pk-graph", "figure"),
    Output("te-strategy-baseline-value", "children"),
    Output("te-strategy-explored-value", "children"),
    Input("te-explore-rate-slider", "value"),
    Input("te-committed-scenario-store", "data"),
    Input("te-explore-opioid-dropdown", "value"),
    prevent_initial_call=True,
)

# Live label above the slider - fires on every drag tick (cheap string
# formatting only), reading the live Explore-opioid dropdown (not the
# committed baseline) so its unit/decimals always match whatever the
# slider is actually configured for.
app.clientside_callback(
    ClientsideFunction(namespace="testExploration", function_name="updateExploreLiveLabel"),
    Output("te-explore-rate-label", "children", allow_duplicate=True),
    Input("te-explore-rate-slider", "value"),
    State("te-explore-opioid-dropdown", "value"),
    prevent_initial_call=True,
)

# Editing ANY Patient/Medication/Targets field while a scenario is
# active immediately snaps te-scenario-active-store back to False -
# hiding Scenario Result, Explore Opioid Strategy, Dose-Response, and all
# 4 prediction graphs again (see render_te_results_area below), so no
# stale result from the previous scenario is ever left on screen. The
# user must click "Run Scenario" again to see results for the new values.
app.clientside_callback(
    ClientsideFunction(namespace="testExploration", function_name="markScenarioStale"),
    Output("te-scenario-active-store", "data", allow_duplicate=True),
    *[Input(field_id, prop) for field_id, prop in _TE_LEFT_COLUMN_FIELDS],
    prevent_initial_call=True,
)

# The page's two-state switch: te-scenario-active-store is the single
# boolean source of truth for whether te-results-area (Scenario Result,
# Explore Opioid Strategy, Dose-Response, and the 4 prediction graphs) is
# rendered at all. Fires on page load too (no prevent_initial_call), so
# the results area starts hidden per this page's default (inactive)
# store value - the left column stays visible regardless, since it lives
# outside te-results-area entirely.
app.clientside_callback(
    ClientsideFunction(namespace="testExploration", function_name="renderTeResultsArea"),
    Output("te-results-area", "style"),
    Input("te-scenario-active-store", "data"),
)


# ============================================================
# Derived pressure callback
# ============================================================

@app.callback(
    Output("derived-map", "children"),
    Output("derived-pp", "children"),
    Input("baseline_sap", "value"),
    Input("baseline_dap", "value"),
)
def update_derived_pressures(baseline_sap, baseline_dap):
    """
    Update the derived mean arterial pressure (MAP) and pulse pressure (PP)
    based on the baseline systolic and diastolic arterial pressures.
    """
    if baseline_sap is None or baseline_dap is None:
        return "-", "-"

    try:
        baseline_sap = float(baseline_sap)
        baseline_dap = float(baseline_dap)

        if baseline_sap <= 0 or baseline_dap <= 0 or baseline_sap <= baseline_dap:
            return "-", "-"

        return (
            f"{compute_map(baseline_sap, baseline_dap):.1f}",
            f"{compute_pp(baseline_sap, baseline_dap):.1f}",
        )

    except Exception:
        return "-", "-"


# ============================================================
# Click-to-edit popover callbacks for patient-parameter fields (age, height,
# weight, baseline SAP/DAP/HR). Each field keeps its existing id/value prop -
# the popover only ever writes back into that same id, so `run_model`'s
# `State(field_id, "value")` reads are unaffected.
# ============================================================

# PRESET_TEST_PATIENTS itself now lives in dashboard_layout.py (imported
# below) - build_patient_parameters_card there needs it directly to
# default the Recommendation page's own input fields to Test Patient 1,
# and dashboard_layout.py can't import from app.py (app.py already
# imports from dashboard_layout.py) without a circular import. Re-
# exported here unchanged so scripts/precompute_remi_cases.py's own
# `from propofol.app import PRESET_TEST_PATIENTS` keeps working exactly
# as before.
_DEFAULT_PATIENT = PRESET_TEST_PATIENTS["1"]

EDITABLE_FIELDS = [
    ("age", _DEFAULT_PATIENT["age"], "EHR"),
    ("height", _DEFAULT_PATIENT["height"], "EHR"),
    ("weight", _DEFAULT_PATIENT["weight"], "EHR"),
    ("baseline_sap", _DEFAULT_PATIENT["baseline_sap"], "Monitor"),
    ("baseline_dap", _DEFAULT_PATIENT["baseline_dap"], "Monitor"),
    ("baseline_hr", _DEFAULT_PATIENT["baseline_hr"], "Monitor"),
    ("bis-target-low", float(TARGET_BIS_LOW), "Default"),
    ("bis-target-high", float(TARGET_BIS_HIGH), "Default"),
    ("map-target-abs", float(MAP_ABS_MIN_TARGET), "Default"),
    ("map-target-rel", float(MAP_REL_FRAC_TARGET) * 100.0, "Default"),
    ("propofol-concentration", float(DEFAULT_PROPOFOL_CONC_MG_ML), "Default"),
    ("remifentanil-concentration", float(DEFAULT_REMI_CONC_MCG_ML), "Default"),
]

# Validation bounds shown in the edit popover. Hard limits block Save;
# typical-range (warn) limits only show a warning and still allow Save.
# These are presentation/UX rules only - they do not by themselves feed into
# run_model (the BIS/MAP target fields are the exception: run_model reads
# their *saved* value as a State, same as every other editable field).
#
# cross_check_field/cross_check_kind generalizes a two-field ordering rule
# ("this field must be below/above that field"), used for SBP/DBP and for
# the BIS lower/upper target pair.
FIELD_RULES = {
    "age": dict(hard_min=0, hard_max=120, warn_min=None, warn_max=100, label="Age", unit="years"),
    "height": dict(
        hard_min=30, hard_max=250, warn_min=120, warn_max=220, label="Height", unit="cm",
    ),
    "weight": dict(
        hard_min=0.5, hard_max=300, warn_min=30, warn_max=250, label="Weight", unit="kg",
    ),
    "baseline_sap": dict(
        hard_min=30, hard_max=300, warn_min=70, warn_max=220, label="SBP", unit="mmHg",
    ),
    "baseline_dap": dict(
        hard_min=10, hard_max=200, warn_min=40, warn_max=140, label="DBP", unit="mmHg",
        cross_check_field="baseline_sap", cross_check_kind="below",
    ),
    "baseline_hr": dict(
        hard_min=0, hard_max=250, warn_min=40, warn_max=180, label="HR", unit="bpm",
    ),
    "bis-target-low": dict(
        hard_min=0, hard_max=100, warn_min=20, warn_max=50,
        label="BIS target (lower)", unit="",
        cross_check_field="bis-target-high", cross_check_kind="below",
    ),
    "bis-target-high": dict(
        hard_min=0, hard_max=100, warn_min=50, warn_max=80,
        label="BIS target (upper)", unit="",
        cross_check_field="bis-target-low", cross_check_kind="above",
    ),
    "map-target-abs": dict(
        hard_min=30, hard_max=150, warn_min=55, warn_max=90,
        label="MAP target (absolute)", unit="mmHg",
    ),
    "map-target-rel": dict(
        hard_min=30, hard_max=100, warn_min=50, warn_max=90,
        label="MAP target (relative)", unit="%",
    ),
    # hard_min is a small positive floor (not 0) so an exact 0 is rejected,
    # matching the "must be positive" rule the old dropdown+custom flow
    # enforced server-side in parse_concentration.
    "propofol-concentration": dict(
        hard_min=0.1, hard_max=100, warn_min=5, warn_max=30,
        label="Propofol concentration", unit="mg/mL",
    ),
    "remifentanil-concentration": dict(
        hard_min=0.1, hard_max=500, warn_min=5, warn_max=100,
        label="Remifentanil concentration", unit="µg/mL",
    ),
}


def _unit_suffix(unit: str) -> str:
    """
    Return a leading-space unit suffix, or "" for dimensionless fields (e.g. BIS).
    """
    return f" {unit}" if unit else ""


def _check_hard_bounds(value: float, rules: dict) -> str | None:
    """
    Return an error message if value violates the field's hard min/max, else None.
    """
    label, unit = rules["label"], _unit_suffix(rules["unit"])
    if value < rules["hard_min"]:
        if rules["hard_min"] == 0:
            return f"{label} cannot be negative."
        return f"{label} must be at least {rules['hard_min']}{unit}."
    if value > rules["hard_max"]:
        return f"{label} cannot exceed {rules['hard_max']}{unit}."
    return None


def _check_cross_field(value: float, other_value, kind: str, this_label: str, other_label: str):
    """
    Return an error message if value does not satisfy the ordering constraint
    ("below" or "above") relative to another field's current value, else None.
    """
    try:
        other = float(other_value)
    except (TypeError, ValueError):
        return None
    if kind == "below" and value >= other:
        return f"{this_label} must be lower than {other_label}."
    if kind == "above" and value <= other:
        return f"{this_label} must be higher than {other_label}."
    return None


def _check_warn_bounds(value: float, rules: dict):
    """
    Return ("warning", message) outside the typical range, else ("normal", "").
    """
    unit = _unit_suffix(rules["unit"])
    warn_min = rules.get("warn_min")
    warn_max = rules.get("warn_max")
    if warn_min is not None and value < warn_min:
        return "warning", (
            f"This value is outside the typical range ({warn_min}–{rules['hard_max']}"
            f"{unit}). Please verify that it has been entered correctly."
        )
    if warn_max is not None and value > warn_max:
        lo = warn_min if warn_min is not None else rules["hard_min"]
        return "warning", (
            f"This value is outside the typical range ({lo}–{warn_max}{unit}). "
            f"Please verify that it has been entered correctly."
        )
    return "normal", ""


def _validate_field_value(field_id: str, raw_value, other_value=None):
    """
    Validate a draft value for an editable field against FIELD_RULES.

    Returns (status, message) where status is "normal", "warning", or
    "error". A status of "error" means the value must not be saved.
    """
    rules = FIELD_RULES[field_id]

    if raw_value is None or (isinstance(raw_value, str) and raw_value.strip() == ""):
        return "error", "This field cannot be empty."

    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return "error", "Please enter a valid number."

    hard_error = _check_hard_bounds(value, rules)
    if hard_error is not None:
        return "error", hard_error

    cross_field = rules.get("cross_check_field")
    if cross_field:
        cross_error = _check_cross_field(
            value, other_value, rules["cross_check_kind"],
            rules["label"], FIELD_RULES[cross_field]["label"],
        )
        if cross_error is not None:
            return "error", cross_error

    return _check_warn_bounds(value, rules)


def _draft_class(status: str) -> str:
    """
    Return the draft input's className for a validation status.
    """
    if status == "error":
        return "field-input field-input--error"
    if status == "warning":
        return "field-input field-input--warning"
    return "field-input"


def _message_class(status: str) -> str:
    """
    Return the validation message div's className for a validation status.
    """
    if status == "error":
        return "field-message field-message--error"
    if status == "warning":
        return "field-message field-message--warning"
    return "field-message"


def _register_live_validation_callback(field_id: str):
    """
    Register real-time draft validation: border color, message, Save gating.
    """
    cross_field = FIELD_RULES[field_id].get("cross_check_field")

    outputs = [
        Output(f"{field_id}-draft", "className"),
        Output(f"{field_id}-message", "children"),
        Output(f"{field_id}-message", "className"),
        Output(f"{field_id}-save-btn", "disabled"),
    ]
    inputs = [Input(f"{field_id}-draft", "value")]
    states = [State(cross_field, "value")] if cross_field else []

    def _validate(draft_value, *extra_states):
        """
        Re-validate the draft value on every keystroke (no Save needed).
        """
        other_value = extra_states[0] if cross_field else None
        status, message = _validate_field_value(field_id, draft_value, other_value)
        return _draft_class(status), message, _message_class(status), status == "error"

    app.callback(*outputs, *inputs, *states)(_validate)


for _field_id in FIELD_RULES:
    _register_live_validation_callback(_field_id)


def _source_for(value, original_value, source_label):
    """
    Return (display_text, className, date_text) for a field's current value.
    """
    if value == original_value:
        return source_label, "param-source", timestamp_display(EHR_RECORD_DATE, EHR_RECORD_TIME)
    today = datetime.date.today().strftime("%d/%m/%y")
    now = datetime.datetime.now().strftime("%H:%M")
    return "Manual", "param-source param-source--manual", timestamp_display(today, now)


def _restore_class_for(value, original_value):
    """
    Return the Restore link's className - hidden when value matches the original.
    """
    if value == original_value:
        return "popover-restore popover-restore--hidden"
    return "popover-restore"


def _field_result(value, original_value, source_label, popover_open=False):
    """
    Build the 7-tuple of Outputs shared by every branch of the commit callback.
    """
    source, source_class, date = _source_for(value, original_value, source_label)
    popover_class = "edit-popover edit-popover--open" if popover_open else "edit-popover"
    return (
        value, popover_class, value, source, source_class, date,
        _restore_class_for(value, original_value),
    )


def _resolve_field_original(field_id: str, static_original_value, selected_patient_key):
    """
    Resolve the "original"/EHR value a field's Restore button and popover
    footer should compare against.

    For the 6 patient-derived fields (age/height/weight/baseline_sap/
    baseline_dap/baseline_hr) this must track whichever test patient is
    currently selected (selected-test-patient-store), not stay pinned to
    Test Patient 1 forever - static_original_value was that fixed Patient-
    1-at-import-time constant, and is now only a fallback for the field
    ids that never dependent on patient selection in the first place
    (bis-target-low/high, map-target-abs/rel, propofol/remifentanil
    concentration - none of these are keys in PRESET_TEST_PATIENTS, so
    they always fall through to their own fixed model default here,
    unchanged from before).
    """
    preset = PRESET_TEST_PATIENTS.get(selected_patient_key)
    if preset is not None and field_id in preset:
        return preset[field_id]
    return static_original_value


def _register_editable_field_callback(field_id: str, original_value: float, source_label: str):
    """
    Register the Save/Cancel/Restore popover callback for one patient-parameter field.

    Output(field_id, "value") is allow_duplicate=True because the 6 patient-
    parameter fields (age/height/weight/baseline_sap/baseline_dap/
    baseline_hr) are also written by apply_test_patient when a preset test
    patient is selected - harmless for the other fields in EDITABLE_FIELDS
    (targets/concentrations), which only this callback ever writes.

    selected-test-patient-store is read as a State (not just used inside
    apply_test_patient) so that Save/Cancel/Restore/popover-open always
    resolve "original" against whichever patient is selected *right now*
    via _resolve_field_original - see that function's docstring.
    """
    cross_field = FIELD_RULES[field_id].get("cross_check_field")

    outputs = [
        Output(field_id, "value", allow_duplicate=True),
        Output(f"{field_id}-popover", "className"),
        Output(f"{field_id}-draft", "value"),
        Output(f"{field_id}-source", "children"),
        Output(f"{field_id}-source", "className"),
        Output(f"{field_id}-date", "children"),
        Output(f"{field_id}-restore-btn", "className"),
        Output(f"{field_id}-original-label", "children"),
    ]
    inputs = [
        Input(f"{field_id}-badge", "n_clicks"),
        Input(f"{field_id}-save-btn", "n_clicks"),
        Input(f"{field_id}-cancel-btn", "n_clicks"),
        Input(f"{field_id}-restore-btn", "n_clicks"),
        Input(f"{field_id}-draft", "n_submit"),
    ]
    states = [
        State(f"{field_id}-draft", "value"),
        State(field_id, "value"),
        State("selected-test-patient-store", "data"),
    ]
    if cross_field:
        states.append(State(cross_field, "value"))

    def _update_field(badge_clicks, save_clicks, cancel_clicks, restore_clicks, draft_submit,
                       *extra_states):
        """
        Open/close the edit popover and apply Save/Cancel/Restore for this field.
        """
        draft_value, current_value, selected_patient_key = extra_states[0], extra_states[1], extra_states[2]
        other_value = extra_states[3] if cross_field else None
        triggered = ctx.triggered_id

        resolved_original = _resolve_field_original(field_id, original_value, selected_patient_key)
        original_label = f"Original {source_label} value: {resolved_original}"

        if triggered == f"{field_id}-restore-btn":
            return (*_field_result(resolved_original, resolved_original, source_label), original_label)

        if triggered in (f"{field_id}-save-btn", f"{field_id}-draft"):
            status, _ = _validate_field_value(field_id, draft_value, other_value)
            if status == "error":
                # Invalid value: refuse to commit, leave the popover open as-is.
                # The live-validation callback already shows the error inline.
                return (no_update,) * 8
            return (*_field_result(float(draft_value), resolved_original, source_label), original_label)

        if triggered == f"{field_id}-badge":
            return (*_field_result(current_value, resolved_original, source_label, popover_open=True), original_label)

        # Cancel (or any other trigger): close the popover without changing the saved value.
        return (*_field_result(current_value, resolved_original, source_label), original_label)

    app.callback(*outputs, *inputs, *states, prevent_initial_call=True)(_update_field)


for _field_id, _original_value, _source_label in EDITABLE_FIELDS:
    _register_editable_field_callback(_field_id, _original_value, _source_label)


# ============================================================
# Patient case dropdown - preset patient selector
#
# Picking one of the 3 presets from "Select patient" writes straight into
# the same age/height/weight/baseline_sap/baseline_dap/baseline_hr/
# sex-dropdown ids every other input already uses - run_model,
# derived-map/derived-pp, and every editable-field popover all read those
# same ids, so they pick up the new patient automatically. What this
# callback must also do - and previously didn't - is refresh each of
# those 6 fields' Source/Last-Updated/Restore-button/popover-footer
# badges immediately: they used to only reflect whichever patient was
# selected when the page first loaded (Test Patient 1), because nothing
# ever told them a different patient had been selected. See
# _resolve_field_original's docstring for the other half of this fix
# (Save/Cancel/Restore inside a field's own popover).
#
# The dropdown itself defaults to "1" directly in the layout
# (build_patient_id_card), matching build_patient_parameters_card's own
# PRESET_TEST_PATIENTS["1"]-derived field defaults, so Test Patient 1 is
# already fully in place on first load with no callback needing to fire
# (prevent_initial_call=True below is therefore correct, not just a
# performance nicety). The only extra care needed when the dropdown
# *changes* is marking any existing recommendation stale and clearing any
# active manual override (mirroring mark_inputs_stale above) - selecting
# a new preset bypasses that callback's own Save/Restore-button-click
# Inputs entirely, so without this the old recommendation would otherwise
# keep displaying as if it still belonged to the newly-selected patient.
# ============================================================

# (field_id, source_label) for the 6 fields whose value/source/date/
# restore-button/popover-footer all reset to "fresh from this patient's
# EHR" the instant a different preset is selected - built once so both
# the Output list and the return-tuple builder below iterate the same
# fields in the same order.
_PATIENT_PRESET_FIELDS = [
    ("age", "EHR"),
    ("height", "EHR"),
    ("weight", "EHR"),
    ("baseline_sap", "Monitor"),
    ("baseline_dap", "Monitor"),
    ("baseline_hr", "Monitor"),
]

_patient_preset_field_outputs = []
for _pf_id, _pf_source_label in _PATIENT_PRESET_FIELDS:
    _patient_preset_field_outputs.extend([
        Output(_pf_id, "value", allow_duplicate=True),
        Output(f"{_pf_id}-source", "children", allow_duplicate=True),
        Output(f"{_pf_id}-source", "className", allow_duplicate=True),
        Output(f"{_pf_id}-date", "children", allow_duplicate=True),
        Output(f"{_pf_id}-restore-btn", "className", allow_duplicate=True),
        Output(f"{_pf_id}-original-label", "children", allow_duplicate=True),
    ])


@app.callback(
    *_patient_preset_field_outputs,
    Output("sex-dropdown", "value"),
    Output("selected-test-patient-store", "data"),
    Output("recommendation-stale-store", "data", allow_duplicate=True),
    Output("manual-scenario-store", "data", allow_duplicate=True),
    Input("rec-patient-dropdown", "value"),
    State("recommendation-store", "data"),
    State("manual-scenario-store", "data"),
    prevent_initial_call=True,
)
def apply_test_patient(preset_key, rec_store, manual_store):
    """
    Apply the selected preset test patient's fields. Mirrors
    mark_inputs_stale's own guard: only mark the recommendation stale /
    clear the manual override if a recommendation actually exists yet.
    """
    no_change = (no_update,) * (len(_patient_preset_field_outputs) + 4)

    preset = PRESET_TEST_PATIENTS.get(preset_key)
    if preset is None:
        return no_change

    stale = True if rec_store is not None else no_update
    clear_manual = None if manual_store is not None else no_update

    field_results = []
    for field_id, source_label in _PATIENT_PRESET_FIELDS:
        value = preset[field_id]
        field_results.extend([
            value,
            source_label,
            "param-source",
            timestamp_display(EHR_RECORD_DATE, EHR_RECORD_TIME),
            "popover-restore popover-restore--hidden",
            f"Original {source_label} value: {value}",
        ])

    return (
        *field_results,
        preset["sex"],
        preset_key,
        stale,
        clear_manual,
    )


# ============================================================
# Auto-focus a popover's draft input the moment it opens
#
# Every edit popover (every field in EDITABLE_FIELDS, plus the manual-
# override dose popover) shares the same open/closed marker: its wrapper
# div's className gains "edit-popover--open". This one clientside function
# is wired to every such wrapper, so opening any popover always focuses
# and selects its draft input immediately - no per-field JS, no change to
# validation/save/cancel/restore logic, which continues to read the same
# draft value exactly as before.
# ============================================================

def _register_focus_on_open_callback(popover_id: str, draft_id: str):
    """
    Focus and select a popover's draft input as soon as its wrapper gains
    the "edit-popover--open" class, so the user can start typing (or
    immediately overwrite the pre-filled value) without a second click.
    Output goes to an otherwise-unused "data-focus-tick" attribute purely
    because Dash clientside callbacks require an Output - nothing reads it.
    """
    app.clientside_callback(
        f"""
        function(popoverClassName) {{
            if (popoverClassName && popoverClassName.indexOf("edit-popover--open") !== -1) {{
                setTimeout(function () {{
                    var el = document.getElementById("{draft_id}");
                    if (el) {{
                        el.focus();
                        el.select();
                    }}
                }}, 0);
            }}
            return window.dash_clientside.no_update;
        }}
        """,
        Output(f"{popover_id}", "data-focus-tick"),
        Input(f"{popover_id}", "className"),
    )


for _field_id, _, _ in EDITABLE_FIELDS:
    _register_focus_on_open_callback(f"{_field_id}-popover", f"{_field_id}-draft")

_register_focus_on_open_callback("override-popover", "override-dose-draft")

# ============================================================
# Escape-to-cancel for the manual dose popover
#
# Pressing Escape in the draft input simply clicks the existing Cancel
# button, so it goes through the exact same handle_override branch a real
# click would - no duplicate cancel logic here. Scoped to just this one
# popover (not the generic per-field editors) since that's the only editor
# this request covers.
# ============================================================

app.clientside_callback(
    """
    function(popoverClassName) {
        if (popoverClassName && popoverClassName.indexOf("edit-popover--open") !== -1) {
            var el = document.getElementById("override-dose-draft");
            if (el && !el.dataset.escBound) {
                el.dataset.escBound = "1";
                el.addEventListener("keydown", function (e) {
                    if (e.key === "Escape") {
                        var cancelBtn = document.getElementById("override-cancel-btn");
                        if (cancelBtn) {
                            cancelBtn.click();
                        }
                    }
                });
            }
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("override-dose-draft", "data-esc-tick"),
    Input("override-popover", "className"),
)


# ============================================================
# Stale-recommendation tracking
#
# Once a recommendation exists, any *committed* input change (Save/Restore
# on a patient/model-setting field, a sex toggle, or a new opiate selection
# - not just opening/canceling a popover) marks it stale and clears any
# active manual override, so update_result_visibility (below) hides the now
# out-of-date cards/graphs behind a "re-run" placeholder instead of letting
# them keep displaying results computed from the previous inputs.
# ============================================================

_STALE_TRACKING_INPUTS = []
for _field_id, _, _ in EDITABLE_FIELDS:
    _STALE_TRACKING_INPUTS.append(Input(f"{_field_id}-save-btn", "n_clicks"))
    _STALE_TRACKING_INPUTS.append(Input(f"{_field_id}-restore-btn", "n_clicks"))
_STALE_TRACKING_INPUTS.append(Input("sex-dropdown", "value"))
_STALE_TRACKING_INPUTS.append(Input("opiate-dropdown", "value"))


@app.callback(
    Output("recommendation-stale-store", "data", allow_duplicate=True),
    Output("manual-scenario-store", "data", allow_duplicate=True),
    *_STALE_TRACKING_INPUTS,
    State("recommendation-store", "data"),
    State("manual-scenario-store", "data"),
    prevent_initial_call=True,
)
def mark_inputs_stale(*args):
    """
    Mark the current recommendation stale and clear any active manual
    override, but only if a recommendation actually exists yet - editing
    inputs before the first "Run recommendation" has nothing to mark stale.

    manual-scenario-store is only written (to None) when an override is
    actually active. Writing it unconditionally on every input change would
    re-trigger render_recommendation's (redundant, several-seconds-long)
    figure recompute every time - and since render_recommendation's outputs
    live inside the dcc.Loading wrapper, that would blank the whole
    recommendation/predictions area behind a loading spinner on every
    keystroke-driven Save, not just when there was actually something to
    clear.
    """
    rec_store, manual_store = args[-2], args[-1]
    if rec_store is None:
        return no_update, no_update
    clear_manual = None if manual_store is not None else no_update
    return True, clear_manual


# ============================================================
# Main recommendation callback
#
# run_model computes the recommendation and stores it (recommendation-store)
# - it no longer renders anything directly. A new "Run recommendation" click
# also clears any active manual override (manual-scenario-store), since a
# stale override computed against the previous inputs/recommendation would
# no longer be meaningful. render_recommendation (below) is the single
# owner of the actual display outputs, driven by both stores, so the same
# rendering logic handles "just ran a new recommendation" and "just
# changed override state" identically.
# ============================================================

def _patient_from_context(context: dict) -> Patient:
    """
    Reconstruct the Patient used for a stored recommendation context.
    """
    return Patient(
        age=float(context["age"]),
        height=float(context["height"]),
        weight=float(context["weight"]),
        sex=context["sex"] or "male",
        opiates=(context["opiate"] == "remifentanil"),
        blood_sampling_site="arterial",
        base_sap=float(context["base_sap"]),
        base_dap=float(context["base_dap"]),
        base_hr=float(context["base_hr"]),
    )


@app.callback(
    Output("recommendation-store", "data"),
    Output("manual-scenario-store", "data"),
    Output("recommendation-stale-store", "data", allow_duplicate=True),
    Output("recommendation-loading-anchor", "children"),
    Input("run-btn", "n_clicks"),
    State("age", "value"),
    State("height", "value"),
    State("weight", "value"),
    State("baseline_sap", "value"),
    State("baseline_dap", "value"),
    State("baseline_hr", "value"),
    State("sex-store", "data"),
    State("opiate-dropdown", "value"),
    State("propofol-concentration", "value"),
    State("remifentanil-concentration", "value"),
    State("bis-target-low", "value"),
    State("bis-target-high", "value"),
    State("map-target-abs", "value"),
    State("map-target-rel", "value"),
    prevent_initial_call=True,
)
def run_model(
    n_clicks,
    age,
    height,
    weight,
    baseline_sap,
    baseline_dap,
    baseline_hr,
    sex,
    opiate,
    propofol_concentration_value,
    remifentanil_concentration_value,
    target_bis_low,
    target_bis_high,
    map_target_abs,
    map_target_rel_pct,
):
    """
    Run the pharmacokinetic model and store the recommendation. Rendering is
    handled separately by render_recommendation.

    The 4th output, recommendation-loading-anchor, carries no data of its
    own (its value is never read anywhere) - it exists purely so this
    callback has an Output inside the dcc.Loading subtree (see
    build_layout in dashboard_layout.py), which keeps the existing purple
    loading overlay active for this callback's own (potentially several
    seconds long) run, not just for render_recommendation's afterward.
    Without it, the overlay only appeared once render_recommendation
    started, since recommendation-store/manual-scenario-store/
    recommendation-stale-store all live outside that subtree.
    """
    try:
        age = float(age)
        height = float(height)
        weight = float(weight)
        baseline_sap = float(baseline_sap)
        baseline_dap = float(baseline_dap)
        baseline_hr = float(baseline_hr)

        if age <= 0:
            raise ValueError("Age must be positive.")
        if height <= 0:
            raise ValueError("Height must be positive.")
        if weight <= 0:
            raise ValueError("Weight must be positive.")
        if baseline_sap <= 0 or baseline_dap <= 0:
            raise ValueError("Baseline SBP and DBP must be positive.")
        if baseline_sap <= baseline_dap:
            raise ValueError("Baseline SBP must be higher than DBP.")
        if baseline_hr <= 0:
            raise ValueError("Baseline HR must be positive.")

        target_bis_low = float(target_bis_low)
        target_bis_high = float(target_bis_high)
        map_target_abs = float(map_target_abs)
        map_target_rel_pct = float(map_target_rel_pct)

        if target_bis_low <= 0 or target_bis_high <= 0:
            raise ValueError("BIS targets must be positive.")
        if target_bis_low >= target_bis_high:
            raise ValueError("BIS target (lower) must be lower than BIS target (upper).")
        if map_target_abs <= 0:
            raise ValueError("MAP target (absolute) must be positive.")
        if map_target_rel_pct <= 0:
            raise ValueError("MAP target (relative) must be positive.")

        map_target_rel_frac = map_target_rel_pct / 100.0

        opiate = opiate or "none"
        if opiate in {"sufentanil", "fentanyl"}:
            raise ValueError("Only remifentanil is currently supported.")

        propofol_concentration = validate_concentration(
            propofol_concentration_value, name="Propofol",
        )
        remifentanil_concentration = validate_concentration(
            remifentanil_concentration_value, name="Remifentanil",
        )

        patient = Patient(
            age=age,
            height=height,
            weight=weight,
            sex=sex or "male",
            opiates=(opiate == "remifentanil"),
            blood_sampling_site="arterial",
            base_sap=baseline_sap,
            base_dap=baseline_dap,
            base_hr=baseline_hr,
        )

        rec = recommend_su2023_regimen(
            patient=patient,
            opiate=opiate,
            propofol_conc_mg_ml=propofol_concentration,
            remifentanil_conc_mcg_ml=remifentanil_concentration,
            target_bis_low=target_bis_low,
            target_bis_high=target_bis_high,
            map_abs_min_target=map_target_abs,
            map_rel_frac_target=map_target_rel_frac,
        )

        # opiate has already been normalized to "none"/"remifentanil" inside
        # recommend_su2023_regimen's own logic via use_remifentanil, but that
        # normalization is internal to it - redo it here so the stored
        # context always holds the normalized value the manual-override path
        # also expects.
        normalized_opiate = "remifentanil" if opiate == "remifentanil" else "none"

        store_data = {
            "error": None,
            "context": {
                "age": age,
                "height": height,
                "weight": weight,
                "sex": sex or "male",
                "base_sap": baseline_sap,
                "base_dap": baseline_dap,
                "base_hr": baseline_hr,
                "opiate": normalized_opiate,
                "propofol_conc_mg_ml": propofol_concentration,
                "remifentanil_conc_mcg_ml": remifentanil_concentration,
                "target_bis_low": target_bis_low,
                "target_bis_high": target_bis_high,
                "map_abs_min_target": map_target_abs,
                "map_rel_frac_target": map_target_rel_frac,
            },
            "result": _rec_to_store_dict(rec),
        }
        return store_data, None, False, n_clicks

    except Exception as e:
        return {"error": str(e), "context": None, "result": None}, None, False, n_clicks


@app.callback(
    Output("summary-output", "children"),
    Output("dose-rationale-graph", "figure"),
    Output("propofol-pk-graph", "figure"),
    Output("remifentanil-pk-graph", "figure"),
    Output("bis-graph", "figure"),
    Output("map-graph", "figure"),
    Input("recommendation-store", "data"),
    Input("manual-scenario-store", "data"),
    State("selected-test-patient-store", "data"),
    prevent_initial_call=True,
)
def render_recommendation(rec_store, manual_store, test_patient_key):
    """
    Render the recommendation card(s) and the 5 prediction graphs from
    recommendation-store (the original recommendation, never overwritten)
    and manual-scenario-store (the active manual override, if any). This is
    the single owner of these outputs - it fires both when a new
    recommendation is run and when the manual-override state changes, so
    there is exactly one rendering code path for both cases.

    test_patient_key (selected-test-patient-store, State not Input - this
    callback should only re-fire on an actual new recommendation/manual-
    override, not merely from switching the patient dropdown) is read here
    and passed to make_summary purely for the confidence-badge override -
    see make_induction_card's docstring. Whichever preset was selected at
    the moment this recommendation/manual override was produced is what
    render_recommendation reads, so editing inputs afterward (without
    switching patient or re-running) can never change which override
    applies, and switching patients then re-running immediately reflects
    the newly-selected patient's own override.
    """
    if rec_store is None:
        return no_update, no_update, no_update, no_update, no_update, no_update

    if rec_store.get("error"):
        error_component = html.Div(
            html.Pre(
                f"Error while running recommendation:\n{rec_store['error']}",
                style={
                    "whiteSpace": "pre-wrap",
                    "fontFamily": "monospace",
                    "color": "#b00020",
                    "margin": 0,
                },
            ),
            className="card",
        )
        empty = make_empty_figure()
        return error_component, empty, empty, empty, empty, empty

    context = rec_store["context"]
    original = _store_dict_to_namespace(rec_store["result"])
    manual = _store_dict_to_namespace(manual_store["result"]) if manual_store else None
    manual_dose_mgkg = float(manual.propofol_bolus_mgkg) if manual is not None else None

    patient = _patient_from_context(context)

    summary = make_summary(original, manual, test_patient_key)

    dose_rationale_fig = make_induction_dose_rationale_figure(
        patient=patient,
        rec=original,
        opiate=context["opiate"],
        propofol_conc_mg_ml=context["propofol_conc_mg_ml"],
        remifentanil_conc_mcg_ml=context["remifentanil_conc_mcg_ml"],
        manual_dose_mgkg=manual_dose_mgkg,
    )

    if manual is None:
        propofol_pk_fig = make_propofol_pk_figure(original)
        remifentanil_pk_fig = make_remifentanil_pk_figure(original)
        bis_fig = make_bis_figure(original)
        map_fig = make_map_figure(original)
    else:
        propofol_pk_fig = make_propofol_pk_figure_dual(original, manual)
        remifentanil_pk_fig = make_remifentanil_pk_figure_dual(original, manual)
        bis_fig = make_bis_figure_dual(original, manual)
        map_fig = make_map_figure_dual(original, manual)

    return summary, dose_rationale_fig, propofol_pk_fig, remifentanil_pk_fig, bis_fig, map_fig


# ============================================================
# Run-button loading feedback
#
# Two clientside callbacks (no server round-trip, so they resolve well
# before run_model's actual computation finishes) give instant feedback on
# click and reset it once the run completes. Disabling a real DOM button
# also means the browser itself refuses further clicks while it is
# disabled - a genuine guarantee against duplicate submissions, not just a
# timing-based one. Neither touches run_model or any model/recommendation
# state - recommendation-loading-store is purely a UI signal consumed by
# update_result_visibility below.
# ============================================================

app.clientside_callback(
    """
    function(n_clicks) {
        return [
            "run-recommendation-btn run-recommendation-btn--loading",
            true,
            "Generating recommendation...",
            true
        ];
    }
    """,
    Output("run-btn", "className"),
    Output("run-btn", "disabled"),
    Output("run-btn-label", "children"),
    Output("recommendation-loading-store", "data"),
    Input("run-btn", "n_clicks"),
    prevent_initial_call=True,
)

app.clientside_callback(
    """
    function(rec_store) {
        return [
            "run-recommendation-btn",
            false,
            "Run recommendation",
            false
        ];
    }
    """,
    Output("run-btn", "className", allow_duplicate=True),
    Output("run-btn", "disabled", allow_duplicate=True),
    Output("run-btn-label", "children", allow_duplicate=True),
    Output("recommendation-loading-store", "data", allow_duplicate=True),
    Input("recommendation-store", "data"),
    prevent_initial_call=True,
)


# ============================================================
# Loading-overlay caption text
#
# "Generating personalized recommendation" is shown/hidden by its own pair
# of clientside callbacks, independent of recommendation-loading-store
# (which - see above - intentionally resets as soon as run_model finishes,
# to restore the button; that is still correct button behavior and is left
# untouched). The caption instead needs to stay visible for the *entire*
# click -> render_recommendation-finished window, matching the purple
# dcc.Loading overlay's own active span (recommendation-loading-anchor
# keeps that overlay active through run_model; render_recommendation's own
# outputs, already inside the same dcc.Loading subtree, keep it active
# through rendering). map-graph is one of render_recommendation's six
# Outputs, all six of which are always written together in a single
# response (including the error path), so it reliably fires exactly once
# rendering has finished, success or error alike.
# ============================================================

app.clientside_callback(
    """
    function(n_clicks) {
        return {display: "flex"};
    }
    """,
    Output("recommendation-loading-text", "style"),
    Input("run-btn", "n_clicks"),
    prevent_initial_call=True,
)

app.clientside_callback(
    """
    function(figure) {
        return {display: "none"};
    }
    """,
    Output("recommendation-loading-text", "style", allow_duplicate=True),
    Input("map-graph", "figure"),
    prevent_initial_call=True,
)


# ============================================================
# Before/after-run state
#
# Purely presentational: toggles which of {empty placeholder, stale
# placeholder, real content} is visible for the recommendation, rationale,
# and predictions sections. It never touches figure/card content itself
# (render_recommendation, above, still owns that), so hidden content stays
# in the DOM - already-computed but out-of-date graphs are just not shown,
# rather than being cleared and recomputed.
#
# The "generating..." loading placeholder is deliberately NOT one of this
# callback's outputs - see toggle_loading_placeholders below for why.
# ============================================================

@app.callback(
    Output("recommendation-empty-placeholder", "style"),
    Output("recommendation-stale-placeholder", "style"),
    Output("summary-output", "style"),
    Output("rationale-empty-placeholder", "style"),
    Output("rationale-stale-placeholder", "style"),
    Output("dose-rationale-card-wrapper", "style"),
    Output("predictions-empty-placeholder", "style"),
    Output("predictions-stale-placeholder", "style"),
    Output("predictions-grid", "style"),
    Input("recommendation-store", "data"),
    Input("recommendation-stale-store", "data"),
)
def update_result_visibility(rec_store, stale):
    """
    Derive the before-run / after-run / stale display state from
    recommendation-store (has a recommendation ever been run?) and
    recommendation-stale-store (have inputs changed since?), and show
    exactly one of the three states for each of the three sections.
    """
    has_result = rec_store is not None
    is_stale = has_result and bool(stale)
    show_fresh = has_result and not is_stale
    show_empty = not has_result

    hidden = {"display": "none"}
    shown = None

    empty_style = shown if show_empty else hidden
    stale_style = shown if is_stale else hidden
    fresh_style = shown if show_fresh else hidden

    return (
        empty_style, stale_style, fresh_style,
        empty_style, stale_style, fresh_style,
        empty_style, stale_style, fresh_style,
    )


# ============================================================
# Loading-placeholder visibility (independent of update_result_visibility)
#
# This is deliberately its OWN clientside callback with recommendation-
# loading-store as its ONLY input, rather than a third input folded into
# update_result_visibility above. Empirically, when a single callback
# depends on both recommendation-loading-store and recommendation-store,
# Dash's front-end appears to batch/defer firing it until BOTH inputs
# have settled (since they share the same ultimate trigger - the
# "Run recommendation" click) - the loading-only render never actually
# reached the page, it jumped straight from empty to fresh. Keeping this
# callback's only input isolated to the loading store (which changes via
# a clientside callback with no shared upstream dependency) sidesteps that
# entirely and fires immediately, exactly like the run-btn button state
# does. When loading ends, this callback only hides its own placeholders
# and leaves every other Output as `no_update`, so it can never clobber
# whatever update_result_visibility has already (correctly) set.
#
# The three *-loading-placeholder cards themselves are kept permanently
# hidden below (never set back to `null`/visible) - the single continuous
# dcc.Loading overlay (see recommendation-loading-anchor above and
# recommendation-loading-text below) is now the only loading indicator
# shown; the empty/stale/content sections are still hidden while loading
# is in progress exactly as before, only the redundant card-level
# "Generating..." messages were dropped. The placeholder components
# themselves are untouched in dashboard_layout.py so this remains a
# one-line-per-branch, fully reversible change.
# ============================================================

app.clientside_callback(
    """
    function(loading) {
        var hidden = {display: "none"};
        var noUpdate = window.dash_clientside.no_update;
        if (loading) {
            return [
                hidden, hidden, hidden, hidden,
                hidden, hidden, hidden, hidden,
                hidden, hidden, hidden, hidden
            ];
        }
        return [
            hidden, noUpdate, noUpdate, noUpdate,
            hidden, noUpdate, noUpdate, noUpdate,
            hidden, noUpdate, noUpdate, noUpdate
        ];
    }
    """,
    Output("recommendation-loading-placeholder", "style"),
    Output("recommendation-empty-placeholder", "style", allow_duplicate=True),
    Output("recommendation-stale-placeholder", "style", allow_duplicate=True),
    Output("summary-output", "style", allow_duplicate=True),
    Output("rationale-loading-placeholder", "style"),
    Output("rationale-empty-placeholder", "style", allow_duplicate=True),
    Output("rationale-stale-placeholder", "style", allow_duplicate=True),
    Output("dose-rationale-card-wrapper", "style", allow_duplicate=True),
    Output("predictions-loading-placeholder", "style"),
    Output("predictions-empty-placeholder", "style", allow_duplicate=True),
    Output("predictions-stale-placeholder", "style", allow_duplicate=True),
    Output("predictions-grid", "style", allow_duplicate=True),
    Input("recommendation-loading-store", "data"),
    prevent_initial_call=True,
)


# ============================================================
# Manual-override callbacks
# ============================================================

@app.callback(
    Output("override-dose-draft", "className"),
    Output("override-dose-message", "children"),
    Output("override-dose-message", "className"),
    Input("override-save-btn", "n_clicks"),
    Input("override-dose-draft", "n_submit"),
    State("override-dose-draft", "value"),
    State("override-unit", "value"),
    State("recommendation-store", "data"),
    prevent_initial_call=True,
)
def validate_override_draft(save_clicks, submit_count, draft_value, unit, rec_store):
    """
    Validate the manual-dose draft only when the user commits it (clicking
    Save or pressing Enter) - not on every keystroke, so partial typing
    (e.g. "1", "10", "100") never flashes a spurious error while the user
    is still typing. The Save button is never programmatically disabled by
    this (there is nothing to keep it in sync with anymore), so it always
    stays clickable and handle_override re-validates independently on its
    own trigger.
    """
    if not rec_store or rec_store.get("error") or not rec_store.get("context"):
        return "field-input", "", "field-message"

    weight_kg = float(rec_store["context"]["weight"])
    status, message, _ = _validate_override_dose(draft_value, weight_kg, unit or "total")
    return _draft_class(status), message, _message_class(status)


@app.callback(
    Output("override-popover", "className"),
    Output("override-dose-draft", "value"),
    Output("override-dose-draft", "className", allow_duplicate=True),
    Output("override-dose-message", "children", allow_duplicate=True),
    Output("override-dose-message", "className", allow_duplicate=True),
    Output("override-unit", "value"),
    Output("manual-scenario-store", "data", allow_duplicate=True),
    Output("override-draft-unit", "data"),
    Input("override-dose-btn", "n_clicks"),
    Input("override-save-btn", "n_clicks"),
    Input("override-cancel-btn", "n_clicks"),
    Input("override-return-btn", "n_clicks"),
    Input("override-dose-draft", "n_submit"),
    State("override-dose-draft", "value"),
    State("override-unit", "value"),
    State("recommendation-store", "data"),
    State("manual-scenario-store", "data"),
    prevent_initial_call=True,
)
def handle_override(
    open_clicks, save_clicks, cancel_clicks, return_clicks, submit_count,
    draft_value, unit, rec_store, manual_store,
):
    """
    Open/close the "Override propofol dose" popover and apply Save / Cancel
    / Return-to-recommendation. Pressing Enter in the draft field
    (n_submit) commits exactly like clicking Save. Save triggers a real
    (fixed-bolus) re-optimization - this is the one action in this
    callback that is not instant, and it shows the existing dcc.Loading
    spinner automatically since these components live inside
    summary-output.

    Opening the popover always resets the unit toggle to "Total dose" (and
    the override-draft-unit store that convert_override_draft_unit uses to
    track which unit the draft text is currently expressed in) and clears
    any leftover validation message/border from a previous edit, so each
    edit starts clean regardless of how the last one ended.

    All five Inputs live inside summary-output, which starts out absent
    and is created in one shot the first time a recommendation renders -
    so the very first time that happens, every one of these ids goes from
    "did not exist" to "exists with its initial value" simultaneously.
    Dash's front-end still resolves ctx.triggered_id to exactly one of
    them for that event (which one is an implementation detail, not
    something to rely on), so every branch below also checks the actual
    n_clicks/n_submit count truthiness - a genuine click/submit is always
    a non-zero/non-None value, while this simultaneous first-mount event
    always carries each input's untouched falsy initial value (0 or None).
    Without this, that first-mount event could be misread as a real click
    on whichever id happens to "win" and pop the edit popover open
    immediately after every "Run recommendation".
    """
    triggered = ctx.triggered_id
    clean = ("field-input", "", "field-message")

    if triggered == "override-return-btn" and return_clicks:
        return "edit-popover", no_update, no_update, no_update, no_update, no_update, None, no_update

    if not rec_store or rec_store.get("error") or not rec_store.get("context"):
        return (no_update,) * 8

    context = rec_store["context"]

    if triggered == "override-dose-btn" and open_clicks:
        prefill = (
            manual_store["dose_mg"]
            if manual_store
            else rec_store["result"]["propofol_bolus_mg"]
        )
        return ("edit-popover edit-popover--open", prefill, *clean, "total", no_update, "total")

    if triggered == "override-cancel-btn" and cancel_clicks:
        return ("edit-popover", no_update, *clean, no_update, no_update, no_update)

    if triggered in ("override-save-btn", "override-dose-draft") and (save_clicks or submit_count):
        weight_kg = float(context["weight"])
        status, _, manual_dose_mg = _validate_override_dose(draft_value, weight_kg, unit or "total")
        if status == "error":
            # Invalid value: refuse to commit, leave the popover open.
            # validate_override_draft (same trigger) shows the error inline.
            return (no_update,) * 8

        try:
            patient = _patient_from_context(context)
            manual_rec = recommend_maintenance_for_fixed_propofol_bolus(
                patient=patient,
                fixed_bolus_mg=manual_dose_mg,
                opiate=context["opiate"],
                propofol_conc_mg_ml=context["propofol_conc_mg_ml"],
                remifentanil_conc_mcg_ml=context["remifentanil_conc_mcg_ml"],
                target_bis_low=context["target_bis_low"],
                target_bis_high=context["target_bis_high"],
                map_abs_min_target=context["map_abs_min_target"],
                map_rel_frac_target=context["map_rel_frac_target"],
            )
        except Exception:
            # Keep the popover open; live-validation already covers the
            # common invalid-input cases, so a failure here is unexpected.
            return (no_update,) * 8

        manual_store_data = {
            "dose_mg": manual_dose_mg,
            "result": _rec_to_store_dict(manual_rec),
        }
        return ("edit-popover", no_update, no_update, no_update, no_update, no_update, manual_store_data, no_update)

    # Any other trigger: close the popover without changing anything.
    return ("edit-popover", no_update, no_update, no_update, no_update, no_update, no_update, no_update)


@app.callback(
    Output("override-dose-draft", "value", allow_duplicate=True),
    Output("override-draft-unit", "data", allow_duplicate=True),
    Input("override-unit", "value"),
    State("override-dose-draft", "value"),
    State("override-draft-unit", "data"),
    State("recommendation-store", "data"),
    prevent_initial_call=True,
)
def convert_override_draft_unit(new_unit, draft_value, prev_unit, rec_store):
    """
    Convert the manual-dose draft text between total mg and mg/kg whenever
    the user actually toggles the unit control - e.g. 120 mg for a 72 kg
    patient becomes 1.67 when switching to mg/kg, and back to 120 mg when
    switching back to Total dose. override-draft-unit tracks which unit the
    currently displayed draft text is expressed in; handle_override resets
    it to "total" alongside override-unit every time it opens/saves/cancels
    the popover (writing the draft text in the correct unit itself), so
    this only converts on a genuine user toggle, never on those
    programmatic resets. This purely reformats the displayed text - the
    value actually saved on Save is still parsed fresh against whichever
    unit is selected at that time (_validate_override_dose), so validation
    and the saved dose are unaffected.
    """
    new_unit = new_unit or "total"
    prev_unit = prev_unit or "total"
    if new_unit == prev_unit:
        return no_update, no_update

    if not rec_store or rec_store.get("error") or not rec_store.get("context"):
        return no_update, new_unit

    try:
        typed_value = float(draft_value)
    except (TypeError, ValueError):
        return no_update, new_unit

    weight_kg = float(rec_store["context"]["weight"])
    value_mg = typed_value * weight_kg if prev_unit == "mgkg" else typed_value
    return _dose_value_for_display(value_mg, weight_kg, new_unit), new_unit


@app.callback(
    Output("override-dose-draft", "placeholder"),
    Input("override-unit", "value"),
)
def update_override_placeholder(unit):
    """
    Keep the draft input's placeholder text matching the selected unit -
    presentation only, does not affect validation or the committed value.
    """
    return "Manual dose in mg/kg" if unit == "mgkg" else "Manual dose in mg"


# ============================================================
# Scenario Exploration page
#
# A fully independent "what-if" pipeline - its own patient (se-age/se-sex/
# se-height/se-weight/se-sbp/se-dbp), its own targets, its own opioid/dose
# scenario, never recommendation-store or manual-scenario-store. It reuses
# the exact same figure-building functions the Recommendation page uses
# (make_bis_figure, make_map_figure, make_propofol_pk_figure,
# make_remifentanil_pk_figure, make_induction_dose_rationale_figure) so its
# graphs are byte-identical in style - only the data fed into them differs.
#
# Two-tier computation, mirroring the app's existing manual-override
# pattern (recommendation-store vs. manual-scenario-store):
#   - compute_scenario_baseline (below) runs the full Powell optimization
#     (recommend_su2023_regimen) - expensive, so it only fires when the
#     patient, targets, or opioid choice change, and every text input is
#     debounce=True so it never fires mid-keystroke. This also computes the
#     90% prediction interval band once.
#   - render_scenario_page (below) does one cheap, single-shot
#     simulate_regimen() call per dose/rate slider change - reusing the
#     baseline's own propofol maintenance schedule and its confidence band
#     unchanged (exactly like the Recommendation page overlays a manual
#     dose on the original recommendation's band), so sliders can update
#     live while dragging without re-running the optimizer.
# ============================================================

def _scenario_patient_from_fields(age, height, weight, sex, sbp, dbp) -> Patient:
    """
    Build a Patient for the Scenario Exploration page. base_hr is fixed at
    70 bpm (matching the Recommendation page's own default) rather than a
    user-editable field, since Scenario Exploration deliberately has no HR
    input (see build_scenario_patient_card) - EleveldPatient still needs
    *some* baseline HR to derive base_tpr for the MAP simulation, so this
    keeps that derivation working without exposing a field the spec didn't
    ask for.
    """
    return Patient(
        age=float(age),
        height=float(height),
        weight=float(weight),
        sex=sex or "male",
        opiates=False,
        blood_sampling_site="arterial",
        base_sap=float(sbp),
        base_dap=float(dbp),
        base_hr=70.0,
    )


@app.callback(
    Output("se-derived-map", "children"),
    Output("se-derived-pp", "children"),
    Input("se-sbp", "value"),
    Input("se-dbp", "value"),
)
def update_scenario_derived_pressures(sbp, dbp):
    """
    Scenario Exploration's own MAP/PP derivation - same formula as
    update_derived_pressures, kept as a separate callback because it reads
    se-sbp/se-dbp, never the Recommendation page's baseline_sap/baseline_dap.
    """
    if sbp is None or dbp is None:
        return "-", "-"

    try:
        sbp = float(sbp)
        dbp = float(dbp)

        if sbp <= 0 or dbp <= 0 or sbp <= dbp:
            return "-", "-"

        return f"{compute_map(sbp, dbp):.1f}", f"{compute_pp(sbp, dbp):.1f}"

    except Exception:
        return "-", "-"


@app.callback(
    Output("se-age", "value"),
    Output("se-sex", "value"),
    Output("se-height", "value"),
    Output("se-weight", "value"),
    Output("se-sbp", "value"),
    Output("se-dbp", "value"),
    Input("se-restore-patient-btn", "n_clicks"),
    prevent_initial_call=True,
)
def restore_scenario_patient(_n_clicks):
    """
    Reset every Patient Scenario field back to SCENARIO_DEFAULT_PATIENT.
    """
    d = SCENARIO_DEFAULT_PATIENT
    return d["age"], d["sex"], d["height"], d["weight"], d["sbp"], d["dbp"]


@app.callback(
    Output("se-remi-rate-block", "style"),
    Input("se-opioid-dropdown", "value"),
)
def toggle_scenario_remi_rate_visibility(opioid):
    """
    Hide the remifentanil infusion-rate slider whenever no opioid (or a
    not-yet-supported one) is selected - there is nothing for it to control.
    """
    return None if opioid == "remifentanil" else {"display": "none"}


def _combine_target_status(status_a: str, message_a: str, status_b: str, message_b: str):
    """
    Combine two independent field validations into one shared message: pick
    whichever is more severe (error > warning > normal), and show whichever
    side actually has a message.
    """
    severity = {"error": 2, "warning": 1, "normal": 0}
    if severity[status_a] >= severity[status_b]:
        return status_a, message_a or message_b
    return status_b, message_b or message_a


def _scenario_compact_input_class(status: str) -> str:
    """
    Like _draft_class(), but for Scenario Exploration's compact inputs:
    _draft_class() hardcodes a "field-input" base class (right for the
    Recommendation page's own fields, which start with that class) - this
    keeps this page's own "scenario-compact-input scenario-range-input"
    base class and only adds the shared field-input--error/--warning color
    modifiers on top, so a validation callback firing doesn't blow away
    this page's compact sizing.
    """
    base = "scenario-compact-input scenario-range-input"
    if status == "error":
        return f"{base} field-input--error"
    if status == "warning":
        return f"{base} field-input--warning"
    return base


@app.callback(
    Output("se-bis-target-low", "className"),
    Output("se-bis-target-high", "className"),
    Output("se-bis-target-message", "children"),
    Output("se-bis-target-message", "className"),
    Input("se-bis-target-low", "value"),
    Input("se-bis-target-high", "value"),
)
def validate_scenario_bis_targets(low, high):
    """
    Validate the compact BIS lower/upper row, reusing FIELD_RULES/
    _validate_field_value unchanged (same clinical bounds and messages as
    the Recommendation page's bis-target-low/bis-target-high fields) - just
    against this page's own se-* inputs, with one shared message below the
    row instead of one message per field.
    """
    status_low, message_low = _validate_field_value("bis-target-low", low, high)
    status_high, message_high = _validate_field_value("bis-target-high", high, low)
    combined_status, combined_message = _combine_target_status(
        status_low, message_low, status_high, message_high,
    )
    return (
        _scenario_compact_input_class(status_low),
        _scenario_compact_input_class(status_high),
        combined_message,
        _message_class(combined_status),
    )


@app.callback(
    Output("se-map-target-value", "className"),
    Output("se-map-target-message", "children"),
    Output("se-map-target-message", "className"),
    Input("se-map-target-value", "value"),
    State("se-map-target-mode", "value"),
)
def validate_scenario_map_target(value, mode):
    """
    Validate the compact MAP target value against whichever rule set
    (map-target-abs or map-target-rel) matches the current mode dropdown -
    same FIELD_RULES bounds/messages as the Recommendation page.
    """
    rule_key = "map-target-rel" if mode == "rel" else "map-target-abs"
    status, message = _validate_field_value(rule_key, value)
    return _scenario_compact_input_class(status), message, _message_class(status)


@app.callback(
    Output("se-map-target-value", "value"),
    Input("se-map-target-mode", "value"),
    prevent_initial_call=True,
)
def reset_scenario_map_target_value(mode):
    """
    Switching Target MAP mode shows that mode's own default value, rather
    than leaving (say) "65" on screen after switching to "Relative (%
    baseline)", where 65 would be a nonsensical percentage.
    """
    return (MAP_REL_FRAC_TARGET * 100.0) if mode == "rel" else MAP_ABS_MIN_TARGET


def _register_scenario_slider_sync(slider_id: str, input_id: str, bounds: tuple[float, float]):
    """
    Keep a Scenario Exploration dose/rate slider and its compact numeric
    input in sync in both directions, as two independent one-directional
    callbacks rather than one callback reading and writing the same
    "value" prop on both sides - a single combined callback here silently
    failed to cascade to render_scenario_page when triggered by the input
    (the same front-end batching quirk this app hit once before with a
    shared-Input callback; two plain one-directional callbacks sidestep it
    entirely, and match the already-working pattern reset_scenario_medication
    uses to drive the slider directly).
    """
    @app.callback(
        Output(input_id, "value"),
        Input(slider_id, "value"),
    )
    def _slider_to_input(slider_value):
        return f"{float(slider_value):.2f}"

    @app.callback(
        Output(slider_id, "value", allow_duplicate=True),
        Input(input_id, "value"),
        State(slider_id, "value"),
        prevent_initial_call=True,
    )
    def _input_to_slider(input_value, current_slider_value):
        try:
            value = float(input_value)
        except (TypeError, ValueError):
            return no_update
        value = max(bounds[0], min(bounds[1], value))
        # Skip the write-back if it would be a no-op (within floating-point
        # tolerance): _slider_to_input reformats every slider move through
        # "%.2f" text, and re-parsing that text here can differ from the
        # slider's own raw float by a trailing bit or two - writing that
        # "different" float back would otherwise re-trigger _slider_to_input
        # in a pointless echo, adding request churn that can arrive out of
        # order relative to render_scenario_page under rapid changes.
        if current_slider_value is not None and abs(value - float(current_slider_value)) < 1e-9:
            return no_update
        return value

    _slider_to_input.__name__ = f"sync_{slider_id.replace('-', '_')}_to_input"
    _input_to_slider.__name__ = f"sync_{input_id.replace('-', '_')}_to_slider"


_register_scenario_slider_sync("se-remi-rate-slider", "se-remi-rate-input", REMI_INFUSION_MCGKGMIN_BOUNDS)


@app.callback(
    Output("se-baseline-store", "data"),
    Input("nav-scenario-btn", "n_clicks"),
    Input("se-age", "value"),
    Input("se-sex", "value"),
    Input("se-height", "value"),
    Input("se-weight", "value"),
    Input("se-sbp", "value"),
    Input("se-dbp", "value"),
    Input("se-opioid-dropdown", "value"),
    Input("se-remi-rate-slider", "value"),
    Input("se-bis-target-low", "value"),
    Input("se-bis-target-high", "value"),
    Input("se-map-target-mode", "value"),
    Input("se-map-target-value", "value"),
    prevent_initial_call=True,
)
def compute_scenario_baseline(
    _nav_clicks, age, sex, height, weight, sbp, dbp, opioid, remi_rate_mcgkgmin,
    bis_low, bis_high, map_mode, map_value,
):
    """
    Run the full Powell-optimized recommendation for the Scenario
    Exploration page's own (independent) patient/targets/opioid choice.

    Opioid strategy is the input here, propofol is the output: when
    remifentanil is selected, its rate is PINNED at the slider's current
    value (recommend_propofol_for_fixed_remifentanil_rate) and only the
    propofol bolus + propofol maintenance are re-optimized in response -
    this is the entire "change the opioid, watch the propofol
    recommendation update" interaction. When no opioid is selected, this
    falls back to the original fully-free recommend_su2023_regimen (there
    is no remifentanil rate to pin).

    Deliberately prevent_initial_call=True with nav-scenario-btn.n_clicks as
    one of the triggers (rather than firing automatically on every app
    load): this is an expensive multi-second Powell optimization, and
    without this guard it would re-run on every single page load for every
    visitor even if they never open Scenario Exploration. Clicking into the
    page the first time is what computes its initial baseline; every
    subsequent patient/target/opioid/rate edit re-triggers it the normal
    way. Every text field feeding this is debounce=True, and the
    remifentanil slider is updatemode="mouseup" (see
    build_scenario_medication_card), so this never fires mid-keystroke or
    mid-drag.
    """
    try:
        age = float(age)
        height = float(height)
        weight = float(weight)
        sbp = float(sbp)
        dbp = float(dbp)

        if age <= 0 or height <= 0 or weight <= 0:
            raise ValueError("Age, height, and weight must be positive.")
        if sbp <= 0 or dbp <= 0 or sbp <= dbp:
            raise ValueError("SBP must be positive and higher than DBP.")

        bis_low = float(bis_low)
        bis_high = float(bis_high)
        map_value = float(map_value)

        if bis_low <= 0 or bis_high <= 0 or bis_low >= bis_high:
            raise ValueError("BIS targets must be positive, with lower below upper.")
        if map_value <= 0:
            raise ValueError("MAP target must be positive.")

        # The compact Target MAP control only exposes one value at a time
        # (mode picks which); whichever mode isn't active keeps its normal
        # module default rather than being left unset - map_abs_min_target
        # and map_rel_frac_target are always both required together.
        if map_mode == "rel":
            map_abs = float(MAP_ABS_MIN_TARGET)
            map_rel_pct = map_value
        else:
            map_abs = map_value
            map_rel_pct = float(MAP_REL_FRAC_TARGET) * 100.0

        map_rel_frac = map_rel_pct / 100.0
        opioid = opioid or "none"
        if opioid in {"sufentanil", "fentanyl"}:
            raise ValueError("Only remifentanil is currently supported.")

        patient = _scenario_patient_from_fields(age, height, weight, sex, sbp, dbp)

        if opioid == "remifentanil":
            remi_rate_mcgkgmin = float(remi_rate_mcgkgmin)
            if remi_rate_mcgkgmin < 0:
                raise ValueError("Remifentanil infusion rate must be non-negative.")
            rec = recommend_propofol_for_fixed_remifentanil_rate(
                patient=patient,
                fixed_remi_rate_mcgkgmin=remi_rate_mcgkgmin,
                propofol_conc_mg_ml=DEFAULT_PROPOFOL_CONC_MG_ML,
                remifentanil_conc_mcg_ml=DEFAULT_REMI_CONC_MCG_ML,
                target_bis_low=bis_low,
                target_bis_high=bis_high,
                map_abs_min_target=map_abs,
                map_rel_frac_target=map_rel_frac,
            )
        else:
            rec = recommend_su2023_regimen(
                patient=patient,
                opiate="none",
                propofol_conc_mg_ml=DEFAULT_PROPOFOL_CONC_MG_ML,
                remifentanil_conc_mcg_ml=DEFAULT_REMI_CONC_MCG_ML,
                target_bis_low=bis_low,
                target_bis_high=bis_high,
                map_abs_min_target=map_abs,
                map_rel_frac_target=map_rel_frac,
            )

        normalized_opioid = "remifentanil" if opioid == "remifentanil" else "none"

        return {
            "error": None,
            "context": {
                "age": age, "height": height, "weight": weight, "sex": sex or "male",
                "sbp": sbp, "dbp": dbp,
                "opiate": normalized_opioid,
                "propofol_conc_mg_ml": DEFAULT_PROPOFOL_CONC_MG_ML,
                "remifentanil_conc_mcg_ml": DEFAULT_REMI_CONC_MCG_ML,
                "target_bis_low": bis_low,
                "target_bis_high": bis_high,
                "map_abs_min_target": map_abs,
                "map_rel_frac_target": map_rel_frac,
            },
            "result": _rec_to_store_dict(rec),
        }

    except Exception as e:
        return {"error": str(e), "context": None, "result": None}


def _baseline_representative_remi_rate(remi_rates) -> float:
    """
    Collapse a (possibly 2-segment, pause + rate_1/rate_2) baseline
    remifentanil schedule down to the single constant rate Scenario
    Exploration's one slider/input controls - the max of the schedule,
    matching whatever "Reset to recommended" sets the slider to. Shared by
    reset_scenario_medication and render_scenario_page's "does the
    scenario match the recommendation" check, so both use the exact same
    definition of "the baseline's rate".
    """
    if remi_rates is None:
        return SCENARIO_DEFAULT_REMI_MCGKGMIN
    remi_rates = np.asarray(remi_rates, dtype=float)
    if remi_rates.size == 0:
        return SCENARIO_DEFAULT_REMI_MCGKGMIN
    return float(np.max(remi_rates))


@app.callback(
    Output("se-remi-rate-slider", "value", allow_duplicate=True),
    Input("se-reset-medication-btn", "n_clicks"),
    State("se-age", "value"),
    State("se-sex", "value"),
    State("se-height", "value"),
    State("se-weight", "value"),
    State("se-sbp", "value"),
    State("se-dbp", "value"),
    State("se-opioid-dropdown", "value"),
    State("se-bis-target-low", "value"),
    State("se-bis-target-high", "value"),
    State("se-map-target-mode", "value"),
    State("se-map-target-value", "value"),
    prevent_initial_call=True,
)
def reset_scenario_medication(
    _n_clicks, age, sex, height, weight, sbp, dbp, opioid, bis_low, bis_high, map_mode, map_value,
):
    """
    Reset the remifentanil rate slider to what the model would recommend
    with NEITHER drug pinned - a fresh, fully free recommend_su2023_regimen
    call (the same joint optimization the main Recommendation page uses),
    not simply the current se-baseline-store (which, since
    compute_scenario_baseline now always pins remifentanil at whatever the
    slider already holds, would just echo the slider's current value back
    at itself). This is the one place Scenario Exploration asks "what does
    the model recommend with everything free?" - it's an expensive Powell
    + confidence call, same cost as compute_scenario_baseline, but only
    runs on this explicit button click.
    """
    if opioid != "remifentanil":
        return no_update

    try:
        age = float(age)
        height = float(height)
        weight = float(weight)
        sbp = float(sbp)
        dbp = float(dbp)
        bis_low = float(bis_low)
        bis_high = float(bis_high)
        map_value = float(map_value)

        if map_mode == "rel":
            map_abs = float(MAP_ABS_MIN_TARGET)
            map_rel_pct = map_value
        else:
            map_abs = map_value
            map_rel_pct = float(MAP_REL_FRAC_TARGET) * 100.0

        patient = _scenario_patient_from_fields(age, height, weight, sex, sbp, dbp)
        rec = recommend_su2023_regimen(
            patient=patient,
            opiate="remifentanil",
            propofol_conc_mg_ml=DEFAULT_PROPOFOL_CONC_MG_ML,
            remifentanil_conc_mcg_ml=DEFAULT_REMI_CONC_MCG_ML,
            target_bis_low=bis_low,
            target_bis_high=bis_high,
            map_abs_min_target=map_abs,
            map_rel_frac_target=map_rel_pct / 100.0,
        )
        return _baseline_representative_remi_rate(rec.remifentanil_inf_rates_mcgkgmin)

    except Exception:
        return no_update


@app.callback(
    Output("se-summary-opioid-strategy", "children"),
    Output("se-summary-rec-propofol-induction", "children"),
    Output("se-summary-rec-opioid-maintenance", "children"),
    Output("se-summary-rec-propofol-maintenance", "children"),
    Input("se-baseline-store", "data"),
)
def render_scenario_recommendation_summary(baseline_store):
    """
    Fill in the "Explored scenario" card's 4 recommendation-reference
    values. Depends only on se-baseline-store (never the dose/rate
    sliders) since none of these values come from the live scenario -
    they describe what the model recommends, not what the user is
    currently exploring.
    """
    if not baseline_store or baseline_store.get("error") or not baseline_store.get("result"):
        return "-", "-", "-", "-"

    context = baseline_store["context"]
    result = baseline_store["result"]
    opioid_selected = context["opiate"] == "remifentanil"

    opioid_strategy = "Remifentanil" if opioid_selected else "None"
    propofol_induction = f"{float(result['propofol_bolus_mgkg']):.2f} mg/kg"

    if opioid_selected:
        remi_rate = _baseline_representative_remi_rate(result.get("remifentanil_inf_rates_mcgkgmin"))
        opioid_maintenance = f"{remi_rate:.2f} µg/kg/min"
    else:
        opioid_maintenance = "-"

    prop_maint_rates = result.get("propofol_inf_rates_mcgkgmin")
    if prop_maint_rates:
        prop_maint_rate = float(np.max(np.asarray(prop_maint_rates, dtype=float)))
        propofol_maintenance = f"{prop_maint_rate:.0f} µg/kg/min"
    else:
        propofol_maintenance = "-"

    return opioid_strategy, propofol_induction, opioid_maintenance, propofol_maintenance


@app.callback(
    Output("se-dose-response-graph", "figure"),
    Output("se-bis-graph", "figure"),
    Output("se-map-graph", "figure"),
    Output("se-propofol-pk-graph", "figure"),
    Output("se-remifentanil-pk-graph", "figure"),
    Input("se-baseline-store", "data"),
)
def render_scenario_page(baseline_store):
    """
    Render the dose-response sweep and the 4 prediction graphs straight
    from se-baseline-store. (The "Explored scenario" card's own text is
    rendered separately by render_scenario_recommendation_summary.)

    Propofol is purely an output of this page now (see
    compute_scenario_baseline, which re-optimizes it for whatever
    remifentanil rate is set) - there is no separate "current scenario"
    trajectory to simulate or compare against a recommendation anymore, so
    every graph is built directly from the one stored recommendation via
    the exact same single-curve functions (make_bis_figure/make_map_figure/
    make_propofol_pk_figure/make_remifentanil_pk_figure) used when no
    manual override is active on the Recommendation page - unchanged,
    reused as-is.
    """
    empty = make_empty_figure()

    if (
        not baseline_store or baseline_store.get("error")
        or not baseline_store.get("context") or not baseline_store.get("result")
    ):
        return empty, empty, empty, empty, empty

    context = baseline_store["context"]
    baseline = _store_dict_to_namespace(baseline_store["result"])

    patient = _scenario_patient_from_fields(
        context["age"], context["height"], context["weight"], context["sex"],
        context["sbp"], context["dbp"],
    )

    dose_response_fig = make_induction_dose_rationale_figure(
        patient=patient,
        rec=baseline,
        opiate=context["opiate"],
        propofol_conc_mg_ml=context["propofol_conc_mg_ml"],
        remifentanil_conc_mcg_ml=context["remifentanil_conc_mcg_ml"],
        compact=True,
    )

    return (
        dose_response_fig,
        make_bis_figure(baseline),
        make_map_figure(baseline),
        make_propofol_pk_figure(baseline),
        make_remifentanil_pk_figure(baseline),
    )


# ============================================================
# Precomputed Remi - real-model outputs for fixed Phase 3 patient cases,
# loaded once at startup (PRECOMPUTED_REMI_DATA, above) from
# src/propofol/data/precomputed_remi_cases.json, generated offline by
# scripts/precompute_remi_cases.py. Nothing below this point ever imports
# or calls recommend_su2023_regimen, Su2023PropofolRemifentanilRecommender,
# or any other recommend_regimen2023.py entry point - every number comes
# straight out of the loaded JSON via _rec_to_store_dict/
# _store_dict_to_namespace (the exact same two functions the
# Recommendation page's own recommendation-store round-trip already
# uses), and every prediction figure reuses the Recommendation page's own
# make_bis_figure/make_bis_figure_dual family unmodified.
# RECOMMENDED_COLOR/MANUAL_OVERRIDE_COLOR are already exactly green/orange
# (#1f9254/#c2680f), matching Test Exploration's own Baseline
# Recommendation/Explored Scenario color language, so no new colors are
# introduced anywhere on this page.
# ============================================================

def _pr_remi_rate_display(ns) -> str:
    """"No opioid" or "Remifentanil X.XXX µg/kg/min" (the achieved rate, not the requested grid value) for a baseline or grid-point namespace."""
    if not ns.remifentanil_selected or ns.remifentanil_inf_rates_mcgkgmin is None:
        return "No opioid"
    rate = float(ns.remifentanil_inf_rates_mcgkgmin[-1])
    return f"Remifentanil {rate:.3f} µg/kg/min"


def make_pr_remifentanil_pk_figure(baseline_ns, explored_ns):
    """
    Own thin wrapper around make_remifentanil_pk_figure_dual: that shared
    function returns early (an empty figure) whenever the *baseline* has
    no remifentanil, regardless of what the manual/explored side has -
    correct for the Recommendation page (a manual override can never
    introduce an opioid the original recommendation didn't already have),
    but wrong here, where the baseline reference and the explored grid
    point are two independently real, potentially different opioid
    choices (e.g. a "No opioid" baseline compared against a real
    remifentanil grid point). Falls back to the shared dual function
    unchanged whenever the baseline does have remifentanil, so this only
    adds the one case that function doesn't already handle.
    """
    if baseline_ns.remifentanil_selected:
        return make_remifentanil_pk_figure_dual(baseline_ns, explored_ns, manual_name="Explored Scenario")
    if not explored_ns.remifentanil_selected:
        return make_remifentanil_pk_figure(baseline_ns)

    fig = go.Figure()
    _add_line(fig, explored_ns.time_min, explored_ns.cp_remifentanil, "Explored Scenario", color=MANUAL_OVERRIDE_COLOR)
    fig.update_layout(
        xaxis_title="Time (min)",
        yaxis_title="Remifentanil concentration (ng/mL)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=60),
    )
    return fig


def make_pr_induction_dose_rationale_figure(baseline_ns, baseline_map: float, explored_ns=None):
    """
    Precomputed Remi's own version of the Recommendation page's
    make_induction_dose_rationale_figure (that function itself is never
    modified or called live from this page - see its own docstring
    above). Reproduces the exact same visual structure - MAP target band
    (red), BIS target band (blue), the green "both targets met" dose
    region with its own annotated bounds, diamond markers, reversed BIS
    axis, green dashed Baseline / orange dashed Explored reference lines
    - but reads the propofol induction-dose sweep from baseline_ns.
    dose_sweep (and explored_ns.dose_sweep, when given) instead of
    calling Su2023PropofolRemifentanilRecommender.simulate_regimen()
    live: those arrays are precomputed once offline by
    scripts/precompute_remi_cases.py's own _dose_sweep() - the exact same
    forward-simulation logic, just run at precompute time - so nothing on
    this page ever triggers a live model call.

    The plotted MAP/BIS curves are the *baseline* regimen's own sweep
    (varying induction dose, baseline's own maintenance schedule held
    fixed) - matching the Recommendation page's own one-curve-per-
    regimen structure. When explored_ns is given, its diamond marker is
    still plotted at its own true precomputed (min MAP, max BIS) - read
    from explored_ns's own dose_sweep at its own dose, not sampled off
    the baseline curve - since baseline and an explored remifentanil-grid
    point are two independently optimized regimens with their own
    maintenance schedules, unlike the Recommendation page's manual-
    override case (which only ever changes the dose, never the schedule,
    so one sweep always covers both).

    baseline_map is this case's own baseline (pre-drug) MAP in mmHg -
    (baseline_sap + 2*baseline_dap) / 3, the same formula Patient itself
    uses for base_map - passed in rather than reconstructed here since
    the caller (update_pr_results) already has the case's own patient
    dict at hand.
    """
    sweep = baseline_ns.dose_sweep
    dose_grid_mgkg = np.asarray(sweep["dose_grid_mgkg"], dtype=float)
    min_maps = np.asarray(sweep["min_map_mmhg"], dtype=float)
    min_bis_values = np.asarray(sweep["min_bis"], dtype=float)
    max_bis_values = np.asarray(sweep["max_bis"], dtype=float)

    selected_dose_mgkg = float(baseline_ns.propofol_bolus_mgkg)
    map_target = float(baseline_ns.map_lower_bound_mmhg)
    map_target_upper = float(1.20 * baseline_map)
    target_bis_low = float(baseline_ns.target_bis_low)
    target_bis_high = float(baseline_ns.target_bis_high)

    explored_dose_mgkg = float(explored_ns.propofol_bolus_mgkg) if explored_ns is not None else None
    x_axis_min, x_axis_max = _dose_axis_limits_mgkg(selected_dose_mgkg, explored_dose_mgkg)

    selected_idx = int(np.argmin(np.abs(dose_grid_mgkg - selected_dose_mgkg)))
    selected_min_map = float(min_maps[selected_idx])
    selected_min_bis = float(min_bis_values[selected_idx])
    selected_max_bis = float(max_bis_values[selected_idx])

    map_axis_min, map_axis_max = _strict_prediction_axis_range(
        min_maps, lower_floor=0.0, upper_ceiling=None,
        default_min=0.0, default_max=max(1.0, map_target_upper), min_span=1.0,
    )
    max_bis_axis_min, max_bis_axis_max = _strict_prediction_axis_range(
        max_bis_values, lower_floor=0.0, upper_ceiling=100.0,
        default_min=0.0, default_max=100.0, min_span=1.0,
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    map_band = _target_rect_y_limits(map_target, map_target_upper, map_axis_min, map_axis_max)
    if map_band is not None:
        fig.add_shape(
            type="rect", xref="paper", x0=0, x1=1, yref="y", y0=map_band[0], y1=map_band[1],
            fillcolor="rgba(220, 0, 0, 0.10)", line=dict(width=0), layer="below",
        )

    bis_band = _target_rect_y_limits(target_bis_low, target_bis_high, max_bis_axis_min, max_bis_axis_max)
    if bis_band is not None:
        fig.add_shape(
            type="rect", xref="paper", x0=0, x1=1, yref="y2", y0=bis_band[0], y1=bis_band[1],
            fillcolor="rgba(0, 85, 220, 0.10)", line=dict(width=0), layer="below",
        )

    both_targets_ok = (
        np.isfinite(min_maps) & np.isfinite(min_bis_values) & np.isfinite(max_bis_values)
        & (min_maps >= map_target) & (min_maps <= map_target_upper)
        & (min_bis_values >= target_bis_low) & (max_bis_values <= target_bis_high)
    )
    # Text version of the target-dose-range boundaries, returned alongside
    # the figure for the "Target dose range" info box below the graph
    # (build_pr_dose_response_card in dashboard_layout.py) - no boundary
    # numbers are drawn inside the plot itself any more (see the removed
    # per-boundary annotations that used to sit just above the green
    # band); "-" when no dose achieves both targets simultaneously.
    dose_range_text = "–"
    if np.any(both_targets_ok):
        ok_doses = dose_grid_mgkg[both_targets_ok]
        ok_start = float(np.min(ok_doses))
        ok_end = float(np.max(ok_doses))
        dose_range_text = f"{ok_start:.2f}–{ok_end:.2f} mg/kg"

        if ok_end > ok_start:
            band_x0, band_x1 = ok_start, ok_end
        else:
            local_step = float(np.nanmedian(np.diff(dose_grid_mgkg))) if len(dose_grid_mgkg) > 1 else 0.05
            band_half_width = max(0.025, 0.5 * local_step)
            band_x0 = max(x_axis_min, ok_start - band_half_width)
            band_x1 = min(x_axis_max, ok_end + band_half_width)

        fig.add_shape(
            type="rect", xref="x", yref="paper", x0=band_x0, x1=band_x1, y0=0, y1=1,
            fillcolor="rgba(0, 150, 0, 0.10)", line=dict(width=0), layer="below",
        )

    fig.add_trace(
        go.Scatter(
            x=dose_grid_mgkg, y=min_maps, mode="lines+markers",
            line=dict(color="red", width=3), marker=dict(color="red", size=6),
            name="MAP", showlegend=True,
            hovertemplate="Dose %{x:.2f} mg/kg<br>Minimal MAP %{y:.1f} mmHg<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=dose_grid_mgkg, y=max_bis_values, customdata=np.stack([min_bis_values], axis=-1),
            mode="lines+markers", line=dict(color="blue", width=3), marker=dict(color="blue", size=6),
            name="BIS", showlegend=True,
            hovertemplate=(
                "Dose %{x:.2f} mg/kg<br>Maximal BIS %{y:.1f}<br>Minimal BIS %{customdata[0]:.1f}<extra></extra>"
            ),
        ),
        secondary_y=True,
    )

    diamond = dict(symbol="diamond", size=14)
    fig.add_trace(
        go.Scatter(
            x=[selected_dose_mgkg], y=[selected_min_map], mode="markers",
            marker=dict(color="red", **diamond), showlegend=False,
            hovertemplate=(
                f"Baseline {selected_dose_mgkg:.2f} mg/kg<br>Minimal MAP %{{y:.1f}} mmHg<br>"
                f"MAP target {map_target:.1f}-{map_target_upper:.1f} mmHg<extra></extra>"
            ),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=[selected_dose_mgkg], y=[selected_max_bis], customdata=[[selected_min_bis]], mode="markers",
            marker=dict(color="blue", **diamond), showlegend=False,
            hovertemplate=(
                f"Baseline {selected_dose_mgkg:.2f} mg/kg<br>Maximal BIS %{{y:.1f}}<br>"
                f"Minimal BIS %{{customdata[0]:.1f}}<br>BIS target {target_bis_low:.0f}-{target_bis_high:.0f}<extra></extra>"
            ),
        ),
        secondary_y=True,
    )

    fig.add_shape(
        type="line", xref="x", yref="paper", x0=selected_dose_mgkg, x1=selected_dose_mgkg, y0=0, y1=1,
        line=dict(color=RECOMMENDED_COLOR, width=3, dash="dash"), layer="above",
    )
    # Two-line "<label>\n<dose>" annotations (label colored, dose value in
    # a neutral dark color). Both normally sit at the same y, each
    # centered above its own dose's x position - but when the baseline and
    # explored doses are close together on the x-axis, same-y placement
    # makes the two (2-line-tall) labels visually collide. baseline_y/
    # explored_y are computed once below, as a fraction of the *actual*
    # x-axis range (not a fixed pixel/dose threshold), so the labels never
    # overlap regardless of the current dose values or how narrow/wide the
    # rendered plot is - close doses stagger vertically instead, still
    # each directly above its own dashed line (no horizontal shift).
    dose_label_value_color = "#1f2330"

    def _dose_label_annotation(x_value: float, y_value: float, label: str, color: str, value_text: str) -> None:
        fig.add_annotation(
            x=x_value, y=y_value, xref="x", yref="paper",
            text=f'<span style="color:{color}"><b>{label}</b></span><br><b>{value_text}</b>',
            showarrow=False, yanchor="bottom", align="center",
            font=dict(color=dose_label_value_color, size=13),
        )

    baseline_y = 1.05
    explored_y = 1.05
    if explored_dose_mgkg is not None:
        x_range = x_axis_max - x_axis_min
        dose_gap_frac = abs(explored_dose_mgkg - selected_dose_mgkg) / x_range if x_range > 0 else 0.0
        if dose_gap_frac < 0.22:
            explored_y = 1.34

    _dose_label_annotation(
        selected_dose_mgkg, baseline_y, "Baseline dose", RECOMMENDED_COLOR, f"{selected_dose_mgkg:.2f} mg/kg",
    )

    if explored_ns is not None:
        explored_sweep = explored_ns.dose_sweep
        explored_dose_grid = np.asarray(explored_sweep["dose_grid_mgkg"], dtype=float)
        explored_idx = int(np.argmin(np.abs(explored_dose_grid - explored_dose_mgkg)))
        explored_min_map = float(np.asarray(explored_sweep["min_map_mmhg"], dtype=float)[explored_idx])
        explored_max_bis = float(np.asarray(explored_sweep["max_bis"], dtype=float)[explored_idx])

        fig.add_trace(
            go.Scatter(
                x=[explored_dose_mgkg], y=[explored_min_map], mode="markers",
                marker=dict(color=MANUAL_OVERRIDE_COLOR, **diamond), showlegend=False,
                hovertemplate=f"Explored {explored_dose_mgkg:.2f} mg/kg<br>Minimal MAP %{{y:.1f}} mmHg<extra></extra>",
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(
                x=[explored_dose_mgkg], y=[explored_max_bis], mode="markers",
                marker=dict(color=MANUAL_OVERRIDE_COLOR, **diamond), showlegend=False,
                hovertemplate=f"Explored {explored_dose_mgkg:.2f} mg/kg<br>Maximal BIS %{{y:.1f}}<extra></extra>",
            ),
            secondary_y=True,
        )
        fig.add_shape(
            type="line", xref="x", yref="paper", x0=explored_dose_mgkg, x1=explored_dose_mgkg, y0=0, y1=1,
            line=dict(color=MANUAL_OVERRIDE_COLOR, width=3, dash="dash"), layer="above",
        )
        _dose_label_annotation(
            explored_dose_mgkg, explored_y, "Explored dose", MANUAL_OVERRIDE_COLOR, f"{explored_dose_mgkg:.2f} mg/kg",
        )

    # No more top legend (the old MAP/BIS/Target/Baseline/Explored legend
    # row) - showlegend=False suppresses it regardless of any trace's own
    # showlegend value, so the MAP/BIS data traces above (still name="MAP"
    # /"BIS" for hover/debugging purposes) never render a legend either.
    # yaxis/yaxis2 titles are intentionally empty: the axis-title text and
    # its arrow are now drawn as a compact HTML/CSS overlay beside the
    # graph (see build_pr_dose_response_card in dashboard_layout.py).
    #
    # Margins: yref="paper" annotation y-values above 1.0 only stay on
    # canvas if the top margin (an absolute pixel count) covers that
    # excess once translated through the plot area's own pixel height -
    # and that pixel height grows with the card (#pr-dose-response-
    # card-wrapper .graph-card--tall in style.css sets it to 620px), so
    # the margin has to be sized for that card height, not guessed.
    # Worked backwards from the actual rendered card: content area is
    # roughly 560px tall, so with a 46px bottom margin the plot area
    # (paper y 0-to-1) is (560 - t - 46) px tall; the staggered label's
    # anchor at y=1.34 sits t - 0.34*(560-t-46) px below the figure's own
    # top edge, and since the annotation is yanchor="bottom" its 2-line
    # text then grows *upward* from that anchor by another ~35px - so
    # t=185 leaves that whole 2-line block comfortably on-canvas instead
    # of overlapping the card header above it. l/r=54 leaves room for
    # the y tick labels next to the now much narrower (~22px) CSS
    # arrow-label columns; b=46 fits the x-axis title without touching
    # the bottom info boxes, which live outside the graph entirely.
    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        margin=dict(t=185, b=46, l=54, r=54),
        xaxis=dict(
            title=dict(text="Propofol induction dose (mg/kg)", font=dict(color="green")),
            tickfont=dict(color="green"), color="green",
            range=[x_axis_min, x_axis_max],
        ),
        yaxis=dict(
            title=dict(text=""),
            tickfont=dict(color="red"), color="red",
            range=[map_axis_min, map_axis_max],
        ),
        yaxis2=dict(
            title=dict(text=""),
            tickfont=dict(color="blue"), color="blue",
            range=[max_bis_axis_max, max_bis_axis_min],
            overlaying="y", side="right",
        ),
    )

    return SimpleNamespace(
        figure=fig,
        map_zone_text=f"≥ {map_target:.0f} mmHg",
        bis_zone_text=f"{target_bis_low:.0f}–{target_bis_high:.0f}",
        dose_range_text=dose_range_text,
    )


def _pr_delta(baseline_val: float, explored_val: float) -> tuple[str, str]:
    """
    Shared percent-change text/class logic for every Scenario Result
    change cell (Propofol induction, Propofol maintenance, Remifentanil/
    opioid strategy) - direction only, not a clinical good/bad judgement,
    so colors are inverted from Test Exploration's own te-delta-good/
    te-delta-bad (there, decrease=green/"good"; here, increase=green,
    decrease=red, via the pr-delta-increase/pr-delta-decrease classes
    added in style.css - te-delta-good/te-delta-bad themselves are never
    touched, so Test Exploration's own coloring is unaffected).
    """
    pct = ((explored_val - baseline_val) / baseline_val) * 100 if baseline_val else 0.0
    if abs(pct) < 0.5:
        return "→ 0%", "te-result-cell te-result-change-value"
    arrow = "↑" if pct > 0 else "↓"
    change_class = "te-result-cell te-result-change-value " + ("pr-delta-increase" if pct > 0 else "pr-delta-decrease")
    return f"{arrow} {abs(pct):.0f}%", change_class


def _pr_result_row(label: str, baseline_val, explored_val, fmt: str = "{:.2f}"):
    """One te-result-row-shaped comparison row (reusing Test Exploration's own CSS classes) with a computed percent change."""
    if baseline_val is None or explored_val is None:
        baseline_text = fmt.format(baseline_val) if baseline_val is not None else "-"
        explored_text = fmt.format(explored_val) if explored_val is not None else "-"
        change_text, change_class = "–", "te-result-cell te-result-change-value"
    else:
        baseline_text = fmt.format(baseline_val)
        explored_text = fmt.format(explored_val)
        change_text, change_class = _pr_delta(baseline_val, explored_val)

    return html.Div(
        [
            html.Div(label, className="te-result-cell te-result-row-label"),
            html.Div(baseline_text, className="te-result-cell te-result-baseline-value"),
            html.Div(className="te-result-cell te-result-arrow-cell"),
            html.Div(explored_text, className="te-result-cell te-result-explored-value"),
            html.Div(change_text, className=change_class),
        ],
        className="te-result-row",
    )


def _pr_text_row(label: str, baseline_text: str, explored_text: str):
    """Same shape as _pr_result_row, for cells that are already plain text (e.g. "Pause") rather than a number to compute a percent change from."""
    return html.Div(
        [
            html.Div(label, className="te-result-cell te-result-row-label"),
            html.Div(baseline_text, className="te-result-cell te-result-baseline-value"),
            html.Div(className="te-result-cell te-result-arrow-cell"),
            html.Div(explored_text, className="te-result-cell te-result-explored-value"),
            html.Div("–", className="te-result-cell te-result-change-value"),
        ],
        className="te-result-row",
    )


def _pr_text_row_with_delta(label: str, baseline_text: str, explored_text: str, baseline_val: float, explored_val: float):
    """
    Same shape as _pr_text_row (baseline/explored shown as already-
    formatted text, e.g. "52 mL/h (120 µg/kg/min)"), but with a real
    colored percent-change in the Change column, computed from the
    underlying numeric rate - used for Propofol maintenance rows so their
    Change column behaves like Propofol Induction/Remifentanil rate
    instead of always showing a static dash.
    """
    change_text, change_class = _pr_delta(baseline_val, explored_val)
    return html.Div(
        [
            html.Div(label, className="te-result-cell te-result-row-label"),
            html.Div(baseline_text, className="te-result-cell te-result-baseline-value"),
            html.Div(className="te-result-cell te-result-arrow-cell"),
            html.Div(explored_text, className="te-result-cell te-result-explored-value"),
            html.Div(change_text, className=change_class),
        ],
        className="te-result-row",
    )


def _pr_induction_row(baseline_mgkg: float, baseline_mg: float, explored_mgkg: float = None, explored_mg: float = None):
    """
    Combined "Propofol – Induction" row - one row instead of two separate
    "(mg/kg)"/"(mg total)" rows, matching Test Exploration's own combined
    induction display (te-result-baseline-induction-value/-total,
    te-result-explored-induction-value/-total - all shared, unmodified
    CSS classes/markup shape). mg/kg stays the primary (larger) value;
    mg total is shown underneath, with the pr-induction-total override
    class (style.css) making it larger than Test Exploration's own 10px
    subtitle while staying visually secondary to mg/kg.

    baseline_mg/explored_mg are propofol_bolus_mg exactly as the optimizer
    returned it (recommend_regimen2023.py's own round_to_step(...,
    PROPOFOL_BOLUS_STEP_MG) already rounds it to the nearest 5 mg before
    it is ever stored) - displayed with the same "{:.0f}" formatting the
    Recommendation page's own induction-dose card uses (app.py's
    build_induction_dose_card, "induction-dose-number"), with no further
    rounding here. This keeps the two pages' displayed induction dose
    identical for the same recommendation. explored_mgkg=None (nothing
    explored yet) shows the same placeholder look Test Exploration uses
    before an opioid is picked.
    """
    baseline_cell = html.Div(
        [
            html.Span(f"{float(baseline_mgkg):.2f} mg/kg", className="te-result-baseline-induction-value"),
            html.Span(
                f"({float(baseline_mg):.0f} mg total)",
                className="te-result-baseline-induction-total pr-induction-total",
            ),
        ],
        className="te-result-cell te-result-baseline-value",
    )

    if explored_mgkg is None:
        explored_cell = html.Div(
            html.Span("-", className="te-result-explored-induction-value te-result-explored-value--placeholder"),
            className="te-result-cell te-result-explored-value",
        )
        change_text, change_class = "–", "te-result-cell te-result-change-value"
    else:
        explored_cell = html.Div(
            [
                html.Span(f"{float(explored_mgkg):.2f} mg/kg", className="te-result-explored-induction-value"),
                html.Span(
                    f"({float(explored_mg):.0f} mg total)",
                    className="te-result-explored-induction-total pr-induction-total",
                ),
            ],
            className="te-result-cell te-result-explored-value",
        )
        change_text, change_class = _pr_delta(float(baseline_mgkg), float(explored_mgkg))

    return html.Div(
        [
            html.Div("Propofol – Induction", className="te-result-cell te-result-row-label"),
            baseline_cell,
            html.Div(className="te-result-cell te-result-arrow-cell"),
            explored_cell,
            html.Div(change_text, className=change_class),
        ],
        className="te-result-row",
    )


def _pr_maintenance_rows(baseline_ns, explored_ns, drug: str = "propofol") -> list:
    """
    One row per compressed maintenance interval for the given drug
    ("propofol" or "remifentanil"), using the EXPLORED regimen's own
    compress_minute_schedule() boundaries (always well-defined - every
    grid point is a fresh, complete optimization) and sampling the
    baseline regimen's own rate at each interval's start minute for
    direct comparison. Baseline and explored can have different switch
    times, but each drug's own _inf_rates_ml_h/_secondary arrays are
    always exactly N_INTERVALS=15 minute-indexed entries for both sides
    (whenever that drug is present at all), so index-sampling baseline by
    minute is always safe, never a length mismatch. Change column is a
    real colored percent-change (mirroring Propofol Induction/
    Remifentanil rate) whenever both sides are actually infusing; a Pause
    interval (rate 0 on either side) keeps the static dash, since a
    percent change against/to zero isn't meaningful.

    Formatting - "Pause" / "{rate:.0f} mL/h ({secondary:.0f} {unit})" -
    intentionally matches the Recommendation page's own _maintenance_rows()
    exactly (same rounding, same "Pause" text, same mL/h + secondary-unit
    pairing), so the two pages show identical numbers for identical
    regimens; only the row shape differs here (this page's baseline/
    explored comparison columns vs. that page's single interval/arrow/
    rate schedule table) - _maintenance_rows() itself is not reused
    because its DOM shape doesn't fit this page's comparison grid.

    remifentanil_selected can differ between baseline and explored (e.g. a
    "No opioid" baseline compared against a real remifentanil grid point)
    - when the baseline has no schedule at all for this drug
    (baseline_ml is None), every explored segment is shown against a flat
    "No opioid" baseline cell instead of a sampled rate.
    """
    if drug == "propofol":
        ml_field = "propofol_inf_rates_ml_h"
        secondary_field = "propofol_inf_rates_mcgkgmin"
        secondary_label = "µg/kg/min"
    elif drug == "remifentanil":
        ml_field = "remifentanil_inf_rates_ml_h"
        secondary_field = "remifentanil_inf_rates_ngkgmin"
        secondary_label = "ng/kg/min"
    else:
        raise ValueError(f"Unknown drug: {drug!r}")

    explored_ml = getattr(explored_ns, ml_field)
    if explored_ml is None:
        return []

    intervals = compress_minute_schedule(explored_ml)
    explored_secondary = getattr(explored_ns, secondary_field)
    baseline_ml = getattr(baseline_ns, ml_field)
    baseline_secondary = getattr(baseline_ns, secondary_field)

    rows = []
    for interval in intervals:
        start = int(interval["start_min"])
        end = int(interval["end_min"])
        label = f"{start}–{end} min"

        explored_rate = float(interval["rate"])
        explored_secondary_rate = float(explored_secondary[min(start, len(explored_secondary) - 1)])
        explored_paused = np.isclose(explored_rate, 0.0, atol=1e-8)
        explored_text = (
            "Pause" if explored_paused
            else f"{explored_rate:.0f} mL/h ({explored_secondary_rate:.0f} {secondary_label})"
        )

        if baseline_ml is None:
            rows.append(_pr_text_row(label, "No opioid", explored_text))
            continue

        baseline_rate = float(baseline_ml[min(start, len(baseline_ml) - 1)])
        baseline_secondary_rate = float(baseline_secondary[min(start, len(baseline_secondary) - 1)])
        baseline_paused = np.isclose(baseline_rate, 0.0, atol=1e-8)
        baseline_text = (
            "Pause" if baseline_paused
            else f"{baseline_rate:.0f} mL/h ({baseline_secondary_rate:.0f} {secondary_label})"
        )

        if baseline_paused or explored_paused:
            rows.append(_pr_text_row(label, baseline_text, explored_text))
        else:
            rows.append(_pr_text_row_with_delta(label, baseline_text, explored_text, baseline_rate, explored_rate))

    return rows


def _build_pr_result_grid(baseline_ns, explored_ns) -> list:
    rows = [_te_result_header_row()]

    rows.append(_pr_induction_row(
        float(baseline_ns.propofol_bolus_mgkg), float(baseline_ns.propofol_bolus_mg),
        float(explored_ns.propofol_bolus_mgkg), float(explored_ns.propofol_bolus_mg),
    ))

    rows.append(_te_result_section_header(
        "Propofol – Early maintenance (0–15 min)", "Infusion regimen relative to induction",
    ))
    rows.extend(_pr_maintenance_rows(baseline_ns, explored_ns, drug="propofol"))

    rows.append(_te_result_section_header(
        "Remifentanil – Early maintenance (0–15 min)", "Infusion regimen relative to induction",
    ))
    remi_rows = _pr_maintenance_rows(baseline_ns, explored_ns, drug="remifentanil")
    rows.extend(remi_rows if remi_rows else [_pr_text_row("Remifentanil", "-", "-")])

    return rows


def _build_pr_result_grid_baseline_only(baseline_ns) -> list:
    """
    "Explore opioid strategy" = None: only the Baseline Recommendation
    column has real data - the Explored Scenario column and its percent
    change show placeholder dashes, since no opioid rate is currently
    being explored (requirement: "Explored Scenario column should stay
    empty or show placeholder values").
    """
    rows = [_te_result_header_row()]

    rows.append(_pr_induction_row(
        float(baseline_ns.propofol_bolus_mgkg), float(baseline_ns.propofol_bolus_mg),
    ))

    rows.append(_te_result_section_header(
        "Propofol – Early maintenance (0–15 min)", "Infusion regimen relative to induction",
    ))
    intervals = compress_minute_schedule(baseline_ns.propofol_inf_rates_ml_h)
    baseline_secondary = baseline_ns.propofol_inf_rates_mcgkgmin
    for interval in intervals:
        start = int(interval["start_min"])
        end = int(interval["end_min"])
        rate = float(interval["rate"])
        secondary_rate = float(baseline_secondary[min(start, len(baseline_secondary) - 1)])
        text = (
            "Pause" if np.isclose(rate, 0.0, atol=1e-8)
            else f"{rate:.0f} mL/h ({secondary_rate:.0f} µg/kg/min)"
        )
        rows.append(_pr_text_row(f"{start}–{end} min", text, "-"))

    rows.append(_te_result_section_header(
        "Remifentanil – Early maintenance (0–15 min)", "Infusion regimen relative to induction",
    ))
    baseline_has_remi = baseline_ns.remifentanil_selected and baseline_ns.remifentanil_inf_rates_ml_h is not None
    if baseline_has_remi:
        remi_intervals = compress_minute_schedule(baseline_ns.remifentanil_inf_rates_ml_h)
        baseline_remi_secondary = baseline_ns.remifentanil_inf_rates_ngkgmin
        for interval in remi_intervals:
            start = int(interval["start_min"])
            end = int(interval["end_min"])
            rate = float(interval["rate"])
            secondary_rate = float(baseline_remi_secondary[min(start, len(baseline_remi_secondary) - 1)])
            text = (
                "Pause" if np.isclose(rate, 0.0, atol=1e-8)
                else f"{rate:.0f} mL/h ({secondary_rate:.0f} ng/kg/min)"
            )
            rows.append(_pr_text_row(f"{start}–{end} min", text, "-"))
    else:
        rows.append(_pr_text_row("Remifentanil", "No opioid", "-"))

    return rows


@app.callback(
    Output("pr-rate-slider-wrapper", "style"),
    Input("pr-explore-opioid-dropdown", "value"),
    prevent_initial_call=True,
)
def toggle_pr_explore_slider(explore_opioid):
    """
    Requirement 2: the remifentanil rate slider only appears once
    "Opioid to explore" = Remifentanil is picked - "None" (or anything
    else, though Sufentanil/Fentanyl can never actually be selected since
    their dropdown options are disabled) keeps it hidden.
    """
    return None if explore_opioid == "remifentanil" else {"display": "none"}


@app.callback(
    Output("pr-detail-age", "value"),
    Output("pr-detail-sex", "value"),
    Output("pr-detail-height", "value"),
    Output("pr-detail-weight", "value"),
    Output("pr-detail-sbp", "value"),
    Output("pr-detail-dbp", "value"),
    Output("pr-detail-hr", "value"),
    Output("pr-detail-map", "children"),
    Output("pr-detail-pp", "children"),
    Output("pr-target-bis-low", "value"),
    Output("pr-target-bis-high", "value"),
    Output("pr-target-map-abs", "value"),
    Output("pr-target-map-rel", "value"),
    Output("pr-baseline-dropdown", "value"),
    Output("pr-case-error-banner", "style"),
    Output("pr-case-error-banner", "children"),
    Input("pr-case-dropdown", "value"),
)
def update_pr_patient_preview(case_id):
    """
    Live preview only: populates the read-only Patient scenario and
    Targets cards for whichever case is currently selected, and resets
    Medication scenario back to "None" - or, if that case's precomputed
    grid data is missing/malformed, shows pr-case-error-banner instead
    (a per-case data problem must show a clear message, never crash the
    app or silently show stale/wrong numbers). Deliberately does not
    touch pr-results-area/pr-selected-case-store/pr-rate-slider - those
    are only committed by run_pr_scenario, below, on "Run Scenario".

    No longer prevent_initial_call: pr-case-dropdown now defaults to "1"
    (Test Patient 1 is the only option), so this callback firing once on
    page load is exactly what populates the Patient scenario/Targets
    cards immediately, without requiring the user to touch the dropdown.
    """
    hidden = {"display": "none"}
    blank = ("-",) * 13 + ("none",)

    if not case_id:
        return blank + (hidden, no_update)

    case = _pr_case(case_id)
    grid = (case or {}).get("remifentanil_grid", {})
    if case is None or not grid.get("rates_mcgkgmin") or not grid.get("results"):
        error_text = (
            f"Case {case_id} has no precomputed grid data. "
            "Re-run scripts/precompute_remi_cases.py for this patient."
        )
        return blank + (None, error_text)

    p = case["patient"]
    map_val = p["baseline_dap"] + (p["baseline_sap"] - p["baseline_dap"]) / 3.0
    pp_val = p["baseline_sap"] - p["baseline_dap"]

    meta = (PRECOMPUTED_REMI_DATA or {}).get("metadata", {})
    bis_low = meta.get("target_bis_low")
    bis_high = meta.get("target_bis_high")
    map_abs = meta.get("map_abs_min_target_mmhg")
    map_rel = meta.get("map_rel_frac_target")

    return (
        f"{p['age']:.0f}",
        str(p["sex"]).capitalize(),
        f"{p['height']:.0f}",
        f"{p['weight']:.0f}",
        f"{p['baseline_sap']:.0f}",
        f"{p['baseline_dap']:.0f}",
        f"{p['baseline_hr']:.0f}",
        f"{map_val:.1f}",
        f"{pp_val:.1f}",
        f"{bis_low:.0f}" if bis_low is not None else "-",
        f"{bis_high:.0f}" if bis_high is not None else "-",
        f"{map_abs:.0f}" if map_abs is not None else "-",
        f"{map_rel * 100:.0f}" if map_rel is not None else "-",
        "none",
        hidden,
        "",
    )


@app.callback(
    Output("pr-scenario-active-store", "data", allow_duplicate=True),
    Input("pr-case-dropdown", "value"),
    Input("pr-baseline-dropdown", "value"),
    prevent_initial_call=True,
)
def mark_pr_scenario_stale(_case_id, _baseline_choice):
    """
    Mirrors Test Exploration's own markScenarioStale (clientside JS) in
    server-side Python: any change to the case or the medication
    (baseline) selection means whatever pr-results-area currently shows,
    if anything, no longer reflects the current left-column selection -
    so it hides again until Run Scenario is clicked, exactly like Test
    Exploration's own Patient/Medication cards require Run Scenario again
    after an edit.
    """
    return False


@app.callback(
    Output("pr-selected-case-store", "data"),
    Output("pr-committed-baseline-store", "data"),
    Output("pr-rate-slider", "value", allow_duplicate=True),
    Output("pr-scenario-active-store", "data", allow_duplicate=True),
    Input("pr-run-scenario-btn", "n_clicks"),
    State("pr-case-dropdown", "value"),
    State("pr-baseline-dropdown", "value"),
    prevent_initial_call=True,
)
def run_pr_scenario(_n_clicks, case_id, baseline_choice):
    """
    "Run Scenario" commit step, mirroring Test Exploration's own Run
    Scenario button: takes whatever case + medication (baseline) is
    currently selected in the left column and commits it, revealing/
    refreshing pr-results-area for it (via pr-scenario-active-store,
    below). No model computation happens here or anywhere else on this
    page - every number is already sitting in PRECOMPUTED_REMI_DATA; this
    callback only decides which of those already-computed numbers to
    show. If the selected case has no precomputed grid data,
    pr-results-area is kept hidden - update_pr_patient_preview, above,
    has already shown pr-case-error-banner for the same case.
    """
    case = _pr_case(case_id)
    grid = (case or {}).get("remifentanil_grid", {})
    if case is None or not grid.get("rates_mcgkgmin") or not grid.get("results"):
        return None, no_update, no_update, False

    rates = grid["rates_mcgkgmin"]
    # PR_DEFAULT_EXPLORE_RATE_MCGKGMIN, not rates[0]: rates[0] is now 0.00
    # (the grid's minimum), and silently resetting to "no remifentanil
    # infusion" on every Run Scenario click would be confusing.
    default_rate = rates[_pr_rate_to_index(rates, PR_DEFAULT_EXPLORE_RATE_MCGKGMIN)]
    return case_id, baseline_choice, default_rate, True


@app.callback(
    Output("pr-results-area", "style"),
    Input("pr-scenario-active-store", "data"),
    prevent_initial_call=True,
)
def render_pr_results_area(is_active):
    return None if is_active else {"display": "none"}


@app.callback(
    Output("pr-strategy-baseline-value", "children"),
    Output("pr-strategy-explored-value", "children"),
    Output("pr-result-grid", "children"),
    Output("pr-rate-label", "children"),
    Output("pr-bis-graph", "figure"),
    Output("pr-map-graph", "figure"),
    Output("pr-propofol-pk-graph", "figure"),
    Output("pr-remifentanil-pk-graph", "figure"),
    Output("pr-dose-response-graph", "figure"),
    Output("pr-dose-response-map-zone-value", "children"),
    Output("pr-dose-response-bis-zone-value", "children"),
    Output("pr-dose-response-dose-range-value", "children"),
    Input("pr-rate-slider", "value"),
    Input("pr-selected-case-store", "data"),
    Input("pr-committed-baseline-store", "data"),
    Input("pr-explore-opioid-dropdown", "value"),
    prevent_initial_call=True,
)
def update_pr_results(rate, case_id, baseline_choice, explore_opioid):
    """
    Everything shown for the committed case + committed baseline
    reference + current grid rate - all 12 outputs are read straight from
    PRECOMPUTED_REMI_DATA (already loaded, module-level, at startup) and
    reused figure-building functions. No model call happens here or
    anywhere else on this page - the slider only ever selects among the
    41 already-computed grid results (step=None + marks on the slider
    itself guarantees it can only land exactly on one of those 41 values,
    never an in-between one, and _pr_rate_to_index above does an exact -
    not nearest-value - lookup from that landed value to its grid index).
    case/baseline are the *committed* values (set only by run_pr_scenario,
    above, on "Run Scenario"), not the raw
    dropdown values, so pr-results-area's content only ever changes on a
    deliberate Run Scenario click, a rate-slider move, or a change to
    "Opioid to explore".

    "Opioid to explore" = None (or anything other than "remifentanil" -
    Sufentanil/Fentanyl can never actually be selected, their dropdown
    options are disabled) means nothing is currently being explored:
    Explored Strategy reads "Not selected", the Scenario Result grid's
    Explored Scenario column is all placeholder dashes
    (_build_pr_result_grid_baseline_only), every prediction graph shows
    only the baseline (single-series make_bis_figure/make_map_figure/
    make_propofol_pk_figure/make_remifentanil_pk_figure, the same
    functions the Recommendation page's own baseline-only display already
    uses - no orange trace), and the induction-dose rationale graph shows
    only the green Baseline reference (make_pr_induction_dose_rationale_
    figure's explored_ns=None branch).
    """
    case = _pr_case(case_id)
    if case is None:
        return (no_update,) * 12

    grid = case.get("remifentanil_grid", {})
    rates = grid.get("rates_mcgkgmin") or []
    results = grid.get("results") or []
    baseline_key = "baseline_remifentanil" if baseline_choice == "remifentanil" else "baseline_none"
    baseline_dict = case.get(baseline_key)

    if not rates or not results or baseline_dict is None:
        empty = make_empty_figure("No precomputed data for this selection")
        return "-", "-", [], "-", empty, empty, empty, empty, empty, "-", "-", "-"

    baseline_ns = _store_dict_to_namespace(baseline_dict)
    baseline_label = _pr_remi_rate_display(baseline_ns)

    p = case["patient"]
    baseline_map = p["baseline_dap"] + (p["baseline_sap"] - p["baseline_dap"]) / 3.0

    if explore_opioid != "remifentanil":
        bis_fig = make_bis_figure(baseline_ns)
        map_fig = make_map_figure(baseline_ns)
        propofol_fig = make_propofol_pk_figure(baseline_ns)
        remifentanil_fig = make_remifentanil_pk_figure(baseline_ns)
        dose_response = make_pr_induction_dose_rationale_figure(baseline_ns, baseline_map)
        grid_children = _build_pr_result_grid_baseline_only(baseline_ns)
        return (
            baseline_label, "Not selected", grid_children, "-",
            bis_fig, map_fig, propofol_fig, remifentanil_fig, dose_response.figure,
            dose_response.map_zone_text, dose_response.bis_zone_text, dose_response.dose_range_text,
        )

    idx = _pr_rate_to_index(rates, rate)
    explored_dict = results[idx]
    explored_ns = _store_dict_to_namespace(explored_dict)

    bis_fig = make_bis_figure_dual(baseline_ns, explored_ns, manual_name="Explored Scenario")
    map_fig = make_map_figure_dual(baseline_ns, explored_ns, manual_name="Explored Scenario")
    propofol_fig = make_propofol_pk_figure_dual(
        baseline_ns, explored_ns,
        manual_cp_name="Explored Scenario (Cp)", manual_ce_name="Explored Scenario (Ce)",
    )
    remifentanil_fig = make_pr_remifentanil_pk_figure(baseline_ns, explored_ns)
    dose_response = make_pr_induction_dose_rationale_figure(baseline_ns, baseline_map, explored_ns)

    explored_label = _pr_remi_rate_display(explored_ns)
    grid_children = _build_pr_result_grid(baseline_ns, explored_ns)
    rate_label = f"{rates[idx]:.2f} µg/kg/min"

    return (
        baseline_label, explored_label, grid_children, rate_label,
        bis_fig, map_fig, propofol_fig, remifentanil_fig, dose_response.figure,
        dose_response.map_zone_text, dose_response.bis_zone_text, dose_response.dose_range_text,
    )


def main() -> None:
    """
    Main entry point for running the Dash application.
    """
    app.run(debug=False)


if __name__ == "__main__":
    main()
