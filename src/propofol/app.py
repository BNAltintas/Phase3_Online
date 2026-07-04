from __future__ import annotations

import datetime
from types import SimpleNamespace
from typing import Optional

import dash
import numpy as np
import plotly.graph_objects as go
from dash import ALL, MATCH, Input, Output, State, ctx, dcc, html, no_update
from plotly.subplots import make_subplots

from propofol.config import BOLUS_MGKG_BOUNDS
from propofol.dashboard_layout import (
    EHR_RECORD_DATE,
    EHR_RECORD_TIME,
    build_layout,
    timestamp_display,
)
from propofol.patient import EleveldPatient as Patient
from propofol.recommend_regimen2023 import (
    DEFAULT_PROPOFOL_CONC_MG_ML,
    DEFAULT_REMI_CONC_MCG_ML,
    MAP_ABS_MIN_TARGET,
    MAP_REL_FRAC_TARGET,
    TARGET_ASSESSMENT_START_MIN,
    TARGET_BIS_HIGH,
    TARGET_BIS_LOW,
    DecodedRegimen,
    Su2023PropofolRemifentanilRecommender,
    compress_minute_schedule,
    recommend_maintenance_for_fixed_propofol_bolus,
    recommend_su2023_regimen,
)

# ============================================================
# App setup
# ============================================================

app = dash.Dash(__name__)
app.title = "Su2023 Propofol Dashboard"
app.layout = build_layout()

# The induction card's "Override propofol dose" button/popover only exist in
# the DOM after the first "Run recommendation" (summary-output starts empty).
# Callbacks that target those ids must be allowed to reference ids that are
# not present in the initial layout - this is the standard Dash mechanism
# for that.
app.config.suppress_callback_exceptions = True

MANUAL_OVERRIDE_COLOR = "#c2680f"


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
    if confidence_percent >= 70:
        return "HIGH"
    if confidence_percent >= 40:
        return "MEDIUM"
    return "LOW"


def _maintenance_rows(rate_ml_h, rate_secondary, secondary_label: str) -> list:
    """
    Build alternating interval/rate grid-cell components for one drug's
    compressed maintenance schedule. A near-zero rate is shown as "Pause"
    instead of "0 mL/h (0 x/kg/min)".
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
        cells.append(html.Div(rate_text, className=rate_class))

    return cells


def _override_popover(prefill_value):
    """
    Build the click-to-open "Override propofol dose" popover. Always
    rendered (closed by default) regardless of whether an override is
    currently active, so its component ids stay stable for the callbacks
    that target them.

    The unit toggle (Total dose / mg/kg) always opens defaulted to "Total
    dose" - handle_override resets it every time the popover is opened, so
    prefill_value (always an absolute mg amount) never needs converting for
    display here.
    """
    return html.Div(
        [
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
            dcc.Input(
                id="override-dose-draft",
                type="text",
                value=prefill_value,
                className="field-input",
                placeholder="Manual dose in mg",
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


def make_induction_card(original, manual=None):
    """
    Build the induction-dose recommendation card.

    With no manual override: total dose + weight-adjusted dose on the left,
    model confidence on the right (unchanged from before manual override
    existed).

    With a manual override active: the manual dose is shown large/prominent
    on the left (with the original recommendation noted smaller underneath),
    and a "MANUAL DOSE ACTIVE" status box + "Return to recommendation"
    button replace the confidence block on the right - confidence is never
    computed for the manual dose, so nothing confidence-shaped is shown.

    Both the confidence block and the manual-active block are always
    rendered (only one is ever visible, via inline display:none) rather
    than conditionally included - "Return to recommendation" is a callback
    Input, and Dash logs a console error if a registered callback's Input
    id is ever absent from the current DOM entirely, so its element must
    always exist even while inactive.
    """
    override_button_label = "Edit manual dose" if manual is not None else "Override propofol dose"
    prefill_value = manual.propofol_bolus_mg if manual is not None else original.propofol_bolus_mg
    manual_active = manual is not None

    tier = confidence_tier(original.confidence_percent)
    tier_class = tier.lower()

    original_dose_block = html.Div(
        [
            html.Div("Total dose", className="induction-dose-label"),
            html.Div(
                f"{original.propofol_bolus_mg:.0f} mg",
                className="induction-dose-value",
            ),
            html.Div(
                f"Weight-adjusted: {original.propofol_bolus_mgkg:.2f} mg/kg",
                className="induction-dose-subtext",
            ),
        ],
        className="induction-dose-block",
        style={"display": "none"} if manual_active else None,
    )
    manual_dose_block = html.Div(
        [
            html.Div(
                "Manual total dose",
                className="induction-dose-label induction-dose-label--manual",
            ),
            html.Div(
                f"{manual.propofol_bolus_mg:.0f} mg" if manual_active else "-",
                className="induction-dose-value induction-dose-value--manual",
            ),
            html.Div(
                f"Weight-adjusted: {manual.propofol_bolus_mgkg:.2f} mg/kg" if manual_active else "-",
                className="induction-dose-subtext",
            ),
            html.Div(
                (
                    f"Model recommendation: {original.propofol_bolus_mg:.0f} mg "
                    f"({original.propofol_bolus_mgkg:.2f} mg/kg)"
                ),
                className="induction-dose-original-note",
            ),
        ],
        className="induction-dose-block",
        style=None if manual_active else {"display": "none"},
    )

    confidence_block = html.Div(
        [
            html.Div("Model confidence", className="induction-confidence-label"),
            html.Div(
                f"{original.confidence_percent:.0f}%",
                className=(
                    f"induction-confidence-value "
                    f"induction-confidence-value--{tier_class}"
                ),
            ),
            html.Div(
                f"{tier} CONFIDENCE",
                className=(
                    f"induction-confidence-tier "
                    f"induction-confidence-tier--{tier_class}"
                ),
            ),
        ],
        className="induction-confidence-block",
        style={"display": "none"} if manual_active else None,
    )
    manual_active_block = html.Div(
        [
            html.Div("MANUAL DOSE ACTIVE", className="induction-manual-badge"),
            html.Button(
                "Return to recommendation", id="override-return-btn", n_clicks=0,
                className="override-return-btn",
            ),
        ],
        className="induction-confidence-block",
        style=None if manual_active else {"display": "none"},
    )

    infeasible_warning = html.Div(
        "⚠ This manual dose may not fully reach the target BIS/MAP range.",
        className="induction-infeasible-warning",
        style=(
            None
            if (manual_active and not (manual.feasible_bis and manual.feasible_map))
            else {"display": "none"}
        ),
    )

    card_children = [
        html.H4("Induction recommendation", className="card-subheading"),
        html.Div(
            [original_dose_block, manual_dose_block, confidence_block, manual_active_block],
            className="induction-card-body",
        ),
        infeasible_warning,
    ]

    card_children.append(
        html.Div(
            [
                html.Button(
                    override_button_label, id="override-dose-btn", n_clicks=0,
                    className="override-dose-btn",
                ),
                _override_popover(prefill_value),
            ],
            className="override-dose-section",
        )
    )

    return html.Div(card_children, className="card induction-card")


def _maintenance_section(rec, label: str | None = None, manual: bool = False):
    """
    Build one drug-schedule section (propofol + optional remifentanil) for
    the maintenance card, optionally preceded by a scenario label
    ("Original recommendation" / "Manual dose (re-optimized)").
    """
    children = []
    if label is not None:
        label_class = "maintenance-scenario-label"
        if manual:
            label_class += " maintenance-scenario-label--manual"
        children.append(html.Div(label, className=label_class))

    table_class = "maintenance-table maintenance-table--manual" if manual else "maintenance-table"

    prop_cells = _maintenance_rows(
        rec.propofol_inf_rates_ml_h, rec.propofol_inf_rates_mcgkgmin, "µg/kg/min",
    )
    children.append(
        html.Div(prop_cells, className=table_class)
        if prop_cells
        else html.Div("No maintenance", className="maintenance-empty")
    )

    if rec.remifentanil_selected:
        children.append(
            html.H4("Remifentanil", className="card-subheading maintenance-subheading"),
        )
        remi_cells = _maintenance_rows(
            rec.remifentanil_inf_rates_ml_h, rec.remifentanil_inf_rates_ngkgmin, "ng/kg/min",
        )
        children.append(
            html.Div(remi_cells, className=table_class)
            if remi_cells
            else html.Div("No maintenance", className="maintenance-empty")
        )

    return children


def make_maintenance_card(original, manual=None):
    """
    Build the maintenance-regimen card. With no manual override, this is
    just the original recommendation's schedule (unchanged from before
    manual override existed). With a manual override active, the original
    schedule remains visible (labeled, de-emphasized) with the manually
    re-optimized schedule shown below it.
    """
    children = [html.H4("Maintenance regimen", className="card-subheading")]

    if manual is None:
        children.extend(_maintenance_section(original))
    else:
        children.extend(_maintenance_section(original, label="Original recommendation"))
        children.extend(
            _maintenance_section(manual, label="Manual dose (re-optimized)", manual=True),
        )

    return html.Div(children, className="card maintenance-card")


def make_summary(original, manual=None):
    """
    Build the recommendation output: an induction-dose card followed by a
    maintenance-regimen card. `manual`, when provided, is the re-optimized
    manual-override scenario shown alongside the original recommendation.
    """
    return [
        make_induction_card(original, manual),
        make_maintenance_card(original, manual),
    ]


# ============================================================
# General figure helpers
# ============================================================

def _add_band(fig: go.Figure, x, y_low, y_high, name: str):
    """
    Add a shaded band to a Plotly figure.
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
            opacity=0.20,
            name=name,
            hoverinfo="skip",
        )
    )


def _add_line(fig: go.Figure, x, y, name: str):
    """
    Add a line to a Plotly figure.
    """
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=name,
        )
    )


def make_propofol_pk_figure(rec):
    """
    Create a Plotly figure for propofol pharmacokinetics (PK).
    """
    fig = go.Figure()
    c = rec.confidence

    _add_band(fig, c.time_min, c.cp_propofol_p05, c.cp_propofol_p95, "Cp CI 5–95%")
    _add_line(fig, rec.time_min, rec.cp_propofol, "Cp deterministic")

    _add_band(fig, c.time_min, c.ce_propofol_p05, c.ce_propofol_p95, "Ce CI 5–95%")
    _add_line(fig, rec.time_min, rec.ce_propofol, "Ce deterministic")

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
        return make_empty_figure("Remifentanil PK")

    fig = go.Figure()
    c = rec.confidence

    if c.cp_remifentanil_p05 is not None and c.cp_remifentanil_p95 is not None:
        _add_band(fig, c.time_min, c.cp_remifentanil_p05, c.cp_remifentanil_p95, "Cp CI 5–95%")

    if rec.cp_remifentanil is not None:
        _add_line(fig, rec.time_min, rec.cp_remifentanil, "Cp deterministic")

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

    _add_band(fig, c.time_min, c.bis_p05, c.bis_p95, "BIS CI 5–95%")
    _add_line(fig, rec.time_min, rec.bis, "BIS deterministic")

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
        title="BIS",
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

    _add_band(fig, c.time_min, c.map_p05, c.map_p95, "MAP CI 5–95%")
    _add_line(fig, rec.time_min, rec.map_mmhg, "MAP deterministic")

    fig.add_hline(y=rec.map_lower_bound_mmhg, line_dash="dash", annotation_text="MAP lower bound")

    y_upper = max(160.0, float(np.nanmax(c.map_p95)) + 10.0)

    fig.update_layout(
        title="MAP",
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

def make_propofol_pk_figure_dual(original, manual):
    """
    Overlay the manual-override propofol PK trace (orange) on the original
    recommendation's propofol PK figure (unchanged colors/bands).
    """
    fig = make_propofol_pk_figure(original)

    fig.add_trace(go.Scatter(
        x=manual.time_min, y=manual.cp_propofol, mode="lines",
        line=dict(color=MANUAL_OVERRIDE_COLOR, width=2),
        name="Cp (manual dose)",
    ))
    fig.add_trace(go.Scatter(
        x=manual.time_min, y=manual.ce_propofol, mode="lines",
        line=dict(color=MANUAL_OVERRIDE_COLOR, width=2, dash="dot"),
        name="Ce (manual dose)",
    ))

    return fig


def make_remifentanil_pk_figure_dual(original, manual):
    """
    Overlay the manual-override remifentanil PK trace (orange) on the
    original recommendation's remifentanil PK figure, if remifentanil is
    selected in both. Propofol-only overrides don't change remifentanil's
    own PK, but its maintenance schedule can shift during re-optimization,
    so the two curves can differ.
    """
    if not original.remifentanil_selected:
        return make_remifentanil_pk_figure(original)

    fig = make_remifentanil_pk_figure(original)

    if manual.remifentanil_selected and manual.cp_remifentanil is not None:
        fig.add_trace(go.Scatter(
            x=manual.time_min, y=manual.cp_remifentanil, mode="lines",
            line=dict(color=MANUAL_OVERRIDE_COLOR, width=2),
            name="Cp (manual dose)",
        ))

    return fig


def make_bis_figure_dual(original, manual):
    """
    Overlay the manual-override BIS trace (orange) on the original
    recommendation's BIS figure (unchanged colors/bands/target lines).
    """
    fig = make_bis_figure(original)

    fig.add_trace(go.Scatter(
        x=manual.time_min, y=manual.bis, mode="lines",
        line=dict(color=MANUAL_OVERRIDE_COLOR, width=2),
        name="BIS (manual dose)",
    ))

    return fig


def make_map_figure_dual(original, manual):
    """
    Overlay the manual-override MAP trace (orange) on the original
    recommendation's MAP figure (unchanged colors/bands/target line).
    """
    fig = make_map_figure(original)

    fig.add_trace(go.Scatter(
        x=manual.time_min, y=manual.map_mmhg, mode="lines",
        line=dict(color=MANUAL_OVERRIDE_COLOR, width=2),
        name="MAP (manual dose)",
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
                None
                if rec.remifentanil_inf_rates_mcgkgmin is None
                else np.asarray(rec.remifentanil_inf_rates_mcgkgmin, dtype=float).copy()
            ),
            remifentanil_rates_ml_h=(
                None
                if rec.remifentanil_inf_rates_ml_h is None
                else np.asarray(rec.remifentanil_inf_rates_ml_h, dtype=float).copy()
            ),
            remifentanil_rates_ngkgmin=(
                None
                if rec.remifentanil_inf_rates_ngkgmin is None
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

        # Lower and upper target-dose limits.
        for x_value, label, x_anchor in [
            (ok_start, f"{ok_start:.2f}", "right"),
            (ok_end, f"{ok_end:.2f}", "left"),
        ]:
            fig.add_annotation(
                x=x_value,
                y=1.01,
                xref="x",
                yref="paper",
                text=f"<b>{label}</b>",
                showarrow=False,
                xanchor=x_anchor,
                yanchor="bottom",
                font=dict(color="green", size=13),
            )

    fig.add_trace(
        go.Scatter(
            x=dose_grid_mgkg,
            y=min_maps,
            mode="lines+markers",
            line=dict(color="red", width=3),
            marker=dict(color="red", size=6),
            name="MAP",
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
            name="BIS",
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

    # Legend-only entries for the shapes below (target band, recommendation
    # line, manual-override line) - Plotly shapes never appear in the
    # legend on their own, so a zero-data dummy trace styled to match is
    # the standard way to add one. Purely presentational: none of these
    # affect the plotted data.
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(symbol="square", size=12, color="rgba(0, 150, 0, 0.25)"),
            name="Target zone",
            showlegend=True,
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(color="green", width=3, dash="dash"),
            name="Recommendation",
            showlegend=True,
        ),
    )
    if manual_dose_mgkg is not None:
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="lines",
                line=dict(color=MANUAL_OVERRIDE_COLOR, width=3, dash="dash"),
                name="Manual override",
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
    fig.add_annotation(
        x=selected_dose_mgkg,
        y=1.07,
        xref="x",
        yref="paper",
        text=f"<b>Recommendation {selected_dose_mgkg:.2f} mg/kg</b>",
        showarrow=False,
        yanchor="bottom",
        font=dict(color="green", size=13),
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
        fig.add_annotation(
            x=manual_dose_mgkg,
            y=1.14,
            xref="x",
            yref="paper",
            text=f"<b>Manual dose {manual_dose_mgkg:.2f} mg/kg</b>",
            showarrow=False,
            yanchor="bottom",
            font=dict(color=MANUAL_OVERRIDE_COLOR, size=13),
        )

    fig.update_layout(
        # No in-plot title text - the card header ("Induction-dose rationale")
        # already shows it in black, so a second grey title here was a
        # redundant duplicate.
        template="plotly_white",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        margin=dict(
            t=115 if manual_dose_mgkg is not None else 95,
            b=90,
            l=100,
            r=90,
        ),
        xaxis=dict(
            title=dict(text="Propofol induction dose (mg/kg)", font=dict(color="green")),
            tickfont=dict(color="green"),
            color="green",
            range=[x_axis_min, x_axis_max],
        ),
        yaxis=dict(
            title=dict(text="Minimal MAP (mmHg)", font=dict(color="red")),
            tickfont=dict(color="red"),
            color="red",
            range=[map_axis_min, map_axis_max],
        ),
        yaxis2=dict(
            title=dict(text="Maximal BIS", font=dict(color="blue")),
            tickfont=dict(color="blue"),
            color="blue",
            range=[max_bis_axis_max, max_bis_axis_min],
            overlaying="y",
            side="right",
        ),
    )

    # Directional arrows next to each y-axis, purely to help interpret which
    # way is "more" on each axis - neither changes any plotted data. ax/ay
    # are pixel offsets for the arrow tail relative to the (x, y) head
    # position (Plotly's axref/ayref only support "pixel" or another axis's
    # domain, not "paper", so the head is anchored in paper coordinates and
    # the tail is just an offset from it).
    # Left axis (MAP): points up, since higher on this axis is higher MAP.
    fig.add_annotation(
        xref="paper", yref="paper",
        x=-0.20, y=0.55,
        ax=0, ay=40,
        showarrow=True,
        arrowhead=2,
        arrowsize=1.3,
        arrowwidth=3,
        arrowcolor="red",
        text="",
    )
    # Right axis (BIS): points down, since this axis is visually inverted -
    # lower on the page is a higher BIS value.
    fig.add_annotation(
        xref="paper", yref="paper",
        x=1.16, y=0.45,
        ax=0, ay=-40,
        showarrow=True,
        arrowhead=2,
        arrowsize=1.3,
        arrowwidth=3,
        arrowcolor="blue",
        text="",
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
# Page navigation (Recommendation <-> More Info)
#
# The two pages are always-mounted siblings (built once in build_layout);
# switching between them only ever toggles which one's `style` is
# display:none, exactly like every other show/hide toggle in this app. No
# page carries any model state, so this cannot affect the recommendation
# pipeline, the manual-override pipeline, or any stored data.
# ============================================================

@app.callback(
    Output("active-page-store", "data"),
    Input("nav-recommendation-btn", "n_clicks"),
    Input("nav-more-info-btn", "n_clicks"),
    Input("back-to-recommendation-btn", "n_clicks"),
    Input("learn-more-link-btn", "n_clicks"),
    prevent_initial_call=True,
)
def set_active_page(rec_nav_clicks, more_info_nav_clicks, back_clicks, learn_more_clicks):
    """
    Track which top-level page is active based on which nav/back control was
    clicked. learn-more-link-btn (the dashboard's own "Learn more about the
    model" link) is just a second entry point to the same "more-info" page
    the sidebar's "More Info" item already switches to.
    """
    triggered = ctx.triggered_id
    if triggered in ("nav-more-info-btn", "learn-more-link-btn"):
        return "more-info"
    if triggered in ("nav-recommendation-btn", "back-to-recommendation-btn"):
        return "recommendation"
    return no_update


@app.callback(
    Output("main-content", "style"),
    Output("more-info-view", "style"),
    Output("nav-recommendation-btn", "className"),
    Output("nav-more-info-btn", "className"),
    Input("active-page-store", "data"),
)
def render_active_page(active_page):
    """
    Show exactly one of the two pages, and keep the sidebar's active
    highlight in sync with it.
    """
    is_recommendation = active_page != "more-info"
    hidden = {"display": "none"}

    return (
        None if is_recommendation else hidden,
        hidden if is_recommendation else None,
        "sidebar-nav-item sidebar-nav-item--active" if is_recommendation else "sidebar-nav-item",
        "sidebar-nav-item sidebar-nav-item--active" if not is_recommendation else "sidebar-nav-item",
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

EDITABLE_FIELDS = [
    ("age", 35, "EHR"),
    ("height", 170, "EHR"),
    ("weight", 70, "EHR"),
    ("baseline_sap", 120, "Monitor"),
    ("baseline_dap", 70, "Monitor"),
    ("baseline_hr", 70, "Monitor"),
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


def _register_editable_field_callback(field_id: str, original_value: float, source_label: str):
    """
    Register the Save/Cancel/Restore popover callback for one patient-parameter field.
    """
    cross_field = FIELD_RULES[field_id].get("cross_check_field")

    outputs = [
        Output(field_id, "value"),
        Output(f"{field_id}-popover", "className"),
        Output(f"{field_id}-draft", "value"),
        Output(f"{field_id}-source", "children"),
        Output(f"{field_id}-source", "className"),
        Output(f"{field_id}-date", "children"),
        Output(f"{field_id}-restore-btn", "className"),
    ]
    inputs = [
        Input(f"{field_id}-badge", "n_clicks"),
        Input(f"{field_id}-save-btn", "n_clicks"),
        Input(f"{field_id}-cancel-btn", "n_clicks"),
        Input(f"{field_id}-restore-btn", "n_clicks"),
        Input(f"{field_id}-draft", "n_submit"),
    ]
    states = [State(f"{field_id}-draft", "value"), State(field_id, "value")]
    if cross_field:
        states.append(State(cross_field, "value"))

    def _update_field(badge_clicks, save_clicks, cancel_clicks, restore_clicks, draft_submit,
                       *extra_states):
        """
        Open/close the edit popover and apply Save/Cancel/Restore for this field.
        """
        draft_value, current_value = extra_states[0], extra_states[1]
        other_value = extra_states[2] if cross_field else None
        triggered = ctx.triggered_id

        if triggered == f"{field_id}-restore-btn":
            return _field_result(original_value, original_value, source_label)

        if triggered in (f"{field_id}-save-btn", f"{field_id}-draft"):
            status, _ = _validate_field_value(field_id, draft_value, other_value)
            if status == "error":
                # Invalid value: refuse to commit, leave the popover open as-is.
                # The live-validation callback already shows the error inline.
                return (no_update,) * 7
            return _field_result(float(draft_value), original_value, source_label)

        if triggered == f"{field_id}-badge":
            return _field_result(current_value, original_value, source_label, popover_open=True)

        # Cancel (or any other trigger): close the popover without changing the saved value.
        return _field_result(current_value, original_value, source_label)

    app.callback(*outputs, *inputs, *states, prevent_initial_call=True)(_update_field)


for _field_id, _original_value, _source_label in EDITABLE_FIELDS:
    _register_editable_field_callback(_field_id, _original_value, _source_label)


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
    Output("recommendation-stale-store", "data"),
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
        return store_data, None, False

    except Exception as e:
        return {"error": str(e), "context": None, "result": None}, None, False


@app.callback(
    Output("summary-output", "children"),
    Output("dose-rationale-graph", "figure"),
    Output("propofol-pk-graph", "figure"),
    Output("remifentanil-pk-graph", "figure"),
    Output("bis-graph", "figure"),
    Output("map-graph", "figure"),
    Input("recommendation-store", "data"),
    Input("manual-scenario-store", "data"),
    prevent_initial_call=True,
)
def render_recommendation(rec_store, manual_store):
    """
    Render the recommendation card(s) and the 5 prediction graphs from
    recommendation-store (the original recommendation, never overwritten)
    and manual-scenario-store (the active manual override, if any). This is
    the single owner of these outputs - it fires both when a new
    recommendation is run and when the manual-override state changes, so
    there is exactly one rendering code path for both cases.
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

    summary = make_summary(original, manual)

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
# Before/after-run state
#
# Purely presentational: toggles which of {empty placeholder, stale
# placeholder, real content} is visible for the recommendation, rationale,
# and predictions sections. It never touches figure/card content itself
# (render_recommendation, above, still owns that), so hidden content stays
# in the DOM - already-computed but out-of-date graphs are just not shown,
# rather than being cleared and recomputed.
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

    Opening the popover always resets the unit toggle to "Total dose" and
    clears any leftover validation message/border from a previous edit, so
    each edit starts clean regardless of how the last one ended.
    """
    triggered = ctx.triggered_id
    clean = ("field-input", "", "field-message")

    if triggered == "override-return-btn":
        return "edit-popover", no_update, no_update, no_update, no_update, no_update, None

    if not rec_store or rec_store.get("error") or not rec_store.get("context"):
        return (no_update,) * 7

    context = rec_store["context"]

    if triggered == "override-dose-btn":
        prefill = (
            manual_store["dose_mg"]
            if manual_store
            else rec_store["result"]["propofol_bolus_mg"]
        )
        return ("edit-popover edit-popover--open", prefill, *clean, "total", no_update)

    if triggered == "override-cancel-btn":
        return ("edit-popover", no_update, *clean, no_update, no_update)

    if triggered in ("override-save-btn", "override-dose-draft"):
        weight_kg = float(context["weight"])
        status, _, manual_dose_mg = _validate_override_dose(draft_value, weight_kg, unit or "total")
        if status == "error":
            # Invalid value: refuse to commit, leave the popover open.
            # validate_override_draft (same trigger) shows the error inline.
            return (no_update,) * 7

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
            return (no_update,) * 7

        manual_store_data = {
            "dose_mg": manual_dose_mg,
            "result": _rec_to_store_dict(manual_rec),
        }
        return ("edit-popover", no_update, no_update, no_update, no_update, no_update, manual_store_data)

    # Any other trigger: close the popover without changing anything.
    return ("edit-popover", no_update, no_update, no_update, no_update, no_update, no_update)


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


def main() -> None:
    """
    Main entry point for running the Dash application.
    """
    app.run(debug=False)


if __name__ == "__main__":
    main()
