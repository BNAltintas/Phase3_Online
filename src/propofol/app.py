from __future__ import annotations

import datetime

import dash
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, ctx, html, no_update
from plotly.subplots import make_subplots

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
    recommend_su2023_regimen,
)

# ============================================================
# App setup
# ============================================================

app = dash.Dash(__name__)
app.title = "Su2023 Propofol Dashboard"
app.layout = build_layout()


# ============================================================
# Small helpers
# ============================================================

def _button_class(selected: str, current: str) -> str:
    """
    Return the toggle-pill className for a button, marking it active if selected.
    """
    if selected == current:
        return "toggle-btn toggle-btn--active"
    return "toggle-btn"


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


def make_induction_card(rec):
    """
    Build the induction-dose recommendation card: total dose and
    weight-adjusted dose on the left, model confidence on the right.
    """
    tier = confidence_tier(rec.confidence_percent)
    tier_class = tier.lower()

    return html.Div(
        [
            html.H4("Induction recommendation", className="card-subheading"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Total dose", className="induction-dose-label"),
                            html.Div(
                                f"{rec.propofol_bolus_mg:.0f} mg",
                                className="induction-dose-value",
                            ),
                            html.Div(
                                f"Weight-adjusted: {rec.propofol_bolus_mgkg:.2f} mg/kg",
                                className="induction-dose-subtext",
                            ),
                        ],
                        className="induction-dose-block",
                    ),
                    html.Div(
                        [
                            html.Div("Model confidence", className="induction-confidence-label"),
                            html.Div(
                                f"{rec.confidence_percent:.0f}%",
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
                    ),
                ],
                className="induction-card-body",
            ),
        ],
        className="card induction-card",
    )


def make_maintenance_card(rec):
    """
    Build the maintenance-regimen card: one row per contiguous dosing segment.
    """
    children = [html.H4("Maintenance regimen", className="card-subheading")]

    prop_cells = _maintenance_rows(
        rec.propofol_inf_rates_ml_h, rec.propofol_inf_rates_mcgkgmin, "µg/kg/min",
    )
    children.append(
        html.Div(prop_cells, className="maintenance-table")
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
            html.Div(remi_cells, className="maintenance-table")
            if remi_cells
            else html.Div("No maintenance", className="maintenance-empty")
        )

    return html.Div(children, className="card maintenance-card")


def make_summary(rec):
    """
    Build the recommendation output: an induction-dose card followed by a
    maintenance-regimen card.
    """
    return [
        make_induction_card(rec),
        make_maintenance_card(rec),
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
# Dose-rationale graph
# ============================================================

def _dose_axis_limits_mgkg(selected_dose_mgkg: float) -> tuple[float, float]:
    """
    Show only the clinically relevant local window around the recommended dose:
        selected dose - 1.5 to selected dose + 1.5 mg/kg,
    clipped to max 0.3-5.0 mg/kg.
    """
    selected_dose_mgkg = float(selected_dose_mgkg)
    x_min = max(0.30, selected_dose_mgkg - 1.50)
    x_max = min(5.00, selected_dose_mgkg + 1.50)

    if x_max <= x_min:
        x_min, x_max = 0.30, 5.00

    return float(x_min), float(x_max)


def _clinical_propofol_dose_grid_mgkg(selected_bolus_mgkg: float) -> np.ndarray:
    """
    Build a local propofol induction-dose grid around the selected dose.

    The displayed axis is selected dose ±1.5 mg/kg, clipped to 0.3-5.0 mg/kg.
    A 0.10 mg/kg spacing gives a smooth rationale curve without making the
    dashboard too slow.
    """
    x_min, x_max = _dose_axis_limits_mgkg(selected_bolus_mgkg)

    dose_grid = np.arange(x_min, x_max + 0.001, 0.10)
    dose_grid = np.unique(np.concatenate([dose_grid, [float(selected_bolus_mgkg)]]))
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
):
    """
    Vary only the propofol induction bolus while keeping the recommended
    maintenance regimens fixed.

    X-axis:
        recommended propofol induction dose ±1.5 mg/kg,
        clipped to 0.3-5.0 mg/kg.

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
    """
    selected_dose_mgkg = float(rec.propofol_bolus_mgkg)
    baseline_map = float(patient.base_map)
    map_target = float(rec.map_lower_bound_mmhg)
    map_target_upper = float(1.20 * baseline_map)

    x_axis_min, x_axis_max = _dose_axis_limits_mgkg(selected_dose_mgkg)
    dose_grid_mgkg = _clinical_propofol_dose_grid_mgkg(selected_dose_mgkg)

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
            showlegend=False,
            hovertemplate="Dose %{x:.3f} mg/kg<br>Minimal MAP %{y:.1f} mmHg<extra></extra>",
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
            showlegend=False,
            hovertemplate=(
                "Dose %{x:.3f} mg/kg<br>"
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
                "Selected dose %{x:.3f} mg/kg<br>"
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
                "Selected dose %{x:.3f} mg/kg<br>"
                "Maximal BIS %{y:.1f}<br>"
                "Minimal BIS %{customdata[0]:.1f}<br>"
                f"BIS target {rec.target_bis_low:.0f}-{rec.target_bis_high:.0f}<extra></extra>"
            ),
        ),
        secondary_y=True,
    )

    # Selected recommended induction dose, shown in bold black.
    fig.add_shape(
        type="line",
        xref="x",
        yref="paper",
        x0=selected_dose_mgkg,
        x1=selected_dose_mgkg,
        y0=0,
        y1=1,
        line=dict(color="black", width=3, dash="dash"),
        layer="above",
    )
    fig.add_annotation(
        x=selected_dose_mgkg,
        y=1.07,
        xref="x",
        yref="paper",
        text=f"<b>Selected {selected_dose_mgkg:.3f} mg/kg</b>",
        showarrow=False,
        yanchor="bottom",
        font=dict(color="black", size=13),
    )

    fig.update_layout(
        title="Induction-dose rationale",
        template="plotly_white",
        showlegend=False,
        margin=dict(t=95),
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

    return fig


# ============================================================
# Button callbacks
# ============================================================

@app.callback(
    Output("sex-store", "data"),
    Output("sex-male-btn", "className"),
    Output("sex-female-btn", "className"),
    Input("sex-male-btn", "n_clicks"),
    Input("sex-female-btn", "n_clicks"),
    State("sex-store", "data"),
)
def update_sex(male_clicks, female_clicks, current_value):
    """
    Update the selected sex based on button clicks.
    """
    triggered = ctx.triggered_id
    value = current_value or "male"

    if triggered == "sex-male-btn":
        value = "male"
    elif triggered == "sex-female-btn":
        value = "female"

    return (
        value,
        _button_class(value, "male"),
        _button_class(value, "female"),
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
# Main recommendation callback
# ============================================================

@app.callback(
    Output("summary-output", "children"),
    Output("dose-rationale-graph", "figure"),
    Output("propofol-pk-graph", "figure"),
    Output("remifentanil-pk-graph", "figure"),
    Output("bis-graph", "figure"),
    Output("map-graph", "figure"),
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
    Run the pharmacokinetic model and generate recommendations based on user inputs.
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

        return (
            make_summary(rec),
            make_induction_dose_rationale_figure(
                patient=patient,
                rec=rec,
                opiate=opiate,
                propofol_conc_mg_ml=propofol_concentration,
                remifentanil_conc_mcg_ml=remifentanil_concentration,
            ),
            make_propofol_pk_figure(rec),
            make_remifentanil_pk_figure(rec),
            make_bis_figure(rec),
            make_map_figure(rec),
        )

    except Exception as e:
        error_component = html.Div(
            html.Pre(
                f"Error while running recommendation:\n{str(e)}",
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


def main() -> None:
    """
    Main entry point for running the Dash application.
    """
    app.run(debug=False)


if __name__ == "__main__":
    main()
