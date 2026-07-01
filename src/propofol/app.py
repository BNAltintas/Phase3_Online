from __future__ import annotations

import dash
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, ctx, html
from plotly.subplots import make_subplots

from propofol.dashboard_layout import build_layout
from propofol.patient import EleveldPatient as Patient
from propofol.recommend_regimen2023 import (
    DecodedRegimen,
    TARGET_ASSESSMENT_START_MIN,
    TARGET_BIS_HIGH,
    TARGET_BIS_LOW,
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

def _button_styles(selected: str, current: str) -> dict:
    if selected == current:
        return {
            "padding": "10px 16px",
            "border": "1px solid #1f77b4",
            "backgroundColor": "#1f77b4",
            "color": "white",
            "cursor": "pointer",
            "marginRight": "8px",
            "borderRadius": "8px",
            "fontWeight": "600",
        }

    return {
        "padding": "10px 16px",
        "border": "1px solid #bbb",
        "backgroundColor": "#f7f7f7",
        "color": "black",
        "cursor": "pointer",
        "marginRight": "8px",
        "borderRadius": "8px",
        "fontWeight": "600",
    }


def compute_map(sap: float, dap: float) -> float:
    return (sap + 2.0 * dap) / 3.0


def compute_pp(sap: float, dap: float) -> float:
    return sap - dap


def parse_concentration(selection, custom_value, name: str) -> float:
    if selection == "custom":
        if custom_value is None or custom_value == "":
            raise ValueError(f"Please provide a custom {name} concentration.")
        value = float(custom_value)
    else:
        value = float(selection)

    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} concentration must be positive.")

    return value


def make_empty_figure(title: str | None = None):
    fig = go.Figure()
    fig.update_layout(template="plotly_white", title=title or None)
    return fig


def _safe_array(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


# ============================================================
# Schedule formatting
# ============================================================

def format_propofol_schedule(rates_ml_h, rates_mcgkgmin) -> list[str]:
    ml_rows = compress_minute_schedule(rates_ml_h)
    mcg_rows = compress_minute_schedule(rates_mcgkgmin)

    if len(ml_rows) == 0:
        return ["No maintenance"]

    lines = []
    for ml_row, mcg_row in zip(ml_rows, mcg_rows):
        lines.append(
            f"{ml_row['start_min']:.0f}–{ml_row['end_min']:.0f} min: "
            f"{ml_row['rate']:.0f} mL/h ({mcg_row['rate']:.0f} µg/kg/min)"
        )

    return lines


def format_remifentanil_schedule(rates_ml_h, rates_ngkgmin) -> list[str]:
    ml_rows = compress_minute_schedule(rates_ml_h)
    ng_rows = compress_minute_schedule(rates_ngkgmin)

    if len(ml_rows) == 0:
        return ["No maintenance"]

    lines = []
    for ml_row, ng_row in zip(ml_rows, ng_rows):
        lines.append(
            f"{ml_row['start_min']:.0f}–{ml_row['end_min']:.0f} min: "
            f"{ml_row['rate']:.0f} mL/h ({ng_row['rate']:.0f} ng/kg/min)"
        )

    return lines


def make_summary(rec):
    lines = [
        f"Propofol induction dose: {rec.propofol_bolus_mg:.0f} mg "
        f"({rec.propofol_bolus_mgkg:.3f} mg/kg)",
        "Propofol maintenance regimen:",
    ]

    lines.extend(
        f"  {line}"
        for line in format_propofol_schedule(
            rec.propofol_infusion_rates_ml_h,
            rec.propofol_infusion_rates_mcgkgmin,
        )
    )

    if rec.remifentanil_selected:
        lines.extend(
            [
                "Remifentanil maintenance regimen:",
            ]
        )
        lines.extend(
            f"  {line}"
            for line in format_remifentanil_schedule(
                rec.remifentanil_infusion_rates_ml_h,
                rec.remifentanil_infusion_rates_ngkgmin,
            )
        )

    lines.append(f"Model confidence: {rec.confidence_percent:.1f}%")

    return html.Pre(
        "\n".join(lines),
        style={
            "whiteSpace": "pre-wrap",
            "fontFamily": "monospace",
            "fontSize": "14px",
            "lineHeight": "1.4",
            "margin": 0,
        },
    )


# ============================================================
# General figure helpers
# ============================================================

def _add_band(fig: go.Figure, x, y_low, y_high, name: str):
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
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name=name,
        )
    )


def make_propofol_pk_figure(rec):
    fig = go.Figure()
    c = rec.confidence

    _add_band(fig, c.time_min, c.cp_propofol_p05, c.cp_propofol_p95, "Cp CI 5–95%")
    _add_line(fig, rec.time_min, rec.cp_propofol, "Cp deterministic")

    _add_band(fig, c.time_min, c.ce_propofol_p05, c.ce_propofol_p95, "Ce CI 5–95%")
    _add_line(fig, rec.time_min, rec.ce_propofol, "Ce deterministic")

    fig.update_layout(
        title="Propofol PK",
        xaxis_title="Time (min)",
        yaxis_title="Propofol concentration (mcg/mL)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    return fig


def make_remifentanil_pk_figure(rec):
    if not rec.remifentanil_selected:
        return make_empty_figure("Remifentanil PK")

    fig = go.Figure()
    c = rec.confidence

    if c.cp_remifentanil_p05 is not None and c.cp_remifentanil_p95 is not None:
        _add_band(fig, c.time_min, c.cp_remifentanil_p05, c.cp_remifentanil_p95, "Cp CI 5–95%")

    if rec.cp_remifentanil is not None:
        _add_line(fig, rec.time_min, rec.cp_remifentanil, "Cp deterministic")

    fig.update_layout(
        title="Remifentanil PK",
        xaxis_title="Time (min)",
        yaxis_title="Remifentanil concentration (ng/mL)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    return fig


def make_bis_figure(rec):
    fig = go.Figure()
    c = rec.confidence

    _add_band(fig, c.time_min, c.bis_p05, c.bis_p95, "BIS CI 5–95%")
    _add_line(fig, rec.time_min, rec.bis, "BIS deterministic")

    fig.add_hline(y=TARGET_BIS_LOW, line_dash="dash", annotation_text="BIS 40")
    fig.add_hline(y=TARGET_BIS_HIGH, line_dash="dash", annotation_text="BIS 60")

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
    clipped to 0.3-5.0 mg/kg.
    """
    selected_dose_mgkg = float(selected_dose_mgkg)
    x_min = max(0.30, selected_dose_mgkg - 1.50)
    x_max = min(5.00, selected_dose_mgkg + 1.50)

    # Avoid a collapsed axis in unusual edge cases.
    if x_max <= x_min:
        x_min, x_max = 0.30, 5.00

    return float(x_min), float(x_max)


def _clinical_propofol_dose_grid_mgkg(
    selected_bolus_mgkg: float,
) -> np.ndarray:
    """
    Build a dose grid only around the selected dose.

    The axis range is selected dose ±1.5 mg/kg, clipped to 0.3-5.0 mg/kg.
    A 0.10 mg/kg grid keeps the graph reasonably smooth without making the
    dashboard too slow.
    """
    x_min, x_max = _dose_axis_limits_mgkg(selected_bolus_mgkg)

    dose_grid = np.arange(x_min, x_max + 0.001, 0.10)
    dose_grid = np.unique(np.concatenate([dose_grid, [float(selected_bolus_mgkg)]]))
    dose_grid = dose_grid[(dose_grid >= x_min) & (dose_grid <= x_max)]
    dose_grid.sort()

    return dose_grid


def _finite_min(values, default: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.min(arr)) if len(arr) else float(default)


def _finite_max(values, default: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.max(arr)) if len(arr) else float(default)


def _round_axis_lower(x: float, step: float = 5.0) -> float:
    return float(step * np.floor(float(x) / step))


def _round_axis_upper(x: float, step: float = 5.0) -> float:
    return float(step * np.ceil(float(x) / step))


def _axis_ranges_crossing_selected(
    selected_map: float,
    selected_bis: float,
    map_values: np.ndarray,
    bis_values: np.ndarray,
    map_target_low: float,
    map_target_high: float,
) -> tuple[float, float, float, float]:
    """
    Compute dynamic MAP and BIS axis ranges that make the MAP and BIS selected
    dose points visually cross.

    Returned values:
        map_axis_min, map_axis_max, bis_axis_bottom, bis_axis_top

    Constraints:
        - MAP axis is not allowed below 0.
        - BIS axis is reversed and not allowed below 0 or above 100.
        - Both axes include their displayed data and target bands when feasible.
    """
    selected_map = float(selected_map)
    selected_bis = float(selected_bis)

    map_values = np.asarray(map_values, dtype=float)
    bis_values = np.asarray(bis_values, dtype=float)

    # Required visible ranges.
    map_low_req = max(
        0.0,
        min(
            _finite_min(map_values, default=selected_map),
            map_target_low,
            selected_map,
        ) - 5.0,
    )
    map_high_req = max(
        _finite_max(map_values, default=selected_map),
        map_target_high,
        selected_map,
    ) + 5.0

    bis_low_req = max(
        0.0,
        min(
            _finite_min(bis_values, default=selected_bis),
            TARGET_BIS_LOW,
            selected_bis,
        ) - 5.0,
    )
    bis_high_req = min(
        100.0,
        max(
            _finite_max(bis_values, default=selected_bis),
            TARGET_BIS_HIGH,
            selected_bis,
        ) + 5.0,
    )

    if (
        not np.isfinite(selected_map)
        or not np.isfinite(selected_bis)
        or selected_map < 0
        or selected_bis < 0
        or selected_bis > 100
    ):
        map_axis_min = max(0.0, _round_axis_lower(map_low_req, step=5.0))
        map_axis_max = _round_axis_upper(max(map_high_req, map_axis_min + 10.0), step=5.0)
        return map_axis_min, map_axis_max, 100.0, 0.0

    # Search over possible vertical crossing fractions. Fraction is measured
    # from the bottom of the plot area. The same fraction is used for the
    # selected MAP and selected BIS points, so the curves cross at the selected
    # dose.
    best = None

    for f in np.linspace(0.10, 0.90, 161):
        f = float(f)

        # BIS axis is reversed:
        #   bottom = selected_bis + f * span
        #   top    = selected_bis - (1 - f) * span
        # Need: bottom >= bis_high_req, top <= bis_low_req,
        # while bottom <= 100 and top >= 0.
        bis_span_min = max(
            10.0,
            (bis_high_req - selected_bis) / max(f, 1e-6),
            (selected_bis - bis_low_req) / max(1.0 - f, 1e-6),
        )
        bis_span_max = min(
            (100.0 - selected_bis) / max(f, 1e-6),
            selected_bis / max(1.0 - f, 1e-6),
        )

        if bis_span_min > bis_span_max:
            continue

        # MAP axis:
        #   min = selected_map - f * span
        #   max = selected_map + (1 - f) * span
        # Need: min <= map_low_req, max >= map_high_req, min >= 0.
        map_span_min = max(
            10.0,
            (selected_map - map_low_req) / max(f, 1e-6),
            (map_high_req - selected_map) / max(1.0 - f, 1e-6),
        )
        map_span_max = selected_map / max(f, 1e-6) if selected_map >= 0 else -np.inf

        if map_span_min > map_span_max:
            continue

        # Use the smallest feasible spans for a zoomed-in graph.
        bis_span = min(bis_span_max, bis_span_min * 1.02)
        map_span = min(map_span_max, map_span_min * 1.02)

        bis_bottom = selected_bis + f * bis_span
        bis_top = selected_bis - (1.0 - f) * bis_span
        map_min = selected_map - f * map_span
        map_max = selected_map + (1.0 - f) * map_span

        # Round axes outward while respecting constraints.
        bis_bottom = min(100.0, _round_axis_upper(bis_bottom, step=5.0))
        bis_top = max(0.0, _round_axis_lower(bis_top, step=5.0))
        map_min = max(0.0, _round_axis_lower(map_min, step=5.0))
        map_max = _round_axis_upper(map_max, step=5.0)

        if map_max <= map_min or bis_bottom <= bis_top:
            continue

        # Recalculate actual crossing fractions after rounding.
        map_f = (selected_map - map_min) / (map_max - map_min)
        bis_f = (bis_bottom - selected_bis) / (bis_bottom - bis_top)
        crossing_error = abs(map_f - bis_f)

        # Prefer exact crossing, then tighter zoom, then less extreme placement.
        score = (
            10_000.0 * crossing_error
            + 0.01 * (map_max - map_min)
            + 0.05 * (bis_bottom - bis_top)
            + abs(f - 0.50)
        )

        if best is None or score < best[0]:
            best = (score, map_min, map_max, bis_bottom, bis_top)

    if best is not None:
        _, map_min, map_max, bis_bottom, bis_top = best
        return float(map_min), float(map_max), float(bis_bottom), float(bis_top)

    # Fallback: zoom both axes independently, respecting constraints. This path
    # should be rare, but prevents crashes for extreme model outputs.
    map_axis_min = max(0.0, _round_axis_lower(map_low_req, step=5.0))
    map_axis_max = _round_axis_upper(max(map_high_req, map_axis_min + 10.0), step=5.0)

    bis_axis_bottom = min(100.0, _round_axis_upper(bis_high_req, step=5.0))
    bis_axis_top = max(0.0, _round_axis_lower(bis_low_req, step=5.0))
    if bis_axis_bottom <= bis_axis_top:
        bis_axis_bottom, bis_axis_top = 100.0, 0.0

    return float(map_axis_min), float(map_axis_max), float(bis_axis_bottom), float(bis_axis_top)


def make_induction_dose_rationale_figure(
    patient: Patient,
    rec,
    opiate: str,
    propofol_concentration_mg_ml: float,
    remifentanil_concentration_mcg_ml: float,
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
        minimal BIS after target start, reversed and constrained to 0-100.

    The y-axes are dynamically zoomed so the MAP and BIS lines cross at the
    recommended induction dose.
    """
    selected_dose_mgkg = float(rec.propofol_bolus_mgkg)
    baseline_map = float(patient.base_map)
    map_target = float(rec.map_lower_bound_mmhg)
    map_target_upper = float(1.20 * baseline_map)

    x_axis_min, x_axis_max = _dose_axis_limits_mgkg(selected_dose_mgkg)
    dose_grid_mgkg = _clinical_propofol_dose_grid_mgkg(selected_dose_mgkg)

    runner = Su2023PropofolRemifentanilRecommender(
        patient=patient,
        use_remifentanil=(opiate == "remifentanil"),
        use_bsv=False,
        propofol_concentration_mg_ml=propofol_concentration_mg_ml,
        remifentanil_concentration_mcg_ml=remifentanil_concentration_mcg_ml,
    )

    min_maps = []
    min_bis_values = []

    for dose_mgkg in dose_grid_mgkg:
        regimen = DecodedRegimen(
            propofol_bolus_mg=float(dose_mgkg * patient.weight),
            propofol_bolus_mgkg=float(dose_mgkg),
            propofol_rates_mgkgh=np.asarray(rec.propofol_infusion_rates_mgkgh, dtype=float).copy(),
            propofol_rates_ml_h=np.asarray(rec.propofol_infusion_rates_ml_h, dtype=float).copy(),
            propofol_rates_mcgkgmin=np.asarray(rec.propofol_infusion_rates_mcgkgmin, dtype=float).copy(),
            remifentanil_selected=bool(rec.remifentanil_selected),
            remifentanil_bolus_mcg=float(rec.remifentanil_bolus_mcg),
            remifentanil_bolus_mcgkg=float(rec.remifentanil_bolus_mcgkg),
            remifentanil_rates_mcgkgmin=(
                None
                if rec.remifentanil_infusion_rates_mcgkgmin is None
                else np.asarray(rec.remifentanil_infusion_rates_mcgkgmin, dtype=float).copy()
            ),
            remifentanil_rates_ml_h=(
                None
                if rec.remifentanil_infusion_rates_ml_h is None
                else np.asarray(rec.remifentanil_infusion_rates_ml_h, dtype=float).copy()
            ),
            remifentanil_rates_ngkgmin=(
                None
                if rec.remifentanil_infusion_rates_ngkgmin is None
                else np.asarray(rec.remifentanil_infusion_rates_ngkgmin, dtype=float).copy()
            ),
        )

        try:
            t, _, _, _, bis, map_mmhg = runner.simulate_regimen(regimen)
            mask = np.asarray(t, dtype=float) >= TARGET_ASSESSMENT_START_MIN

            if not np.any(mask):
                min_maps.append(np.nan)
                min_bis_values.append(np.nan)
                continue

            min_maps.append(float(np.nanmin(np.asarray(map_mmhg, dtype=float)[mask])))
            min_bis_values.append(float(np.nanmin(np.asarray(bis, dtype=float)[mask])))

        except Exception:
            min_maps.append(np.nan)
            min_bis_values.append(np.nan)

    min_maps = np.asarray(min_maps, dtype=float)
    min_bis_values = np.asarray(min_bis_values, dtype=float)

    selected_idx = int(np.argmin(np.abs(dose_grid_mgkg - selected_dose_mgkg)))
    selected_min_map = float(min_maps[selected_idx])
    selected_min_bis = float(min_bis_values[selected_idx])

    (
        map_axis_min,
        map_axis_max,
        bis_axis_bottom,
        bis_axis_top,
    ) = _axis_ranges_crossing_selected(
        selected_map=selected_min_map,
        selected_bis=selected_min_bis,
        map_values=min_maps,
        bis_values=min_bis_values,
        map_target_low=map_target,
        map_target_high=map_target_upper,
    )

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # MAP target band: lower target to 20% above baseline.
    fig.add_shape(
        type="rect",
        xref="paper",
        x0=0,
        x1=1,
        yref="y",
        y0=map_target,
        y1=map_target_upper,
        fillcolor="rgba(220, 0, 0, 0.10)",
        line=dict(width=0),
        layer="below",
    )

    # BIS target band.
    fig.add_shape(
        type="rect",
        xref="paper",
        x0=0,
        x1=1,
        yref="y2",
        y0=TARGET_BIS_LOW,
        y1=TARGET_BIS_HIGH,
        fillcolor="rgba(0, 85, 220, 0.10)",
        line=dict(width=0),
        layer="below",
    )

    # Warning background below MAP target.
    if map_axis_min < map_target:
        fig.add_shape(
            type="rect",
            xref="paper",
            x0=0,
            x1=1,
            yref="y",
            y0=map_axis_min,
            y1=map_target,
            fillcolor="rgba(220, 0, 0, 0.035)",
            line=dict(width=0),
            layer="below",
        )

    # Warning background for BIS >60 on the reversed right axis.
    if bis_axis_bottom > TARGET_BIS_HIGH:
        fig.add_shape(
            type="rect",
            xref="paper",
            x0=0,
            x1=1,
            yref="y2",
            y0=TARGET_BIS_HIGH,
            y1=bis_axis_bottom,
            fillcolor="rgba(0, 85, 220, 0.035)",
            line=dict(width=0),
            layer="below",
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
            y=min_bis_values,
            mode="lines+markers",
            line=dict(color="blue", width=3),
            marker=dict(color="blue", size=6),
            showlegend=False,
            hovertemplate="Dose %{x:.3f} mg/kg<br>Minimal BIS %{y:.1f}<extra></extra>",
        ),
        secondary_y=True,
    )

    # Range of doses where both minimal MAP and minimal BIS are within target.
    both_targets_ok = (
        np.isfinite(min_maps)
        & np.isfinite(min_bis_values)
        & (min_maps >= map_target)
        & (min_maps <= map_target_upper)
        & (min_bis_values >= TARGET_BIS_LOW)
        & (min_bis_values <= TARGET_BIS_HIGH)
    )

    if np.any(both_targets_ok):
        ok_doses = dose_grid_mgkg[both_targets_ok]
        ok_start = float(np.min(ok_doses))
        ok_end = float(np.max(ok_doses))

        fig.add_vline(
            x=ok_start,
            line_dash="dot",
            line_color="green",
            annotation_text=f"{ok_start:.2f}",
            annotation_position="top left",
        )
        fig.add_vline(
            x=ok_end,
            line_dash="dot",
            line_color="green",
            annotation_text=f"{ok_end:.2f}",
            annotation_position="top right",
        )
        if ok_end > ok_start:
            fig.add_vrect(
                x0=ok_start,
                x1=ok_end,
                fillcolor="rgba(0, 150, 0, 0.05)",
                line_width=0,
                layer="below",
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
            y=[selected_min_bis],
            mode="markers",
            marker=dict(color="blue", size=14, symbol="diamond"),
            showlegend=False,
            hovertemplate=(
                "Selected dose %{x:.3f} mg/kg<br>"
                "Minimal BIS %{y:.1f}<br>"
                f"BIS target {TARGET_BIS_LOW:.0f}-{TARGET_BIS_HIGH:.0f}<extra></extra>"
            ),
        ),
        secondary_y=True,
    )

    fig.add_vline(
        x=selected_dose_mgkg,
        line_dash="dash",
        line_color="black",
        annotation_text=f"Selected {selected_dose_mgkg:.3f} mg/kg",
        annotation_position="top",
    )

    fig.update_layout(
        title="Induction-dose rationale",
        template="plotly_white",
        showlegend=False,
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
            title=dict(text="Minimal BIS", font=dict(color="blue")),
            tickfont=dict(color="blue"),
            color="blue",
            range=[bis_axis_bottom, bis_axis_top],
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
    Output("sex-male-btn", "style"),
    Output("sex-female-btn", "style"),
    Input("sex-male-btn", "n_clicks"),
    Input("sex-female-btn", "n_clicks"),
    State("sex-store", "data"),
)
def update_sex(male_clicks, female_clicks, current_value):
    triggered = ctx.triggered_id
    value = current_value or "male"

    if triggered == "sex-male-btn":
        value = "male"
    elif triggered == "sex-female-btn":
        value = "female"

    return (
        value,
        _button_styles(value, "male"),
        _button_styles(value, "female"),
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
    State("propofol-concentration-dropdown", "value"),
    State("propofol-concentration-custom", "value"),
    State("remifentanil-concentration-dropdown", "value"),
    State("remifentanil-concentration-custom", "value"),
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
    propofol_concentration_selection,
    propofol_concentration_custom,
    remifentanil_concentration_selection,
    remifentanil_concentration_custom,
):
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
            raise ValueError("Baseline SAP and DAP must be positive.")
        if baseline_sap <= baseline_dap:
            raise ValueError("Baseline SAP must be higher than DAP.")
        if baseline_hr <= 0:
            raise ValueError("Baseline HR must be positive.")

        opiate = opiate or "none"
        if opiate in {"sufentanil", "fentanyl"}:
            raise ValueError("Only remifentanil is currently supported.")

        propofol_concentration = parse_concentration(
            propofol_concentration_selection,
            propofol_concentration_custom,
            name="propofol",
        )
        remifentanil_concentration = parse_concentration(
            remifentanil_concentration_selection,
            remifentanil_concentration_custom,
            name="remifentanil",
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
            propofol_concentration_mg_ml=propofol_concentration,
            remifentanil_concentration_mcg_ml=remifentanil_concentration,
        )

        return (
            make_summary(rec),
            make_induction_dose_rationale_figure(
                patient=patient,
                rec=rec,
                opiate=opiate,
                propofol_concentration_mg_ml=propofol_concentration,
                remifentanil_concentration_mcg_ml=remifentanil_concentration,
            ),
            make_propofol_pk_figure(rec),
            make_remifentanil_pk_figure(rec),
            make_bis_figure(rec),
            make_map_figure(rec),
        )

    except Exception as e:
        error_component = html.Pre(
            f"Error while running recommendation:\n{str(e)}",
            style={
                "whiteSpace": "pre-wrap",
                "fontFamily": "monospace",
                "color": "#b00020",
                "margin": 0,
            },
        )

        empty = make_empty_figure()
        return error_component, empty, empty, empty, empty, empty


def main() -> None:
    app.run(debug=False)


if __name__ == "__main__":
    main()
