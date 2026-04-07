from __future__ import annotations

import dash
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, ctx

from propofol.dashboard_layout import build_layout
from propofol.recommend_regimen import recommend_propofol_regimen
from propofol.patient import EleveldPatient as Patient


app = dash.Dash(__name__)
app.title = "Propofol Dashboard"
app.layout = build_layout()

def _button_styles(selected: str, current: str):
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
    """Compute MAP from SAP and DAP for display."""
    map = (sap + 2.0 * dap) / 3.0
    return map

def compute_pp(sap: float, dap: float) -> float:
    """Compute PP from SAP and DAP for display."""
    pp = sap - dap
    return pp


# Button callbacks (sex, opiates, mode) - update stored value and button styles
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


@app.callback(
    Output("opiates-store", "data"),
    Output("opiates-no-btn", "style"),
    Output("opiates-yes-btn", "style"),
    Input("opiates-no-btn", "n_clicks"),
    Input("opiates-yes-btn", "n_clicks"),
    State("opiates-store", "data"),
)
def update_opiates(no_clicks, yes_clicks, current_value):
    triggered = ctx.triggered_id
    value = current_value if current_value is not None else False

    if triggered == "opiates-no-btn":
        value = False
    elif triggered == "opiates-yes-btn":
        value = True

    selected = "yes" if value else "no"
    return (
        value,
        _button_styles(selected, "no"),
        _button_styles(selected, "yes"),
    )


@app.callback(
    Output("mode-store", "data"),
    Output("mode-lazy-btn", "style"),
    Output("mode-auto-btn", "style"),
    Output("mode-accurate-btn", "style"),
    Input("mode-lazy-btn", "n_clicks"),
    Input("mode-auto-btn", "n_clicks"),
    Input("mode-accurate-btn", "n_clicks"),
    State("mode-store", "data"),
)
def update_mode(lazy_clicks, auto_clicks, accurate_clicks, current_value):
    triggered = ctx.triggered_id
    value = current_value or "auto"

    if triggered == "mode-lazy-btn":
        value = "lazy"
    elif triggered == "mode-auto-btn":
        value = "auto"
    elif triggered == "mode-accurate-btn":
        value = "accurate"

    return (
        value,
        _button_styles(value, "lazy"),
        _button_styles(value, "auto"),
        _button_styles(value, "accurate"),
    )

# Update derived MAP and PP when baseline SAP or DAP changes
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
        baseline_map = compute_map(float(baseline_sap), float(baseline_dap))
        baseline_pp = compute_pp(float(baseline_sap), float(baseline_dap))
        return f"{baseline_map:.1f}", f"{baseline_pp:.1f}"
    except Exception:
        return "-", "-"

# Build figures for PK, BIS, and MAP from the recommendation result
def make_pk_figure(time_min, cp, ce):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_min, y=cp, mode="lines", name="Plasma concentration (Cp)"))
    fig.add_trace(go.Scatter(x=time_min, y=ce, mode="lines", name="Effect-site concentration (Ce)"))
    fig.update_layout(
        xaxis_title="Time (min)",
        yaxis_title="Propofol concentration (µg/L)",
        template="plotly_white",
    )
    return fig


def make_bis_figure(time_min, bis):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_min, y=bis, mode="lines"))
    fig.add_hline(y=40, line_dash="dash")
    fig.add_hline(y=60, line_dash="dash")
    fig.update_layout(
        xaxis_title="Time (min)",
        yaxis_title="BIS",
        yaxis=dict(range=[0, 100]),
        template="plotly_white",
    )
    return fig


def make_map_figure(time_min, map_mmHg, baseline_map):
    lower_bound = max(65.0, 0.70 * baseline_map)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_min, y=map_mmHg, mode="lines"))
    fig.add_hline(y=lower_bound, line_dash="dash", annotation_text="Lower bound")
    fig.update_layout(
        xaxis_title="Time (min)",
        yaxis_title="MAP (mmHg)",
        yaxis=dict(range=[0, 160]),
        template="plotly_white",
    )
    return fig

# Main callback to run the model and update outputs
@app.callback(
    Output("summary-output", "children"),
    Output("pk-graph", "figure"),
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
    State("opiates-store", "data"),
    State("mode-store", "data"),
    State("maintenance_rate_step", "value"),
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
    opiates,
    mode,
    maintenance_rate_step,
):
    try:
        baseline_sap = float(baseline_sap)
        baseline_dap = float(baseline_dap)
        baseline_hr = float(baseline_hr)

        patient = Patient(
            age=float(age),
            height=float(height),
            weight=float(weight),
            sex=sex,
            opiates=bool(opiates),
            blood_sampling_site="arterial",
            base_sap=baseline_sap,
            base_dap=baseline_dap,
            base_hr=baseline_hr,
        )

        rec = recommend_propofol_regimen(
            patient=patient,
            mode=mode,
            maintenance_rate_step=maintenance_rate_step,
        )

        minute_lines = []
        for i, rate in enumerate(rec.infusion_rates_mgkgh):
            label = "pause" if np.isclose(rate, 0.0, atol=1e-8) else f"{rate:.2f} mg/kg/h"
            minute_lines.append(f"minute {i:02d}: {label}")

        summary = (
            f"Induction dose: {rec.bolus_mg:.0f} mg ({rec.bolus_mgkg:.3f} mg/kg)\n"
            f"Maintenance regimen:\n" + "\n".join(minute_lines)
        )

        pk_fig = make_pk_figure(rec.time_min, rec.cp, rec.ce)
        bis_fig = make_bis_figure(rec.time_min, rec.bis)
        map_fig = make_map_figure(rec.time_min, rec.map_mmHg, patient.base_map)

        return summary, pk_fig, bis_fig, map_fig

    except Exception as e:
        error_text = f"Error while running recommendation:\n{str(e)}"
        empty_fig = go.Figure()
        empty_fig.update_layout(template="plotly_white")
        return error_text, empty_fig, empty_fig, empty_fig


if __name__ == "__main__":
    app.run(debug=False)