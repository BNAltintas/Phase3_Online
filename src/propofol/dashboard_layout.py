from __future__ import annotations

from dash import dcc, html

from propofol.recommend_regimen2023 import (
    CONFIDENCE_N_SIMULATIONS,
    DEFAULT_PROPOFOL_CONCENTRATION_MG_ML,
    DEFAULT_REMI_CONCENTRATION_MCG_ML,
    MAP_ABS_MIN_TARGET,
    MAP_REL_FRAC_TARGET,
    TARGET_ASSESSMENT_START_MIN,
    TARGET_BIS_HIGH,
    TARGET_BIS_LOW,
)


CARD_STYLE = {
    "padding": "18px",
    "border": "1px solid #ddd",
    "borderRadius": "12px",
    "marginBottom": "20px",
    "backgroundColor": "white",
}

INPUT_STYLE = {
    "width": "100%",
    "padding": "9px 10px",
    "border": "1px solid #ccc",
    "borderRadius": "8px",
    "fontSize": "14px",
    "boxSizing": "border-box",
}

DROPDOWN_STYLE = {
    "width": "100%",
    "fontSize": "14px",
}

DEFAULT_BUTTON_STYLE = {
    "padding": "10px 16px",
    "border": "1px solid #bbb",
    "backgroundColor": "#f7f7f7",
    "cursor": "pointer",
    "marginRight": "8px",
    "borderRadius": "8px",
    "fontWeight": "600",
}

NOTE_STYLE = {
    "fontSize": "13px",
    "color": "#666",
    "lineHeight": "1.35",
    "marginTop": "6px",
}


def input_block(label: str, component, width: str = "220px", note: str | None = None):
    """
    Create a labeled input block with an optional note.
    """
    children = [
        html.Label(
            label,
            style={
                "fontWeight": "600",
                "marginBottom": "6px",
                "display": "block",
            },
        ),
        component,
    ]

    if note:
        children.append(html.Div(note, style=NOTE_STYLE))

    return html.Div(
        children,
        style={
            "width": width,
            "display": "inline-block",
            "verticalAlign": "top",
            "marginRight": "18px",
            "marginBottom": "18px",
        },
    )


def value_display_block(label: str, value_id: str, width: str = "220px"):
    """
    Create a labeled value display block.
    """
    return html.Div(
        [
            html.Label(
                label,
                style={
                    "fontWeight": "600",
                    "marginBottom": "6px",
                    "display": "block",
                },
            ),
            html.Div(
                id=value_id,
                children="-",
                style={
                    "padding": "10px 12px",
                    "border": "1px solid #ddd",
                    "borderRadius": "8px",
                    "backgroundColor": "#fafafa",
                    "minHeight": "20px",
                },
            ),
        ],
        style={
            "width": width,
            "display": "inline-block",
            "verticalAlign": "top",
            "marginRight": "18px",
            "marginBottom": "18px",
        },
    )


def segmented_buttons(label: str, button_ids_and_labels: list[tuple[str, str]]):
    """
    Create a segmented button group with a label.
    """
    buttons = [
        html.Button(
            text,
            id=button_id,
            n_clicks=0,
            style=DEFAULT_BUTTON_STYLE,
        )
        for button_id, text in button_ids_and_labels
    ]

    return html.Div(
        [
            html.Label(
                label,
                style={
                    "fontWeight": "600",
                    "marginBottom": "6px",
                    "display": "block",
                },
            ),
            html.Div(buttons),
        ],
        style={"marginBottom": "18px"},
    )


def concentration_dropdown(
    label: str,
    dropdown_id: str,
    custom_input_id: str,
    options: list[dict],
    value,
    unit: str,
):
    """
    Create a concentration dropdown with a custom input option.
    """
    return html.Div(
        [
            input_block(
                label,
                dcc.Dropdown(
                    id=dropdown_id,
                    options=options,
                    value=value,
                    clearable=False,
                    style=DROPDOWN_STYLE,
                ),
                width="280px",
            ),
            input_block(
                f"Custom {unit}",
                dcc.Input(
                    id=custom_input_id,
                    type="number",
                    value=None,
                    min=0,
                    step=0.1,
                    placeholder="Only used if custom",
                    style=INPUT_STYLE,
                ),
                width="220px",
            ),
        ],
        style={"display": "block"},
    )


def graph_card(title: str, graph_id: str, note: str | None = None):
    """
    Create a graph card with an optional note.
    """
    children = [
        html.H4(title, style={"marginTop": 0, "marginBottom": "8px"}),
    ]

    if note:
        children.append(html.Div(note, style=NOTE_STYLE | {"marginBottom": "8px"}))

    children.append(dcc.Graph(id=graph_id))

    return html.Div(children, style=CARD_STYLE)


def build_layout():
    """
    Build the main layout of the dashboard.
    """
    return html.Div(
        [
            dcc.Store(id="sex-store", data="male"),

            html.H2(
                "Propofol ± Opiate Dose Recommendation",
                style={"marginBottom": "8px"},
            ),

            html.Div(
                [
                    html.H4("Patient inputs"),

                    html.Div(
                        [
                            input_block(
                                "Age (years)",
                                dcc.Input(
                                    id="age",
                                    type="number",
                                    value=35,
                                    min=18,
                                    step=1,
                                    style=INPUT_STYLE,
                                ),
                            ),
                            input_block(
                                "Height (cm)",
                                dcc.Input(
                                    id="height",
                                    type="number",
                                    value=170,
                                    min=100,
                                    step=1,
                                    style=INPUT_STYLE,
                                ),
                            ),
                            input_block(
                                "Weight (kg)",
                                dcc.Input(
                                    id="weight",
                                    type="number",
                                    value=70,
                                    min=20,
                                    step=0.1,
                                    style=INPUT_STYLE,
                                ),
                            ),
                            input_block(
                                "Baseline SAP (mmHg)",
                                dcc.Input(
                                    id="baseline_sap",
                                    type="number",
                                    value=120,
                                    min=1,
                                    step=0.1,
                                    style=INPUT_STYLE,
                                ),
                            ),
                            input_block(
                                "Baseline DAP (mmHg)",
                                dcc.Input(
                                    id="baseline_dap",
                                    type="number",
                                    value=70,
                                    min=1,
                                    step=0.1,
                                    style=INPUT_STYLE,
                                ),
                            ),
                            input_block(
                                "Baseline HR (bpm)",
                                dcc.Input(
                                    id="baseline_hr",
                                    type="number",
                                    value=70,
                                    min=1,
                                    step=0.1,
                                    style=INPUT_STYLE,
                                ),
                            ),
                        ]
                    ),

                    html.Div(
                        [
                            value_display_block("Derived baseline MAP (mmHg)", "derived-map"),
                            value_display_block("Derived baseline PP (mmHg)", "derived-pp"),
                        ]
                    ),
                ],
                style=CARD_STYLE,
            ),

            html.Div(
                [
                    html.H4("Options"),

                    segmented_buttons(
                        "Sex",
                        [
                            ("sex-male-btn", "Male"),
                            ("sex-female-btn", "Female"),
                        ],
                    ),

                    input_block(
                        "Select opiate",
                        dcc.Dropdown(
                            id="opiate-dropdown",
                            options=[
                                {"label": "No opiate / propofol only", "value": "none"},
                                {"label": "Remifentanil", "value": "remifentanil"},
                                {
                                    "label": "Sufentanil (upcoming)",
                                    "value": "sufentanil",
                                    "disabled": True,
                                },
                                {
                                    "label": "Fentanyl (upcoming)",
                                    "value": "fentanyl",
                                    "disabled": True,
                                },
                            ],
                            value="none",
                            clearable=False,
                            placeholder="Select opiate",
                            style=DROPDOWN_STYLE,
                        ),
                        width="320px",
                    ),
                ],
                style=CARD_STYLE,
            ),

            html.Div(
                [
                    html.H4("Concentrations"),

                    concentration_dropdown(
                        label="Propofol concentration (mg/mL)",
                        dropdown_id="propofol-concentration-dropdown",
                        custom_input_id="propofol-concentration-custom",
                        options=[
                            {"label": "10 mg/mL", "value": "10"},
                            {"label": "20 mg/mL", "value": "20"},
                            {"label": "Custom", "value": "custom"},
                        ],
                        value=str(int(DEFAULT_PROPOFOL_CONCENTRATION_MG_ML)),
                        unit="mg/mL",
                    ),

                    concentration_dropdown(
                        label="Remifentanil concentration (µg/mL)",
                        dropdown_id="remifentanil-concentration-dropdown",
                        custom_input_id="remifentanil-concentration-custom",
                        options=[
                            {"label": "5 µg/mL", "value": "5"},
                            {"label": "10 µg/mL", "value": "10"},
                            {"label": "20 µg/mL", "value": "20"},
                            {"label": "25 µg/mL", "value": "25"},
                            {"label": "40 µg/mL", "value": "40"},
                            {"label": "50 µg/mL", "value": "50"},
                            {"label": "100 µg/mL", "value": "100"},
                            {"label": "Custom", "value": "custom"},
                        ],
                        value=str(int(DEFAULT_REMI_CONCENTRATION_MCG_ML)),
                        unit="µg/mL",
                    ),
                ],
                style=CARD_STYLE,
            ),

            html.Button(
                "Run recommendation",
                id="run-btn",
                n_clicks=0,
                style={
                    "padding": "12px 20px",
                    "fontSize": "16px",
                    "fontWeight": "700",
                    "borderRadius": "10px",
                    "border": "none",
                    "backgroundColor": "#1f77b4",
                    "color": "white",
                    "cursor": "pointer",
                    "marginBottom": "20px",
                },
            ),

            dcc.Loading(
                id="recommendation-loading",
                type="default",
                children=[
                    html.Div(
                        id="summary-output",
                        style={
                            "whiteSpace": "pre-wrap",
                            "padding": "18px",
                            "border": "1px solid #ddd",
                            "borderRadius": "12px",
                            "marginBottom": "20px",
                            "backgroundColor": "#fafafa",
                            "minHeight": "80px",
                        },
                    ),

                    graph_card(
                        "Induction-dose rationale",
                        "dose-rationale-graph",
                    ),

                    html.Div(
                        [
                            html.Div(
                                graph_card("Propofol PK", "propofol-pk-graph"),
                                style={"flex": "1 1 500px", "minWidth": "320px"},
                            ),
                            html.Div(
                                graph_card("Remifentanil PK", "remifentanil-pk-graph"),
                                style={"flex": "1 1 500px", "minWidth": "320px"},
                            ),
                        ],
                        style={
                            "display": "flex",
                            "flexWrap": "wrap",
                            "gap": "12px",
                            "alignItems": "stretch",
                        },
                    ),

                    html.Div(
                        [
                            html.Div(
                                graph_card("BIS", "bis-graph"),
                                style={"flex": "1 1 500px", "minWidth": "320px"},
                            ),
                            html.Div(
                                graph_card("MAP", "map-graph"),
                                style={"flex": "1 1 500px", "minWidth": "320px"},
                            ),
                        ],
                        style={
                            "display": "flex",
                            "flexWrap": "wrap",
                            "gap": "12px",
                            "alignItems": "stretch",
                        },
                    ),
                ],
            ),
        ],
        style={
            "maxWidth": "1300px",
            "margin": "0 auto",
            "padding": "24px",
            "fontFamily": "Arial, sans-serif",
            "backgroundColor": "#ffffff",
        },
    )
