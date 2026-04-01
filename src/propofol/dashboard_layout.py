from __future__ import annotations

from dash import dcc, html


def input_block(label: str, component, width: str = "220px"):
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
            component,
        ],
        style={
            "width": width,
            "display": "inline-block",
            "verticalAlign": "top",
            "marginRight": "18px",
            "marginBottom": "18px",
        },
    )


def value_display_block(label: str, value_id: str, width: str = "220px"):
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
    buttons = [
        html.Button(
            text,
            id=btn_id,
            n_clicks=0,
            style={
                "padding": "10px 16px",
                "border": "1px solid #bbb",
                "backgroundColor": "#f7f7f7",
                "cursor": "pointer",
                "marginRight": "8px",
                "borderRadius": "8px",
                "fontWeight": "600",
            },
        )
        for btn_id, text in button_ids_and_labels
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


def build_layout():
    return html.Div(
        [
            dcc.Store(id="sex-store", data="male"),
            dcc.Store(id="opiates-store", data=False),
            dcc.Store(id="mode-store", data="auto"),

            html.H2("Propofol Induction & Initial Maintenance Recommended Dose"),

            html.Div(
                "Predicts a recommended bolus and minute-wise maintenance schedule for 15 minutes, "
                "constraining BIS between 40 and 60, and MAP above 65 mmHg or 70% of baseline.",
                style={"marginBottom": "20px"},
            ),

            html.Div(
                [
                    html.H4("Patient inputs"),

                    html.Div(
                        [
                            input_block(
                                "Age (years)",
                                dcc.Input(id="age", type="number", value=35, min=18, step=1),
                            ),
                            input_block(
                                "Height (cm)",
                                dcc.Input(id="height", type="number", value=170, min=100, step=1),
                            ),
                            input_block(
                                "Weight (kg)",
                                dcc.Input(id="weight", type="number", value=70, min=20, step=0.1),
                            ),
                            input_block(
                                "Baseline SAP (mmHg)",
                                dcc.Input(id="baseline_sap", type="number", value=120, min=1, step=0.1),
                            ),
                            input_block(
                                "Baseline DAP (mmHg)",
                                dcc.Input(id="baseline_dap", type="number", value=70, min=1, step=0.1),
                            ),
                            input_block(
                                "Baseline HR (bpm)",
                                dcc.Input(id="baseline_hr", type="number", value=70, min=1, step=0.1),
                            ),
                        ]
                    ),

                    html.Div(
                        [
                            value_display_block("Derived baseline MAP (mmHg)", "derived-map"),
                            value_display_block("Derived baseline PP (mmHg)", "derived-pp"),
                        ],
                    ),
                ],
                style={
                    "padding": "18px",
                    "border": "1px solid #ddd",
                    "borderRadius": "12px",
                    "marginBottom": "20px",
                },
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

                    segmented_buttons(
                        "Opiates",
                        [
                            ("opiates-no-btn", "No"),
                            ("opiates-yes-btn", "Yes"),
                        ],
                    ),

                    segmented_buttons(
                        "Mode",
                        [
                            ("mode-lazy-btn", "Lazy"),
                            ("mode-auto-btn", "Auto"),
                            ("mode-accurate-btn", "Accurate"),
                        ],
                    ),

                    input_block(
                        "Maintenance rate rounding step (mg/kg/h, optional)",
                        dcc.Input(
                            id="maintenance_rate_step",
                            type="number",
                            value=None,
                            step=0.1,
                            placeholder="e.g. 0.5",
                        ),
                        width="280px",
                    ),
                ],
                style={
                    "padding": "18px",
                    "border": "1px solid #ddd",
                    "borderRadius": "12px",
                    "marginBottom": "20px",
                },
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

            html.Div(
                id="summary-output",
                style={
                    "whiteSpace": "pre-wrap",
                    "padding": "18px",
                    "border": "1px solid #ddd",
                    "borderRadius": "12px",
                    "marginBottom": "20px",
                    "backgroundColor": "#fafafa",
                },
            ),

            html.Div(
                [
                    dcc.Graph(id="pk-graph"),
                ],
                style={"marginBottom": "12px"},
            ),

            html.Div(
                [
                    html.Div(
                        dcc.Graph(id="bis-graph"),
                        style={
                            "flex": "1 1 500px",
                            "minWidth": "320px",
                        },
                    ),
                    html.Div(
                        dcc.Graph(id="map-graph"),
                        style={
                            "flex": "1 1 500px",
                            "minWidth": "320px",
                        },
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
        style={
            "maxWidth": "1300px",
            "margin": "0 auto",
            "padding": "24px",
            "fontFamily": "Arial, sans-serif",
        },
    )