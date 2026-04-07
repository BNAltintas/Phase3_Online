from __future__ import annotations

from dash import dcc, html
from propofol.config import (
    BIS_HIGH,
    BIS_LOW,
    MAP_ABS_MIN,
    MAP_REL_FRAC,
    SIM_MIN,
)


def input_block(label: str, component, width: str = "220px"):
    """Create a labeled input block with consistent styling.

    Parameters
    ----------
    label : str
        The label text displayed above the input component.
    component
        The Dash component (e.g., dcc.Input) to display.
    width : str, optional
        CSS width of the block, by default "220px".

    Returns
    -------
    html.Div
        A styled container with the label and component.
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
    """Create a labeled read-only value display block.

    Parameters
    ----------
    label : str
        The label text displayed above the value.
    value_id : str
        The Dash component ID where the value will be updated.
    width : str, optional
        CSS width of the block, by default "220px".

    Returns
    -------
    html.Div
        A styled container with the label and a value display area.
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
    """Create a group of segmented buttons with a label.

    Parameters
    ----------
    label : str
        The label text displayed above the button group.
    button_ids_and_labels : list[tuple[str, str]]
        List of (button_id, button_label) tuples defining the buttons.

    Returns
    -------
    html.Div
        A styled container with the label and segmented button group.
    """
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
    """Build the main dashboard layout structure.

    Returns
    -------
    html.Div
        The complete dashboard layout with patient inputs, options, and output sections.
    """
    return html.Div(
        [
            dcc.Store(id="sex-store", data="male"),
            dcc.Store(id="opiates-store", data=False),
            dcc.Store(id="mode-store", data="auto"),

            html.H2("Propofol Induction & Initial Maintenance Recommended Dose"),

            html.Div(
                f"Predicts a recommended bolus and minute-wise maintenance schedule for {SIM_MIN} "
                f"minutes, constraining BIS between {BIS_LOW} and {BIS_HIGH}, and MAP above "
                f"{MAP_ABS_MIN} mmHg or {int(MAP_REL_FRAC * 100)}% of baseline.",
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
                                dcc.Input(id="baseline_sap", type="number", value=120, min=1,
                                          step=0.1),
                            ),
                            input_block(
                                "Baseline DAP (mmHg)",
                                dcc.Input(id="baseline_dap", type="number", value=70, min=1,
                                          step=0.1),
                            ),
                            input_block(
                                "Baseline HR (bpm)",
                                dcc.Input(id="baseline_hr", type="number", value=70, min=1,
                                          step=0.1),
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
