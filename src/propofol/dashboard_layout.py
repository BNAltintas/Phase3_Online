from __future__ import annotations

from dash import dcc, html

from propofol.recommend_regimen2023 import (
    DEFAULT_PROPOFOL_CONC_MG_ML,
    DEFAULT_REMI_CONC_MCG_ML,
    MAP_ABS_MIN_TARGET,
    MAP_REL_FRAC_TARGET,
    TARGET_BIS_HIGH,
    TARGET_BIS_LOW,
)

# Static placeholder date/time shown next to fields whose value still matches
# the original EHR/Monitor-sourced default (cosmetic only - there is no real
# EHR/monitor feed wired up yet).
EHR_RECORD_DATE = "01/07/26"
EHR_RECORD_TIME = "08:00"

DROPDOWN_STYLE = {
    "width": "100%",
    "fontSize": "14px",
}


def timestamp_display(date_str: str, time_str: str) -> str:
    """Format a date and time as a single-line "DD/MM/YY HH:MM" timestamp."""
    return f"{date_str} {time_str}"


def field_block(label: str, component):
    """
    Create a labeled input/display block for use inside a `.card`.
    """
    return html.Div(
        [
            html.Label(label, className="field-label"),
            component,
        ],
        className="field-block",
    )


def derived_value_row(label: str, value_id: str, tooltip: str):
    """
    Create a compact, unboxed label/value row for a derived (read-only) value.
    """
    return html.Div(
        [
            html.Span(
                [label, html.Span("i", title=tooltip, className="info-icon")],
                className="derived-label",
            ),
            html.Span(id=value_id, children="-", className="derived-value"),
        ],
        className="derived-value-row",
    )


def param_table_header(show_date: bool = True):
    """Build the PARAMETER / VALUE / SOURCE (/ DATE-TIME) header row for a `.param-table`."""
    header = [
        html.Div("Parameter", className="param-table-header"),
        html.Div("Value", className="param-table-header"),
        html.Div("Source", className="param-table-header"),
    ]
    if show_date:
        header.append(html.Div("Date/Time", className="param-table-header"))
    return header


def editable_field_row(
    label: str,
    field_id: str,
    value: float,
    original_value: float,
    min_value: float | None = None,
    step: float = 1,
    source_label: str = "EHR",
    show_date: bool = True,
):
    """
    Build a click-to-edit parameter row: value badge, source/date, and edit popover.

    The visible value badge is the real `dcc.Input` that callbacks in `app.py`
    read via `State(field_id, "value")` - only its presentation (read-only,
    styled as a badge) changes. Clicking it opens a popover (a hidden sibling
    `html.Div`) containing a scratch input plus Save/Cancel/Restore controls;
    the popover only ever writes back into the same `field_id` value, so the
    data flow into `run_model` is unchanged.

    When show_date is False, the date cell is still rendered (app.py's
    generic popover callback always writes to it) but hidden via inline
    style, so it doesn't reserve a column in `.param-table`.
    """
    return [
        html.Div(label, className="param-label"),
        html.Div(
            dcc.Input(
                id=field_id,
                type="number",
                value=value,
                min=min_value,
                step=step,
                readOnly=True,
                className="value-badge",
            ),
            id=f"{field_id}-badge",
            n_clicks=0,
            className="value-badge-wrapper",
        ),
        html.Div(source_label, id=f"{field_id}-source", className="param-source"),
        html.Div(
            timestamp_display(EHR_RECORD_DATE, EHR_RECORD_TIME),
            id=f"{field_id}-date",
            className="param-date",
            style=None if show_date else {"display": "none"},
        ),
        html.Div(
            [
                dcc.Input(
                    id=f"{field_id}-draft",
                    type="text",
                    value=value,
                    className="field-input",
                ),
                html.Div(id=f"{field_id}-message", className="field-message"),
                html.Div(
                    [
                        html.Button(
                            "Save", id=f"{field_id}-save-btn", n_clicks=0,
                            className="popover-btn popover-btn--save",
                        ),
                        html.Button(
                            "Cancel", id=f"{field_id}-cancel-btn", n_clicks=0,
                            className="popover-btn popover-btn--cancel",
                        ),
                    ],
                    className="popover-actions",
                ),
                html.Div(
                    [
                        html.Span(
                            f"Original {source_label} value: {original_value}",
                            className="popover-original",
                        ),
                        html.Button(
                            "Restore", id=f"{field_id}-restore-btn", n_clicks=0,
                            className="popover-restore popover-restore--hidden",
                        ),
                    ],
                    className="popover-footer",
                ),
            ],
            id=f"{field_id}-popover",
            className="edit-popover",
        ),
    ]


def concentration_block(
    label: str,
    dropdown_id: str,
    custom_input_id: str,
    custom_label: str,
    options: list[dict],
    value,
):
    """
    Create a concentration dropdown plus its "custom value" input, stacked
    for use inside the Model settings card.
    """
    return html.Div(
        [
            field_block(
                label,
                dcc.Dropdown(
                    id=dropdown_id,
                    options=options,
                    value=value,
                    clearable=False,
                    style=DROPDOWN_STYLE,
                ),
            ),
            field_block(
                custom_label,
                dcc.Input(
                    id=custom_input_id,
                    type="number",
                    value=None,
                    min=0,
                    step=0.1,
                    placeholder="Only used if custom",
                    className="field-input",
                ),
            ),
        ]
    )


def graph_card(title: str, graph_id: str, note: str | None = None, tall: bool = False):
    """
    Create a graph card with an optional note.

    The card has a fixed height and is a flex column so the graph fills
    the remaining space immediately, before Plotly's async chunk paints
    anything - this avoids the card growing/jumping after render.
    `config={"responsive": True}` keeps the plot in sync with its
    container if the container is ever resized (e.g. window resize).
    """
    children = [
        html.H4(title, className="graph-card-title"),
    ]

    if note:
        children.append(html.Div(note, className="graph-card-note"))

    children.append(
        dcc.Graph(
            id=graph_id,
            className="graph-card-plot",
            style={"width": "100%", "height": "100%"},
            config={"responsive": True},
        )
    )

    card_class = "graph-card graph-card--tall" if tall else "graph-card"
    return html.Div(children, className=card_class)


def build_patient_parameters_card():
    """
    Build the card containing patient, baseline vitals, and derived value inputs.
    """
    return html.Div(
        [
            html.H4("Patient parameters", className="card-subheading"),
            html.Div(
                [
                    *param_table_header(),
                    *editable_field_row(
                        "Age (years)", "age", 35, 35, min_value=0, step=1,
                    ),
                    *editable_field_row(
                        "Height (cm)", "height", 170, 170, min_value=30, step=1,
                    ),
                    *editable_field_row(
                        "Weight (kg)", "weight", 70, 70, min_value=0.5, step=0.1,
                    ),
                    html.Div("Sex", className="param-label"),
                    html.Div(
                        html.Div(
                            [
                                html.Button(
                                    "Male", id="sex-male-btn", n_clicks=0,
                                    className="toggle-btn",
                                ),
                                html.Button(
                                    "Female", id="sex-female-btn", n_clicks=0,
                                    className="toggle-btn",
                                ),
                            ],
                            className="toggle-buttons",
                        ),
                        className="param-value-toggle",
                        style={"gridColumn": "2 / span 2"},
                    ),
                    html.Div(
                        timestamp_display(EHR_RECORD_DATE, EHR_RECORD_TIME),
                        className="param-date",
                        style={"gridColumn": "4"},
                    ),
                ],
                className="param-table",
            ),
            html.H4("Baseline vitals", className="card-subheading"),
            html.Div(
                [
                    *param_table_header(),
                    *editable_field_row(
                        "SBP (mmHg)", "baseline_sap", 120, 120, min_value=30, step=0.1,
                        source_label="Monitor",
                    ),
                    *editable_field_row(
                        "DBP (mmHg)", "baseline_dap", 70, 70, min_value=10, step=0.1,
                        source_label="Monitor",
                    ),
                    *editable_field_row(
                        "HR (bpm)", "baseline_hr", 70, 70, min_value=0, step=0.1,
                        source_label="Monitor",
                    ),
                ],
                className="param-table",
            ),
            html.H4("Derived values", className="card-subheading"),
            html.Div(
                [
                    derived_value_row(
                        "MAP (mmHg)", "derived-map", "MAP = DBP + (SBP − DBP) / 3",
                    ),
                    derived_value_row(
                        "Baseline PP (mmHg)", "derived-pp", "Baseline PP = SBP − DBP",
                    ),
                ],
                className="derived-values",
            ),
        ],
        className="card",
    )


def build_model_settings_card():
    """
    Build the card containing opiate/concentration selection and read-only model targets.
    """
    return html.Div(
        [
            html.H4("Model settings", className="card-subheading"),
            field_block(
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
            ),
            concentration_block(
                label="Propofol concentration (mg/mL)",
                dropdown_id="propofol-concentration-dropdown",
                custom_input_id="propofol-concentration-custom",
                custom_label="Custom propofol concentration (mg/mL)",
                options=[
                    {"label": "10 mg/mL", "value": "10"},
                    {"label": "20 mg/mL", "value": "20"},
                    {"label": "Custom", "value": "custom"},
                ],
                value=str(int(DEFAULT_PROPOFOL_CONC_MG_ML)),
            ),
            concentration_block(
                label="Remifentanil concentration (µg/mL)",
                dropdown_id="remifentanil-concentration-dropdown",
                custom_input_id="remifentanil-concentration-custom",
                custom_label="Custom remifentanil concentration (µg/mL)",
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
                value=str(int(DEFAULT_REMI_CONC_MCG_ML)),
            ),
            html.H4("Target BIS", className="card-subheading"),
            html.Div(
                [
                    *param_table_header(show_date=False),
                    *editable_field_row(
                        "Lower", "bis-target-low", TARGET_BIS_LOW, TARGET_BIS_LOW,
                        min_value=0, step=1, source_label="Default", show_date=False,
                    ),
                    *editable_field_row(
                        "Upper", "bis-target-high", TARGET_BIS_HIGH, TARGET_BIS_HIGH,
                        min_value=0, step=1, source_label="Default", show_date=False,
                    ),
                ],
                className="param-table param-table--no-date",
            ),
            html.H4("Target MAP", className="card-subheading"),
            html.Div(
                [
                    *param_table_header(show_date=False),
                    *editable_field_row(
                        "Absolute (mmHg)", "map-target-abs",
                        MAP_ABS_MIN_TARGET, MAP_ABS_MIN_TARGET,
                        min_value=30, step=1, source_label="Default", show_date=False,
                    ),
                    *editable_field_row(
                        "Relative (% baseline)", "map-target-rel",
                        MAP_REL_FRAC_TARGET * 100, MAP_REL_FRAC_TARGET * 100,
                        min_value=30, step=1, source_label="Default", show_date=False,
                    ),
                ],
                className="param-table param-table--no-date",
            ),
        ],
        className="card",
    )


def build_sidebar():
    """
    Build the left navigation sidebar.

    Only "Recommendation" is functional; "Scenario Exploration" and "More
    Info" are placeholders for pages that do not exist yet.
    """
    return html.Div(
        [
            html.Div("DosePilot", className="sidebar-brand"),
            html.Div(
                [
                    html.Div(
                        "Recommendation",
                        className="sidebar-nav-item sidebar-nav-item--active",
                    ),
                    html.Div(
                        "Scenario Exploration",
                        className="sidebar-nav-item sidebar-nav-item--disabled",
                    ),
                    html.Div(
                        "More Info",
                        className="sidebar-nav-item sidebar-nav-item--disabled",
                    ),
                ],
                className="sidebar-nav",
            ),
        ],
        className="sidebar",
    )


def build_input_column():
    """
    Build the left dashboard column: patient/model inputs plus the run button.
    """
    return html.Div(
        [
            html.Div("INPUT PARAMETERS", className="section-label"),
            build_patient_parameters_card(),
            build_model_settings_card(),
            html.Button(
                "Run recommendation",
                id="run-btn",
                n_clicks=0,
                style={
                    "width": "100%",
                    "padding": "12px 20px",
                    "fontSize": "15px",
                    "fontWeight": "700",
                    "borderRadius": "10px",
                    "border": "none",
                    "backgroundColor": "#1f77b4",
                    "color": "white",
                    "cursor": "pointer",
                },
            ),
        ],
        className="input-column",
    )


def build_recommendation_column():
    """
    Build the center dashboard column: the recommendation summary, confidence,
    and maintenance regimen (all rendered together into `summary-output`).
    """
    return html.Div(
        [
            html.Div("RECOMMENDATION", className="section-label"),
            html.Div(
                id="summary-output",
                style={
                    "whiteSpace": "pre-wrap",
                    "padding": "18px",
                    "border": "1px solid #ddd",
                    "borderRadius": "12px",
                    "backgroundColor": "#fafafa",
                    "minHeight": "80px",
                },
            ),
        ],
        className="recommendation-column",
    )


def build_explanation_column():
    """
    Build the right dashboard column: the induction-dose rationale (explanation) graph.
    """
    return html.Div(
        [
            html.Div("EXPLANATION", className="section-label"),
            graph_card(
                "Induction-dose rationale",
                "dose-rationale-graph",
                tall=True,
            ),
        ],
        className="explanation-column",
    )


def build_predictions_section():
    """
    Build the bottom predictions section: propofol/remifentanil PK and BIS/MAP graphs.
    """
    return html.Div(
        [
            html.Div("PREDICTIONS", className="section-label"),
            html.Div(
                [
                    graph_card("Propofol PK", "propofol-pk-graph"),
                    graph_card("Remifentanil PK", "remifentanil-pk-graph"),
                    graph_card("BIS", "bis-graph"),
                    graph_card("MAP", "map-graph"),
                ],
                className="predictions-grid",
            ),
        ],
        className="predictions-section",
    )


def build_layout():
    """
    Build the main layout of the dashboard: a navigation sidebar plus a
    three-column input/recommendation/explanation dashboard, with the
    prediction graphs below.

    The input column sits outside `dcc.Loading` (as before, inputs never
    show a loading overlay); the recommendation column, explanation column,
    and predictions section are all inside one `dcc.Loading`, so every
    `run_model` output shows the same loading feedback it did previously.
    """
    return html.Div(
        [
            dcc.Store(id="sex-store", data="male"),

            build_sidebar(),

            html.Div(
                [
                    html.H2(
                        "Propofol ± Opiate Dose Recommendation",
                        className="page-title",
                    ),

                    html.Div(
                        [
                            build_input_column(),
                            dcc.Loading(
                                id="recommendation-loading",
                                type="default",
                                parent_className="output-column",
                                children=[
                                    html.Div(
                                        [
                                            build_recommendation_column(),
                                            build_explanation_column(),
                                        ],
                                        className="content-grid",
                                    ),
                                    build_predictions_section(),
                                ],
                            ),
                        ],
                        className="content-grid",
                    ),
                ],
                className="main-content",
            ),
        ],
        className="app-shell",
    )
