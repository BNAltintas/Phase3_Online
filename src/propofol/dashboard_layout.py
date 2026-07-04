from __future__ import annotations

from dash import dcc, html

from propofol.more_info_content import MORE_INFO_SECTIONS, MORE_INFO_SUBTITLE, MORE_INFO_TITLE
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


def graph_card(
    title: str,
    graph_id: str,
    note: str | None = None,
    tall: bool = False,
    default_visible: bool = True,
):
    """
    Create a graph card with a per-card "Show" checkbox in its header.

    The header (title left, "Show" checkbox right) is always rendered.
    The graph content below it collapses to nothing (no empty space) when
    unchecked - handled generically for every graph card by
    app.py's toggle_graph_visibility callback, keyed on graph_id via a
    Dash pattern-matching id, so this is one mechanism reused by all 5
    graph cards rather than bespoke per-card callbacks.

    The card has a fixed height while expanded and is a flex column so the
    graph fills the remaining space immediately, before Plotly's async
    chunk paints anything - this avoids the card growing/jumping after
    render. `config={"responsive": True}` keeps the plot in sync with its
    container if the container is ever resized (e.g. window resize, or
    becoming visible again after being hidden).
    """
    header = html.Div(
        [
            html.H4(title, className="graph-card-title"),
            dcc.Checklist(
                id={"type": "graph-visibility-toggle", "graph": graph_id},
                options=[{"label": "Show", "value": "show"}],
                value=["show"] if default_visible else [],
                className="graph-card-toggle",
            ),
        ],
        className="graph-card-header",
    )

    content_children = []
    if note:
        content_children.append(html.Div(note, className="graph-card-note"))

    content_children.append(
        dcc.Graph(
            id=graph_id,
            className="graph-card-plot",
            style={"width": "100%", "height": "100%"},
            config={"responsive": True},
        )
    )

    content = html.Div(
        content_children,
        id={"type": "graph-card-content", "graph": graph_id},
        className="graph-card-content",
        style=None if default_visible else {"display": "none"},
    )

    card_class = "graph-card graph-card--tall" if tall else "graph-card"
    if not default_visible:
        card_class += " graph-card--collapsed"

    return html.Div(
        [header, content],
        id={"type": "graph-card", "graph": graph_id},
        className=card_class,
    )


STALE_MESSAGE = "Input parameters changed. Re-run recommendation to update dosing guidance."


def _placeholder_card(text: str, card_id: str, stale: bool = False, hidden: bool = False):
    """
    Build a "no result to show yet" placeholder card - either the initial
    "not run yet" message or the "inputs changed, re-run" stale message.
    Purely presentational: it carries no data and is only ever shown/hidden
    via its `style` prop by `update_result_visibility` in app.py.
    """
    class_name = "card result-placeholder"
    if stale:
        class_name += " result-placeholder--stale"
    return html.Div(
        text,
        id=card_id,
        className=class_name,
        style={"display": "none"} if hidden else None,
    )


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
                        dcc.Dropdown(
                            id="sex-dropdown",
                            options=[
                                {"label": "Male", "value": "male"},
                                {"label": "Female", "value": "female"},
                            ],
                            value="male",
                            clearable=False,
                            searchable=False,
                        ),
                        className="param-value-dropdown",
                    ),
                    html.Div("EHR", className="param-source"),
                    html.Div(
                        timestamp_display(EHR_RECORD_DATE, EHR_RECORD_TIME),
                        className="param-date",
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
            html.H4("Concentrations", className="card-subheading"),
            html.Div(
                [
                    *param_table_header(show_date=False),
                    *editable_field_row(
                        "Propofol (mg/mL)", "propofol-concentration",
                        DEFAULT_PROPOFOL_CONC_MG_ML, DEFAULT_PROPOFOL_CONC_MG_ML,
                        min_value=0.1, step=0.1, source_label="Default", show_date=False,
                    ),
                    *editable_field_row(
                        "Remifentanil (µg/mL)", "remifentanil-concentration",
                        DEFAULT_REMI_CONC_MCG_ML, DEFAULT_REMI_CONC_MCG_ML,
                        min_value=0.1, step=0.1, source_label="Default", show_date=False,
                    ),
                ],
                className="param-table param-table--no-date",
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


def _accordion_section(section_id: str, title: str, body_markdown: str, is_open: bool):
    """
    Build one collapsible "More Info" section: a clickable header (title +
    chevron) and a body pane holding its Markdown text.

    Only styling/visibility (className) is data-driven from `is_open` here
    and by `render_more_info_accordion` in app.py afterwards - the
    multi-expand open/collapse behavior is entirely presentational and
    never touches MORE_INFO_SECTIONS' content.
    """
    chevron_class = "accordion-chevron accordion-chevron--open" if is_open else "accordion-chevron"
    body_class = "accordion-body" if is_open else "accordion-body accordion-body--collapsed"

    return html.Div(
        [
            html.Div(
                [
                    html.Span(title, className="accordion-title"),
                    html.Span(
                        "⌄",
                        id={"type": "more-info-accordion-chevron", "section": section_id},
                        className=chevron_class,
                    ),
                ],
                id={"type": "more-info-accordion-header", "section": section_id},
                n_clicks=0,
                className="accordion-header",
            ),
            html.Div(
                dcc.Markdown(body_markdown, className="accordion-body-text"),
                id={"type": "more-info-accordion-body", "section": section_id},
                className=body_class,
            ),
        ],
        className="card accordion-section",
    )


def build_more_info_view():
    """
    Build the "More Info" page: a "Back to recommendation" button, a
    centered title/subtitle, and a multi-expand accordion of explanatory
    sections (content defined in more_info_content.py, not here).

    The accordion supports any number of sections open at once - each
    header toggles only its own section's membership in
    more-info-open-sections (a list of open ids), so opening one section
    never closes another. The first section starts open, matching the
    reference design.
    """
    sections = [
        _accordion_section(
            section["id"], section["title"], section["body"],
            is_open=(index == 0),
        )
        for index, section in enumerate(MORE_INFO_SECTIONS)
    ]

    return html.Div(
        html.Div(
            [
                # List of currently-open section ids - the first section
                # starts open, matching the sections' own is_open above.
                dcc.Store(id="more-info-open-sections", data=[MORE_INFO_SECTIONS[0]["id"]]),
                html.Button(
                    "← Back to recommendation",
                    id="back-to-recommendation-btn",
                    n_clicks=0,
                    className="back-to-recommendation-btn",
                ),
                html.H2(MORE_INFO_TITLE, className="more-info-title"),
                html.P(MORE_INFO_SUBTITLE, className="more-info-subtitle"),
                html.Div(sections, className="accordion"),
            ],
            className="more-info-content",
        ),
        id="more-info-view",
        className="more-info-view",
        style={"display": "none"},
    )


def build_sidebar():
    """
    Build the left navigation sidebar.

    "Recommendation" and "More Info" switch between the two pages;
    "Scenario Exploration" is still a placeholder for a page that does not
    exist yet.
    """
    return html.Div(
        [
            html.Div("DosePilot", className="sidebar-brand"),
            html.Div(
                [
                    html.Div(
                        "Recommendation",
                        id="nav-recommendation-btn",
                        n_clicks=0,
                        className="sidebar-nav-item sidebar-nav-item--active",
                    ),
                    html.Div(
                        "Scenario Exploration",
                        className="sidebar-nav-item sidebar-nav-item--disabled",
                    ),
                    html.Div(
                        "More Info",
                        id="nav-more-info-btn",
                        n_clicks=0,
                        className="sidebar-nav-item",
                    ),
                ],
                className="sidebar-nav",
            ),
        ],
        className="sidebar",
    )


def build_patient_id_card():
    """
    Build the small patient-identification card shown above Input Parameters.

    Static placeholder content only (hardcoded "Test Patient" / "TEST-001") -
    no ids, no callbacks, no effect on model inputs or data flow.
    """
    return html.Div(
        [
            html.Span("👤", className="patient-id-icon"),
            html.Div(
                [
                    html.Div("Test Patient", className="patient-id-name"),
                    html.Div("Patient ID: TEST-001", className="patient-id-label"),
                ],
                className="patient-id-info",
            ),
        ],
        className="card patient-id-card",
    )


def build_input_column():
    """
    Build the left dashboard column: patient ID card, patient/model inputs,
    and the run button.
    """
    return html.Div(
        [
            build_patient_id_card(),
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
    Build the center dashboard column: the recommendation summary (rendered
    into `summary-output` as an induction-dose card followed by a
    maintenance-regimen card, each with its own `.card` styling), with the
    induction-dose rationale graph stacked directly underneath.

    Before a recommendation has ever been run, and whenever the current
    recommendation is stale (inputs changed since it was generated),
    `summary-output` and the rationale graph card are hidden behind a
    placeholder card instead - see `update_result_visibility` in app.py,
    which is the sole owner of all of these `style` props.
    """
    return html.Div(
        [
            html.Div("RECOMMENDATION", className="section-label"),
            _placeholder_card(
                "Run recommendation to generate personalized dosing guidance.",
                "recommendation-empty-placeholder",
            ),
            _placeholder_card(
                STALE_MESSAGE, "recommendation-stale-placeholder", stale=True, hidden=True,
            ),
            html.Div(id="summary-output", style={"display": "none"}),
            _placeholder_card(
                "Explanation will appear after running recommendation.",
                "rationale-empty-placeholder",
            ),
            _placeholder_card(
                STALE_MESSAGE, "rationale-stale-placeholder", stale=True, hidden=True,
            ),
            html.Div(
                graph_card(
                    "Induction-dose rationale",
                    "dose-rationale-graph",
                    tall=True,
                    default_visible=True,
                ),
                id="dose-rationale-card-wrapper",
                style={"display": "none"},
            ),
        ],
        className="recommendation-column",
    )


def build_predictions_column():
    """
    Build the right dashboard column: BIS, MAP, and PK graphs stacked
    vertically, hidden behind a placeholder card before a recommendation
    exists or while it is stale - see build_recommendation_column's
    docstring and `update_result_visibility` in app.py.
    """
    return html.Div(
        [
            html.Div("PREDICTIONS", className="section-label"),
            _placeholder_card(
                "Predictions will appear after running recommendation.",
                "predictions-empty-placeholder",
            ),
            _placeholder_card(
                STALE_MESSAGE, "predictions-stale-placeholder", stale=True, hidden=True,
            ),
            html.Div(
                [
                    graph_card("BIS", "bis-graph", default_visible=True),
                    graph_card("MAP", "map-graph", default_visible=True),
                    graph_card("Propofol PK/PD", "propofol-pk-graph", default_visible=False),
                    graph_card("Remifentanil PK", "remifentanil-pk-graph", default_visible=False),
                ],
                id="predictions-grid",
                className="predictions-grid",
                style={"display": "none"},
            ),
        ],
        className="predictions-column",
    )


def build_layout():
    """
    Build the main layout of the dashboard: a navigation sidebar plus a
    three-column input/recommendation/predictions dashboard, or the "More
    Info" page - the two views are siblings, and only one is ever visible
    at a time (Output("main-content", "style") /
    Output("more-info-view", "style"), driven by active-page-store and
    owned by render_active_page in app.py).

    The input column sits outside `dcc.Loading` (as before, inputs never
    show a loading overlay); the recommendation column (including the
    induction-dose rationale graph) and the predictions column are both
    inside one `dcc.Loading`, so every `run_model` output shows the same
    loading feedback it did previously.
    """
    return html.Div(
        [
            dcc.Store(id="sex-store", data="male"),
            # Holds the original model recommendation (context + result),
            # populated only by "Run recommendation" - never mutated by the
            # manual-override flow.
            dcc.Store(id="recommendation-store", data=None),
            # Holds the manual-override scenario, if any. None when inactive.
            # Cleared by "Return to recommendation" and by every new
            # "Run recommendation" click.
            dcc.Store(id="manual-scenario-store", data=None),
            # True when a patient/model-setting input has changed since
            # recommendation-store's current data was generated. Reset to
            # False by every "Run recommendation" click.
            dcc.Store(id="recommendation-stale-store", data=False),
            # "recommendation" or "more-info" - which top-level page is shown.
            dcc.Store(id="active-page-store", data="recommendation"),

            build_sidebar(),

            html.Div(
                [
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
                                            build_predictions_column(),
                                        ],
                                        className="content-grid",
                                    ),
                                ],
                            ),
                        ],
                        className="content-grid",
                    ),
                ],
                id="main-content",
                className="main-content",
            ),

            build_more_info_view(),
        ],
        className="app-shell",
    )
