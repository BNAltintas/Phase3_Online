from __future__ import annotations

from dash import dcc, html

from propofol.config import BOLUS_MGKG_BOUNDS
from propofol.more_info_content import MORE_INFO_SECTIONS, MORE_INFO_SUBTITLE, MORE_INFO_TITLE
from propofol.recommend_regimen2023 import (
    DEFAULT_PROPOFOL_CONC_MG_ML,
    DEFAULT_REMI_CONC_MCG_ML,
    MAP_ABS_MIN_TARGET,
    MAP_REL_FRAC_TARGET,
    REMI_INFUSION_MCGKGMIN_BOUNDS,
    TARGET_BIS_HIGH,
    TARGET_BIS_LOW,
)

# Default patient/medication scenario used the first time the Scenario
# Exploration page loads, and restored by "Restore original patient" -
# intentionally the same defaults as the Recommendation page's Input
# Parameters, since this page is a fully independent, self-contained
# what-if sandbox (its own patient, never the Recommendation page's).
SCENARIO_DEFAULT_PATIENT = {
    "age": 35, "sex": "male", "height": 170, "weight": 70, "sbp": 120, "dbp": 70,
}
SCENARIO_DEFAULT_PROPOFOL_MGKG = 1.5
SCENARIO_DEFAULT_REMI_MCGKGMIN = 0.10

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


def param_subsection_label(text: str, first: bool = False):
    """
    Build a full-width subsection label row (e.g. "Concentrations", "Target
    BIS") for use inside a single shared `.param-table`, so several groups
    of fields can share one PARAMETER/VALUE/SOURCE header instead of each
    repeating it. `first=True` (the group directly under the header row)
    omits the top border/margin the other groups use to separate
    themselves from the group above.
    """
    class_name = "param-subsection-label"
    if first:
        class_name += " param-subsection-label--first"
    return html.Div(text, className=class_name)


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


def _placeholder_card(
    text: str, card_id: str, stale: bool = False, loading: bool = False, hidden: bool = False,
):
    """
    Build a "no result to show yet" placeholder card - the initial "not
    run yet" message, the "inputs changed, re-run" stale message, or (with
    loading=True) a spinner + "generating..." message shown the instant
    "Run recommendation" is clicked, before the model finishes. Purely
    presentational: it carries no data and is only ever shown/hidden via
    its `style` prop by `update_result_visibility` in app.py.
    """
    class_name = "card result-placeholder"
    if stale:
        class_name += " result-placeholder--stale"
    if loading:
        class_name += " result-placeholder--loading"

    content = (
        [html.I(className="fa-solid fa-spinner fa-spin result-placeholder-spinner"), text]
        if loading
        else text
    )

    return html.Div(
        content,
        id=card_id,
        className=class_name,
        style={"display": "none"} if hidden else None,
    )


def build_patient_parameters_card():
    """
    Build the card containing patient, baseline vitals, and derived value
    inputs.

    Patient parameters and Baseline vitals share a single
    PARAMETER/VALUE/SOURCE/DATE-TIME table instead of each repeating that
    header. The Age/Height/Weight/Sex rows sit directly under the header
    (no subsection label - "Patient parameters" is already the card's own
    title just above), and a single param_subsection_label("Baseline
    vitals") row marks where that second group starts within the same
    shared `.param-table` grid, mirroring the Model Settings card's
    layout. Derived values (MAP, Baseline PP) are calculated, read-only
    figures with no Source/Date-Time of their own, so they keep their
    existing simpler label/value layout rather than joining the grid.
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
                    param_subsection_label("Baseline vitals"),
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
            html.H4("Derived values", className="card-subheading card-subheading--divider"),
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
    Build the card containing opiate/concentration selection and read-only
    model targets.

    Concentrations, Target BIS, and Target MAP all share a single
    PARAMETER/VALUE/SOURCE table instead of each repeating that header -
    param_subsection_label() rows (spanning the full grid width, see
    .param-subsection-label in style.css) mark where each group starts
    within the one shared `.param-table` grid.
    """
    return html.Div(
        [
            html.H4("Model settings", className="card-subheading"),
            field_block(
                "Select opiate",
                html.Div(
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
                    className="opiate-dropdown-wrapper",
                ),
            ),
            html.Div(
                [
                    *param_table_header(show_date=False),
                    param_subsection_label("Concentrations", first=True),
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
                    param_subsection_label("Target BIS"),
                    *editable_field_row(
                        "Lower", "bis-target-low", TARGET_BIS_LOW, TARGET_BIS_LOW,
                        min_value=0, step=1, source_label="Default", show_date=False,
                    ),
                    *editable_field_row(
                        "Upper", "bis-target-high", TARGET_BIS_HIGH, TARGET_BIS_HIGH,
                        min_value=0, step=1, source_label="Default", show_date=False,
                    ),
                    param_subsection_label("Target MAP"),
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

    "Recommendation", "Scenario Exploration", and "Model Info" switch
    between the three pages. Each item's icon is purely presentational (a
    Font Awesome glyph + the existing label text) - none of the ids,
    n_clicks, or className-driven active/disabled styling changed, so
    navigation and active-page highlighting work exactly as before.
    """
    return html.Div(
        [
            html.Div("DosePilot", className="sidebar-brand"),
            html.Div(
                [
                    html.Div(
                        [
                            html.I(className="fa-solid fa-syringe sidebar-nav-icon"),
                            "Recommendation",
                        ],
                        id="nav-recommendation-btn",
                        n_clicks=0,
                        className="sidebar-nav-item sidebar-nav-item--active",
                    ),
                    html.Div(
                        [
                            html.I(className="fa-solid fa-chart-line sidebar-nav-icon"),
                            "Scenario Exploration",
                        ],
                        id="nav-scenario-btn",
                        n_clicks=0,
                        className="sidebar-nav-item",
                    ),
                    html.Div(
                        [
                            html.I(className="fa-solid fa-circle-info sidebar-nav-icon"),
                            "Model Info",
                        ],
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
    no ids, no callbacks, no effect on model inputs or data flow. Uses a
    Font Awesome glyph (already loaded for the sidebar icons) instead of
    an emoji so it can be recolored via CSS.
    """
    return html.Div(
        [
            html.I(className="fa-solid fa-circle-user patient-id-icon"),
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
                [
                    html.I(className="fa-solid fa-play run-btn-play-icon"),
                    html.I(className="fa-solid fa-spinner fa-spin run-btn-spinner"),
                    html.Span("Run recommendation", id="run-btn-label"),
                ],
                id="run-btn",
                n_clicks=0,
                className="run-recommendation-btn",
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

    Before a recommendation has ever been run, while one is actively being
    generated (the instant "Run recommendation" is clicked), and whenever
    the current recommendation is stale (inputs changed since it was
    generated), `summary-output` and the rationale graph card are hidden
    behind a placeholder card instead - see `update_result_visibility` in
    app.py, which is the sole owner of all of these `style` props.
    """
    return html.Div(
        [
            html.Div("RECOMMENDATION", className="section-label"),
            _placeholder_card(
                "Run recommendation to generate personalized dosing guidance.",
                "recommendation-empty-placeholder",
            ),
            _placeholder_card(
                "Generating personalized recommendation...",
                "recommendation-loading-placeholder", loading=True, hidden=True,
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
                "Generating personalized recommendation...",
                "rationale-loading-placeholder", loading=True, hidden=True,
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

    Its header also carries a "Learn more about the model" link, placed
    here so it sits in the upper-right of the main dashboard (this is the
    rightmost column) without needing a separate top bar above the whole
    3-column layout. Clicking it drives the same active-page-store switch
    as the sidebar's "More Info" item (see set_active_page in app.py) - it
    does not reload the page or touch any recommendation/input state.
    """
    return html.Div(
        [
            html.Div(
                [
                    html.Div("PREDICTIONS", className="section-label"),
                    html.Button(
                        ["Learn more about the model", html.Span(" ↗", className="learn-more-link-icon")],
                        id="learn-more-link-btn",
                        n_clicks=0,
                        className="learn-more-link",
                    ),
                ],
                className="predictions-header-row",
            ),
            _placeholder_card(
                "Predictions will appear after running recommendation.",
                "predictions-empty-placeholder",
            ),
            _placeholder_card(
                "Simulating BIS and MAP predictions...",
                "predictions-loading-placeholder", loading=True, hidden=True,
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


# ============================================================
# Scenario Exploration page
#
# A fully independent, self-contained "what-if" sandbox: its own patient,
# own targets, own opioid/dose scenario, never the Recommendation page's
# recommendation-store/manual-scenario-store. It reuses the same `.card`,
# `graph_card()`, `field_block()`, and `derived_value_row()` building
# blocks the Recommendation page uses, so the two pages share a visual
# language without sharing any data or callback wiring.
# ============================================================

def _scenario_summary_row(label: str, value_id: str, color_class: str):
    """
    Build one "Explored scenario" summary row - the same unboxed
    label/value layout as derived_value_row(), minus the info-icon tooltip
    (these are live scenario inputs/outputs, not derivation formulas), with
    a color class so propofol/remifentanil/recommendation values are each
    visually distinct.
    """
    return html.Div(
        [
            html.Span(label, className="derived-label"),
            html.Span(id=value_id, children="-", className=f"derived-value {color_class}"),
        ],
        className="derived-value-row",
    )


def _compact_row(label_children, component, extra_class: str = ""):
    """
    Build one compact "label left, small control right" row for the
    Scenario Exploration page. Unlike field_block() (label stacked above a
    full-width control, used by the Recommendation page's patient fields),
    this keeps every Patient/Targets row to a single short line so the
    whole left column stays compact - no sliders, no stepper buttons.
    """
    class_name = "scenario-compact-row"
    if extra_class:
        class_name += f" {extra_class}"
    return html.Div(
        [html.Div(label_children, className="scenario-compact-label"), component],
        className=class_name,
    )


def _compact_input(input_id: str, value, extra_class: str = ""):
    """
    A small, right-aligned text input for one compact row. type="text" (not
    "number") deliberately avoids the browser/Dash-rendered +/- stepper
    buttons - values are still parsed as floats server-side in app.py,
    exactly like every other text-typed draft input in this app already.
    """
    class_name = "scenario-compact-input"
    if extra_class:
        class_name += f" {extra_class}"
    return dcc.Input(id=input_id, type="text", value=value, debounce=True, className=class_name)


def _compact_derived_row(label: str, value_id: str, tooltip: str):
    """
    A compact, read-only derived-value row - same info-icon markup/style as
    derived_value_row() on the Recommendation page, just laid out as a
    label-left/value-right row (a small shaded box) instead of an unboxed
    label/value pair, to match this page's compact row rhythm.
    """
    return _compact_row(
        html.Span([label, html.Span("i", title=tooltip, className="info-icon")], className="derived-label"),
        html.Div(id=value_id, children="-", className="scenario-compact-input scenario-compact-input--readonly"),
    )


def build_scenario_patient_card():
    """
    Build the "Patient scenario" card: age/sex/height/weight, SBP/DBP
    baseline vitals, and derived MAP/PP - a simplified, always-editable
    version of the Recommendation page's Patient Parameters card (compact
    label-left/value-right rows instead of click-to-edit popovers or
    stacked full-width fields, no Source/Date-Time columns), since this
    page has no EHR-sourced values to distinguish from manual edits.
    `debounce=True` on every input defers firing its `value` change until
    blur/Enter, so the (expensive, Powell-optimized) scenario
    recommendation isn't recomputed on every keystroke.
    """
    defaults = SCENARIO_DEFAULT_PATIENT
    return html.Div(
        [
            html.H4("Patient scenario", className="card-subheading"),
            param_subsection_label("Patient characteristics", first=True),
            _compact_row("Age (years)", _compact_input("se-age", defaults["age"])),
            _compact_row(
                "Sex",
                dcc.Dropdown(
                    id="se-sex",
                    options=[
                        {"label": "Male", "value": "male"},
                        {"label": "Female", "value": "female"},
                    ],
                    value=defaults["sex"], clearable=False, searchable=False,
                    className="scenario-compact-dropdown",
                ),
            ),
            _compact_row("Height (cm)", _compact_input("se-height", defaults["height"])),
            _compact_row("Weight (kg)", _compact_input("se-weight", defaults["weight"])),
            html.Div("Baseline vitals", className="param-subsection-label"),
            _compact_row("SBP (mmHg)", _compact_input("se-sbp", defaults["sbp"])),
            _compact_row("DBP (mmHg)", _compact_input("se-dbp", defaults["dbp"])),
            html.Div("Derived values", className="param-subsection-label"),
            _compact_derived_row("MAP (mmHg)", "se-derived-map", "MAP = DBP + (SBP − DBP) / 3"),
            _compact_derived_row("Baseline PP (mmHg)", "se-derived-pp", "Baseline PP = SBP − DBP"),
            html.Button(
                [html.I(className="fa-solid fa-rotate-left"), "Restore original patient"],
                id="se-restore-patient-btn", n_clicks=0,
                className="scenario-secondary-btn",
            ),
        ],
        className="card",
    )


def _compact_slider_control(
    label: str,
    value_input_id: str,
    slider_id: str,
    value,
    unit: str,
    bounds: tuple[float, float],
    step: float,
    drug: str,
    wrapper_id: str | None = None,
):
    """
    Build one compact dose/rate control: a label, a small editable numeric
    field (kept in sync with the slider by _register_scenario_slider_sync
    in app.py) plus its unit suffix centered above a slim slider with
    min/max marks. Propofol and remifentanil both use this exact same
    structure - only `drug` ("propofol"/"remifentanil", which selects blue
    vs. purple) and bounds/unit differ - so "compact styling" is shared by
    construction, not just visually similar.
    """
    value_class = f"scenario-slider-value--{drug}"
    slider_class = f"scenario-slider scenario-slider--{drug}"
    control = html.Div(
        [
            html.Div(
                [
                    dcc.Input(
                        id=value_input_id, type="text", value=f"{value:.2f}", debounce=True,
                        className=f"scenario-slider-value-input {value_class}",
                    ),
                    html.Span(unit, className=f"scenario-slider-unit {value_class}"),
                ],
                className="scenario-slider-value-row",
            ),
            dcc.Slider(
                id=slider_id,
                min=bounds[0], max=bounds[1], step=step, value=value,
                marks={bounds[0]: f"{bounds[0]:.2f}", bounds[1]: f"{bounds[1]:.2f}"},
                tooltip={"placement": "bottom", "always_visible": False},
                updatemode="drag",
                allow_direct_input=False,
                className=slider_class,
            ),
        ],
        className="scenario-slider-block",
    )
    field = field_block(label, control)
    return html.Div(field, id=wrapper_id) if wrapper_id else field


def build_scenario_medication_card():
    """
    Build the "Medication scenario" card: opioid choice, propofol induction
    dose, and (when an opioid is selected) its infusion rate - the two
    dose/rate sliders (and their synced numeric inputs) drive a cheap
    single simulate_regimen() call per change (see render_scenario_page in
    app.py), never the Powell optimizer, so they can update live while
    dragging or typing.
    """
    return html.Div(
        [
            html.H4("Medication scenario", className="card-subheading"),
            field_block(
                "Opioid",
                html.Div(
                    dcc.Dropdown(
                        id="se-opioid-dropdown",
                        options=[
                            {"label": "No opiate / propofol only", "value": "none"},
                            {"label": "Remifentanil", "value": "remifentanil"},
                            {"label": "Sufentanil (upcoming)", "value": "sufentanil", "disabled": True},
                            {"label": "Fentanyl (upcoming)", "value": "fentanyl", "disabled": True},
                        ],
                        value="remifentanil", clearable=False, searchable=False,
                        style=DROPDOWN_STYLE,
                    ),
                    className="opiate-dropdown-wrapper",
                ),
            ),
            _compact_slider_control(
                "Infusion rate (µg/kg/min)",
                "se-remi-rate-input", "se-remi-rate-slider",
                SCENARIO_DEFAULT_REMI_MCGKGMIN, "µg/kg/min",
                REMI_INFUSION_MCGKGMIN_BOUNDS, 0.01,
                "remifentanil",
                wrapper_id="se-remi-rate-block",
            ),
            _compact_slider_control(
                "Propofol induction dose (mg/kg)",
                "se-propofol-dose-input", "se-propofol-dose-slider",
                SCENARIO_DEFAULT_PROPOFOL_MGKG, "mg/kg",
                BOLUS_MGKG_BOUNDS, 0.05,
                "propofol",
            ),
            html.Button(
                [html.I(className="fa-solid fa-rotate-left"), "Reset to recommended"],
                id="se-reset-medication-btn", n_clicks=0,
                className="scenario-secondary-btn",
            ),
        ],
        className="card",
    )


def build_scenario_targets_card():
    """
    Build the "Targets" card - the same BIS/MAP target clinical validation
    rules as the Recommendation page's Model Settings card (see
    FIELD_RULES in app.py, reused unchanged for these fields), laid out
    compactly: BIS lower/upper share one row, and MAP target is a single
    mode dropdown (Absolute/Relative) + one value, rather than two
    always-shown fields - simpler for a quick what-if scenario, still
    backed by the exact same target math (see compute_scenario_baseline).
    """
    return html.Div(
        [
            html.H4("Targets", className="card-subheading"),
            html.Div("Target BIS", className="param-subsection-label param-subsection-label--first"),
            html.Div(
                [
                    _compact_input("se-bis-target-low", TARGET_BIS_LOW, extra_class="scenario-range-input"),
                    html.Span("-", className="scenario-range-sep"),
                    _compact_input("se-bis-target-high", TARGET_BIS_HIGH, extra_class="scenario-range-input"),
                ],
                className="scenario-range-row",
            ),
            html.Div(id="se-bis-target-message", className="field-message"),
            html.Div("Target MAP", className="param-subsection-label"),
            html.Div(
                [
                    dcc.Dropdown(
                        id="se-map-target-mode",
                        options=[
                            {"label": "Absolute (mmHg)", "value": "abs"},
                            {"label": "Relative (% baseline)", "value": "rel"},
                        ],
                        value="abs", clearable=False, searchable=False,
                        className="scenario-map-mode-dropdown",
                    ),
                    _compact_input("se-map-target-value", MAP_ABS_MIN_TARGET, extra_class="scenario-range-input"),
                ],
                className="scenario-range-row",
            ),
            html.Div(id="se-map-target-message", className="field-message"),
        ],
        className="card",
    )


def build_scenario_input_column():
    """Build the Scenario Exploration page's left column."""
    return html.Div(
        [
            html.Div("PATIENT SCENARIO", className="section-label"),
            build_scenario_patient_card(),
            html.Div("MEDICATION SCENARIO", className="section-label"),
            build_scenario_medication_card(),
            html.Div("TARGETS", className="section-label"),
            build_scenario_targets_card(),
        ],
        className="input-column",
    )


def build_scenario_summary_card():
    """Build the "Explored scenario" summary card at the top of the center column."""
    return html.Div(
        [
            html.H4("Explored scenario", className="card-subheading"),
            html.Div(
                [
                    _scenario_summary_row(
                        "Propofol induction dose", "se-summary-propofol", "scenario-value--propofol",
                    ),
                    _scenario_summary_row(
                        "Remifentanil infusion rate", "se-summary-remi", "scenario-value--remifentanil",
                    ),
                    _scenario_summary_row(
                        "Scenario recommendation", "se-summary-recommendation", "scenario-value--recommended",
                    ),
                ],
                className="derived-values",
            ),
        ],
        className="card",
    )


def build_scenario_center_column():
    """
    Build the Scenario Exploration page's center column: the summary card
    plus the dose-response interaction graph (a relabeled reuse of the
    Recommendation page's induction-dose-rationale sweep - same function,
    same styling, see make_induction_dose_rationale_figure in app.py).
    """
    return html.Div(
        [
            html.Div("EXPLORED SCENARIO", className="section-label"),
            build_scenario_summary_card(),
            graph_card(
                "Dose-response interaction",
                "se-dose-response-graph",
                note="How propofol dose affects MAP and BIS - with remifentanil",
                tall=True,
            ),
        ],
        className="recommendation-column",
    )


def build_scenario_predictions_column():
    """
    Build the Scenario Exploration page's right column: the same 4
    prediction graph cards as the Recommendation page's Predictions column
    (BIS, MAP, Propofol PK/PD, Remifentanil PK), all expanded by default.
    Each card's "Show" checkbox is handled by the same generic
    toggle_graph_visibility pattern-matching callback in app.py that
    already drives every other graph card - no new visibility code needed.
    """
    return html.Div(
        [
            html.Div("PREDICTIONS", className="section-label"),
            graph_card("Predicted BIS", "se-bis-graph", default_visible=True),
            graph_card("Predicted MAP", "se-map-graph", default_visible=True),
            graph_card("Propofol PK/PD", "se-propofol-pk-graph", default_visible=True),
            graph_card("Remifentanil PK", "se-remifentanil-pk-graph", default_visible=True),
        ],
        className="predictions-column",
    )


def build_scenario_exploration_page():
    """
    Build the Scenario Exploration page - a sibling of main-content and
    more-info-view, shown/hidden the same way (display:none toggling owned
    by render_active_page in app.py).
    """
    return html.Div(
        html.Div(
            [
                build_scenario_input_column(),
                html.Div(
                    [
                        build_scenario_center_column(),
                        build_scenario_predictions_column(),
                    ],
                    className="content-grid",
                ),
            ],
            className="content-grid",
        ),
        id="scenario-exploration-view",
        className="main-content",
        style={"display": "none"},
    )


def build_layout():
    """
    Build the main layout of the dashboard: a navigation sidebar plus a
    three-column input/recommendation/predictions dashboard, the Scenario
    Exploration page, or the "More Info" page - the three views are
    siblings, and only one is ever visible at a time (Output("main-content",
    "style") / Output("scenario-exploration-view", "style") /
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
            # True from the instant "Run recommendation" is clicked until
            # recommendation-store's data actually updates - lets
            # update_result_visibility show a "generating..." placeholder
            # immediately, well before run_model's computation finishes.
            # Set/cleared by two clientside callbacks in app.py so the
            # "start" transition has zero server round-trip latency.
            dcc.Store(id="recommendation-loading-store", data=False),
            # "recommendation", "scenario-exploration", or "more-info" - which
            # top-level page is shown.
            dcc.Store(id="active-page-store", data="recommendation"),
            # Holds the Scenario Exploration page's own computed baseline
            # recommendation (context + result) - entirely separate from
            # recommendation-store, since this page never touches the
            # Recommendation page's patient/model state.
            dcc.Store(id="se-baseline-store", data=None),

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

            build_scenario_exploration_page(),
            build_more_info_view(),
        ],
        className="app-shell",
    )
