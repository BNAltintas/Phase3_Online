from __future__ import annotations

from urllib.parse import quote

from dash import dcc, html

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

# ============================================================
# Thin-stroke header icons (Recommendation page's own Patient Case/Input
# Parameters/Model Settings headers only)
#
# Font Awesome Free (the icon set every other icon in this app uses) has
# no regular/outline glyph at all for "sliders" or "gear" - rendering
# fa-regular fa-sliders/fa-gear produces an empty/missing-glyph box, not a
# thinner version of the icon. Rather than substitute a semantically
# unrelated Font Awesome icon (a clipboard, a pencil) for these two, these
# three icons are hand-built inline SVGs in the same Feather/Lucide-style
# "thin outline" formula (24x24 viewBox, 1.8 stroke, round caps/joins) so
# all three headers share one consistent, deliberately-chosen icon family
# instead of two different icon sets bolted together. Rendered as
# html.Img data-URIs (dash.html has no native <svg>/<path> components in
# the installed Dash version) - the stroke color is baked into the SVG
# markup itself for the same reason (a data-URI image can't read the
# page's CSS custom properties), matching --blue-accent's #2f6fed exactly.
# ============================================================

_HEADER_ICON_STROKE = "#2f6fed"

_HEADER_ICON_PATHS = {
    "user": (
        '<circle cx="12" cy="7.5" r="3.8"/>'
        '<path d="M4.5 20c0-4.2 3.4-6.8 7.5-6.8s7.5 2.6 7.5 6.8"/>'
    ),
    "sliders": (
        '<line x1="4" y1="6" x2="20" y2="6"/>'
        '<circle cx="9" cy="6" r="2.1" fill="#ffffff"/>'
        '<line x1="4" y1="12" x2="20" y2="12"/>'
        '<circle cx="15" cy="12" r="2.1" fill="#ffffff"/>'
        '<line x1="4" y1="18" x2="20" y2="18"/>'
        '<circle cx="9" cy="18" r="2.1" fill="#ffffff"/>'
    ),
    "gear": (
        '<path d="M12 15a3 3 0 100-6 3 3 0 000 6z"/>'
        '<path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 11-2.83 2.83l-.06-.06a1.65 1.65 0 00-1.82-.33 '
        '1.65 1.65 0 00-1 1.51V21a2 2 0 11-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 '
        '11-2.83-2.83l.06-.06a1.65 1.65 0 00.33-1.82 1.65 1.65 0 00-1.51-1H3a2 2 0 110-4h.09A1.65 1.65 0 004.6 '
        '9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 112.83-2.83l.06.06a1.65 1.65 0 001.82.33H9a1.65 1.65 0 001-1.51'
        'V3a2 2 0 114 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 112.83 2.83l-.06.06a1.65 '
        '1.65 0 00-.33 1.82V9a1.65 1.65 0 001.51 1H21a2 2 0 110 4h-.09a1.65 1.65 0 00-1.51 1z"/>'
    ),
}


def _header_icon(name: str, class_name: str, size: int = 18):
    """
    Build one of the hand-drawn thin-outline header icons above as an
    html.Img data-URI - see the module-level comment for why these three
    are custom SVG rather than Font Awesome like every other icon in the
    app. stroke-width 2.0 (up from an earlier 1.8) keeps the icon reading
    at least as visually heavy as the semibold header text next to it,
    now that both are rendered noticeably larger.
    """
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{_HEADER_ICON_STROKE}" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round">{_HEADER_ICON_PATHS[name]}</svg>'
    )
    return html.Img(src=f"data:image/svg+xml,{quote(svg)}", className=class_name)


# Default patient/medication scenario used the first time the Scenario
# Exploration page loads, and restored by "Restore original patient" - its
# own independent, self-contained what-if sandbox default (never the
# Recommendation page's own patient/inputs). Historically chosen to match
# the Recommendation page's own (pre-Test-Patient-1-default) input
# defaults; the Recommendation page itself now defaults to
# PRESET_TEST_PATIENTS["1"] instead (see build_patient_parameters_card),
# so this dict is retained unchanged as Scenario Exploration's own
# independent starting point, not a mirror of the Recommendation page.
SCENARIO_DEFAULT_PATIENT = {
    "age": 35, "sex": "male", "height": 170, "weight": 70, "sbp": 120, "dbp": 70,
}
SCENARIO_DEFAULT_REMI_MCGKGMIN = 0.10

# The 3 fixed Phase 3 test-patient presets - single source of truth for:
#   - the Recommendation page's own "Select patient" dropdown (default
#     patient on load, and re-applied whenever the dropdown changes;
#     see build_patient_id_card/apply_test_patient in app.py),
#   - the Precomputed Remi page's "Select patient" dropdown (PR_CASE_
#     OPTIONS' labels below reuse these same "1"/"2"/"3" keys), and
#   - scripts/precompute_remi_cases.py's own offline precompute run
#     (imports this dict from propofol.app, which re-exports it from
#     here).
# Defined here (not in app.py) so dashboard_layout.py's own layout-
# building functions (build_patient_parameters_card, in particular) can
# use it directly without a circular import - app.py imports
# dashboard_layout.py already, never the other way around.
PRESET_TEST_PATIENTS = {
    "1": {
        "label": "Test Patient 1",
        "age": 35, "sex": "male", "weight": 72, "height": 180,
        "baseline_sap": 130, "baseline_dap": 75, "baseline_hr": 72,
    },
    "2": {
        "label": "Test Patient 2",
        "age": 67, "sex": "male", "weight": 90, "height": 170,
        "baseline_sap": 180, "baseline_dap": 98, "baseline_hr": 78,
    },
    "3": {
        "label": "Test Patient 3",
        "age": 79, "sex": "female", "weight": 54, "height": 165,
        "baseline_sap": 160, "baseline_dap": 90, "baseline_hr": 84,
    },
}

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
    """Build the Parameter / Value / Source (/ Last Updated) header row for a `.param-table`."""
    header = [
        html.Div("Parameter", className="param-table-header"),
        html.Div("Value", className="param-table-header"),
        html.Div("Source", className="param-table-header"),
    ]
    if show_date:
        header.append(html.Div("Last Updated", className="param-table-header"))
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
    compact: bool = False,
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

    compact=True (used for Target BIS/Target MAP's paired side-by-side
    rows, see .param-pair-row) additionally hides the Source cell the
    same way - still rendered, never removed, just display:none - so two
    fields' Label+Value cells alone can sit side by side.
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
        html.Div(
            source_label, id=f"{field_id}-source", className="param-source",
            style={"display": "none"} if compact else None,
        ),
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
                            id=f"{field_id}-original-label",
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

    Every default value below comes from PRESET_TEST_PATIENTS["1"] (the
    same dict the "Select patient" dropdown and the precompute script
    use) - the Recommendation page now defaults to Test Patient 1 on
    first load, matching the "Select patient" dropdown's own default
    value of "1" (see build_patient_id_card), instead of the page's
    former standalone default patient.
    """
    default_patient = PRESET_TEST_PATIENTS["1"]
    return html.Div(
        [
            html.H4("Patient Parameters", className="card-subheading"),
            html.Div(
                [
                    *param_table_header(),
                    *editable_field_row(
                        "Age (years)", "age", default_patient["age"], default_patient["age"],
                        min_value=0, step=1,
                    ),
                    *editable_field_row(
                        "Height (cm)", "height", default_patient["height"], default_patient["height"],
                        min_value=30, step=1,
                    ),
                    *editable_field_row(
                        "Weight (kg)", "weight", default_patient["weight"], default_patient["weight"],
                        min_value=0.5, step=0.1,
                    ),
                    html.Div("Sex", className="param-label"),
                    html.Div(
                        dcc.Dropdown(
                            id="sex-dropdown",
                            options=[
                                {"label": "Male", "value": "male"},
                                {"label": "Female", "value": "female"},
                            ],
                            value=default_patient["sex"],
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
                    param_subsection_label("Baseline Vitals"),
                    *editable_field_row(
                        "SBP (mmHg)", "baseline_sap",
                        default_patient["baseline_sap"], default_patient["baseline_sap"],
                        min_value=30, step=0.1, source_label="Monitor",
                    ),
                    *editable_field_row(
                        "DBP (mmHg)", "baseline_dap",
                        default_patient["baseline_dap"], default_patient["baseline_dap"],
                        min_value=10, step=0.1, source_label="Monitor",
                    ),
                    *editable_field_row(
                        "HR (bpm)", "baseline_hr",
                        default_patient["baseline_hr"], default_patient["baseline_hr"],
                        min_value=0, step=0.1, source_label="Monitor",
                    ),
                ],
                className="param-table",
            ),
            html.H4("Derived Values", className="card-subheading card-subheading--divider"),
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
            html.H4(
                [
                    _header_icon("gear", "card-subheading-icon"),
                    "Model Settings",
                ],
                className="card-subheading card-subheading--main",
            ),
            # Same label|dropdown row pattern _te_dropdown_field builds for
            # "Select patient" on the Patient Case card above (.te-field-row/
            # .te-field-label/.te-field-dropdown) rather than field_block's
            # stacked label-above-control layout, so the two selectors read
            # as one consistent component instead of two different ones.
            # opiate-dropdown-wrapper is kept alongside .te-field-dropdown
            # purely so this dropdown's own existing background/border
            # styling (see style.css) still applies - the id, options, and
            # value are all unchanged.
            html.Div(
                [
                    html.Div("Select opiate", className="te-field-label"),
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
                        ),
                        className="te-field-dropdown opiate-dropdown-wrapper",
                    ),
                ],
                className="te-field-row",
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
                    # Hidden (not removed) whenever "No opiate / propofol
                    # only" is selected - see toggle_remifentanil_concentration_row
                    # in app.py. The field itself, and its id, stay in the
                    # DOM unchanged either way, so run_model's own
                    # State("remifentanil-concentration", "value") always
                    # resolves. display:contents (via .param-row-group)
                    # when shown keeps its 5 cells as direct .param-table
                    # grid children, exactly as if this wrapper didn't
                    # exist; only the "display: none" override actually
                    # hides it.
                    html.Div(
                        editable_field_row(
                            "Remifentanil (µg/mL)", "remifentanil-concentration",
                            DEFAULT_REMI_CONC_MCG_ML, DEFAULT_REMI_CONC_MCG_ML,
                            min_value=0.1, step=0.1, source_label="Default", show_date=False,
                        ),
                        id="remifentanil-concentration-row",
                        className="param-row-group",
                        style={"display": "none"},
                    ),
                    param_subsection_label("Target BIS"),
                    # Lower/Upper side by side (Test Exploration's own
                    # compact Target BIS layout) instead of stacked full
                    # table rows - .param-pair-row is its own nested grid
                    # (grid-column: 1/-1 so it still reads as one
                    # full-width band within the outer .param-table), and
                    # compact=True hides each field's Source cell so only
                    # Label+Value show. Ids/values/callbacks are entirely
                    # unchanged - purely a layout change.
                    html.Div(
                        [
                            *editable_field_row(
                                "Lower", "bis-target-low", TARGET_BIS_LOW, TARGET_BIS_LOW,
                                min_value=0, step=1, source_label="Default", show_date=False,
                                compact=True,
                            ),
                            *editable_field_row(
                                "Upper", "bis-target-high", TARGET_BIS_HIGH, TARGET_BIS_HIGH,
                                min_value=0, step=1, source_label="Default", show_date=False,
                                compact=True,
                            ),
                        ],
                        className="param-pair-row",
                    ),
                    param_subsection_label("Target MAP"),
                    # Absolute/Relative side by side, same technique as
                    # Target BIS above. Both fields stay simultaneously
                    # live model inputs exactly as before (the model
                    # combines them via map_lower_bound_from_baseline =
                    # max(absolute, relative% x baseline MAP) - there is
                    # no mode-toggle in the real model here, unlike Test
                    # Exploration's own fake map_mode, so this is a
                    # side-by-side pairing of both real fields rather than
                    # a single-value-with-mode-dropdown control.
                    html.Div(
                        [
                            *editable_field_row(
                                "Absolute (mmHg)", "map-target-abs",
                                MAP_ABS_MIN_TARGET, MAP_ABS_MIN_TARGET,
                                min_value=30, step=1, source_label="Default", show_date=False,
                                compact=True,
                            ),
                            *editable_field_row(
                                "Relative (% baseline)", "map-target-rel",
                                MAP_REL_FRAC_TARGET * 100, MAP_REL_FRAC_TARGET * 100,
                                min_value=30, step=1, source_label="Default", show_date=False,
                                compact=True,
                            ),
                        ],
                        className="param-pair-row",
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
                            html.I(className="fa-solid fa-flask sidebar-nav-icon"),
                            "Test Exploration",
                        ],
                        id="nav-test-exploration-btn",
                        n_clicks=0,
                        className="sidebar-nav-item",
                    ),
                    html.Div(
                        [
                            html.I(className="fa-solid fa-bolt sidebar-nav-icon"),
                            "Precomputed Remi",
                        ],
                        id="nav-precomputed-remi-btn",
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
    Build the "Patient case" card shown above Input Parameters - visually
    identical to the Precomputed Remi page's own Patient Case card
    (_te_card_header + _te_dropdown_field, the same "card te-card"
    classes), reusing PR_CASE_OPTIONS directly so both pages' dropdowns
    list the exact same 3 presets with the exact same labels. Defaults to
    "1" (Test Patient 1), matching build_patient_parameters_card's own
    default input values below, so the dropdown and the fields it drives
    agree on first load without any callback needing to fire.

    Selecting a different patient here is handled by one callback in
    app.py (apply_test_patient) - this function only builds static
    markup.
    """
    return html.Div(
        [
            _te_card_header(None, "Patient Case", icon=_header_icon("user", "te-card-icon")),
            _te_dropdown_field("Select patient", "rec-patient-dropdown", PR_CASE_OPTIONS, "1"),
        ],
        className="card te-card",
    )


def build_input_column():
    """
    Build the left dashboard column: patient ID card, patient/model inputs,
    and the run button.
    """
    return html.Div(
        [
            build_patient_id_card(),
            html.Div(
                [
                    _header_icon("sliders", "section-label-icon"),
                    "INPUT PARAMETERS",
                ],
                className="section-label",
            ),
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
    min/max marks.

    updatemode="mouseup" (not "drag"): this slider's value now feeds
    compute_scenario_baseline, a multi-second Powell re-optimization, not a
    cheap live simulate - firing on every in-flight drag position would
    queue up many redundant expensive recomputations, so it only reports
    once the user releases the handle (the paired compact input still
    commits on blur/Enter via its own debounce=True).
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
                updatemode="mouseup",
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
    Build the "Medication scenario" card: opioid choice and (when
    remifentanil is selected) its infusion rate - the ONLY adjustable
    medication parameter on this page. Propofol is deliberately not
    user-editable here: this page explores opioid strategy, and the
    propofol induction dose/maintenance are outputs the model recomputes
    in response (see compute_scenario_baseline in app.py, which
    re-optimizes propofol for whatever remifentanil rate is set via
    recommend_propofol_for_fixed_remifentanil_rate), never a second input
    the user drags independently.
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
    """
    Build the "Explored scenario" card - the recommendation's own reference
    values only (opioid strategy, recommended propofol induction dose,
    recommended opioid maintenance, recommended propofol maintenance), not
    graphs and not the live slider values (those are already visible on
    the sliders/inputs in the left column). Populated by
    render_scenario_recommendation_summary in app.py, which only depends
    on se-baseline-store - it never needs to re-run when the dose/rate
    sliders move, since none of these four values come from them.
    """
    return html.Div(
        [
            html.H4("Explored scenario", className="card-subheading"),
            html.Div(
                [
                    _scenario_summary_row(
                        "Current opioid strategy", "se-summary-opioid-strategy", "",
                    ),
                    _scenario_summary_row(
                        "Recommended propofol induction dose",
                        "se-summary-rec-propofol-induction", "scenario-value--recommended",
                    ),
                    _scenario_summary_row(
                        "Recommended opioid maintenance",
                        "se-summary-rec-opioid-maintenance", "scenario-value--recommended",
                    ),
                    _scenario_summary_row(
                        "Recommended propofol maintenance",
                        "se-summary-rec-propofol-maintenance", "scenario-value--recommended",
                    ),
                ],
                className="derived-values",
            ),
        ],
        className="card",
    )


def build_scenario_center_column():
    """
    Build the Scenario Exploration page's center column: the "Explored
    scenario" recommendation-reference card, plus the dose-response
    interaction graph underneath (a relabeled reuse of the Recommendation
    page's induction-dose-rationale sweep - same function, same styling,
    see make_induction_dose_rationale_figure in app.py) at a reduced,
    secondary height - this column explains *why* the recommendation is
    what it is; the right column's BIS/MAP/PK predictions are the primary,
    decision-support output and always start at the same height as this
    column, never pushed down by it.
    """
    return html.Div(
        [
            html.Div("EXPLORED SCENARIO", className="section-label"),
            build_scenario_summary_card(),
            html.Div(
                graph_card(
                    "Dose-response interaction",
                    "se-dose-response-graph",
                    note="How propofol dose affects MAP and BIS - with remifentanil",
                    tall=True,
                ),
                id="se-dose-response-card-wrapper",
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

    The center+predictions wrapper uses "output-column" (the same class
    the Recommendation page's own center+predictions wrapper uses) - its
    flex:1 1 0%; min-width:0 is what lets that inner content-grid shrink to
    fit the remaining width instead of forcing the predictions column to
    wrap onto a second row below the center column, which is what a bare
    content-grid div (flex:0 1 auto, no min-width:0) was doing here.
    """
    return html.Div(
        html.Div(
            [
                build_scenario_input_column(),
                html.Div(
                    html.Div(
                        [
                            build_scenario_center_column(),
                            build_scenario_predictions_column(),
                        ],
                        className="content-grid",
                    ),
                    className="output-column",
                ),
            ],
            className="content-grid",
        ),
        id="scenario-exploration-view",
        className="main-content",
        style={"display": "none"},
    )


# ============================================================
# Test Exploration - a fully independent UI sandbox page.
#
# No PK/PD, no optimization, no recommendation logic, and no shared state
# with the Recommendation or Scenario Exploration pages: every id and
# default below is its own, and the only callbacks that touch them (see
# the "Test Exploration" section in app.py) are new, self-contained
# slider<->input syncs, a restore button, and plain MAP/PP arithmetic -
# nothing here ever calls into recommend_regimen2023 or any other model
# module.
# ============================================================

TEST_EXPLORATION_DEFAULT_PATIENT = {
    "age": 35, "sex": "male", "height": 170, "weight": 70, "sbp": 120, "dbp": 70,
}
TEST_EXPLORATION_DEFAULT_BIS_LOW = 40
TEST_EXPLORATION_DEFAULT_BIS_HIGH = 60
TEST_EXPLORATION_DEFAULT_MAP_ABS = 65
TEST_EXPLORATION_DEFAULT_MAP_REL = 70

# Per-opioid slider bounds/units for the "Explore opioid strategy" card -
# mirrored in test_exploration.js's own TE_OPIOID_CONFIG (the source of
# truth once the page is loaded; this Python copy only seeds the slider's
# initial render for the default "remifentanil" selection).
TEST_EXPLORATION_OPIOID_DEFAULTS = {
    "remifentanil": {"min": 0.02, "max": 0.20, "step": 0.01, "baseline": 0.10, "unit": "µg/kg/min"},
    "sufentanil": {"min": 2, "max": 15, "step": 0.5, "baseline": 6, "unit": "µg/h"},
    "fentanyl": {"min": 30, "max": 150, "step": 5, "baseline": 80, "unit": "µg/h"},
}


def _te_input_field(label: str, input_id: str, value):
    """
    Build one compact "label | numeric input" row - label on the left,
    a small light-blue-filled text input on the right, no slider. Range
    clamping is enforced standalone by _register_te_input_clamp in
    app.py (there is no slider left to provide it).
    """
    return html.Div(
        [
            html.Div(label, className="te-field-label"),
            dcc.Input(
                id=input_id, type="text", value=value, debounce=True,
                className="te-field-input-plain",
            ),
        ],
        className="te-field-row",
    )


def _te_derived_plain_row(label: str, value_id: str, tooltip: str):
    """
    A Test-Exploration-only derived-value row: plain text, no input-style
    box/border/background - deliberately NOT _compact_derived_row (shared
    with the Scenario Exploration page, which keeps its own shaded
    read-only box look unchanged).
    """
    return html.Div(
        [
            html.Span([label, html.Span("i", title=tooltip, className="info-icon")], className="te-derived-plain-label"),
            html.Span(id=value_id, children="-", className="te-derived-plain-value"),
        ],
        className="te-derived-plain-row",
    )


def _te_dropdown_field(label: str, dropdown_id: str, options: list, value, narrow: bool = False):
    """
    Build a "label | dropdown" row, column-aligned with _te_input_field's
    rows. `narrow=True` (Sex only) shrinks the dropdown to the same
    fixed width as the numeric input cells instead of its normal
    fill-the-row width, so it reads as a compact form field rather than
    a full-width control - the row's own justify-content:space-between
    then places it flush against the same right edge the numeric inputs
    already share, without needing any other layout change.
    """
    wrapper_class = "te-field-dropdown te-field-dropdown--narrow" if narrow else "te-field-dropdown"
    return html.Div(
        [
            html.Div(label, className="te-field-label"),
            html.Div(
                dcc.Dropdown(id=dropdown_id, options=options, value=value, clearable=False, searchable=False),
                className=wrapper_class,
            ),
        ],
        className="te-field-row",
    )


def _te_card_header(icon_class: str | None, title: str, icon: object = None):
    """
    Icon + all-caps title shown at the top of every Test Exploration card,
    with a divider beneath it. Every existing caller passes a Font Awesome
    icon_class string, unchanged. The Recommendation page's own Patient
    Case card instead passes icon=_header_icon(...) (a pre-built
    html.Img), with icon_class left as None, for the custom thin-outline
    look shared with Input Parameters/Model Settings - see the
    _header_icon comment above for why those three use hand-drawn SVG
    instead of a Font Awesome class.
    """
    return html.Div(
        [
            icon if icon is not None else html.I(className=f"fa-solid {icon_class} te-card-icon"),
            html.Span(title, className="te-card-title"),
        ],
        className="te-card-header",
    )


def build_te_patient_card():
    """
    Build the Test Exploration page's "Patient scenario" card - patient
    characteristics, baseline vitals, derived MAP/PP, and a restore
    button. Every numeric field is a plain light-blue text input (no
    slider) - range clamping is enforced standalone by
    _register_te_input_clamp in app.py.
    """
    defaults = TEST_EXPLORATION_DEFAULT_PATIENT
    return html.Div(
        [
            _te_card_header("fa-user", "Patient scenario"),
            param_subsection_label("Patient characteristics", first=True),
            _te_input_field("Age (years)", "te-age-input", defaults["age"]),
            _te_dropdown_field(
                "Sex", "te-sex",
                [{"label": "Male", "value": "male"}, {"label": "Female", "value": "female"}],
                defaults["sex"], narrow=True,
            ),
            _te_input_field("Height (cm)", "te-height-input", defaults["height"]),
            _te_input_field("Weight (kg)", "te-weight-input", defaults["weight"]),
            html.Div("Baseline vitals", className="param-subsection-label"),
            _te_input_field("SBP (mmHg)", "te-sbp-input", defaults["sbp"]),
            _te_input_field("DBP (mmHg)", "te-dbp-input", defaults["dbp"]),
            html.Div("Derived values", className="param-subsection-label"),
            _te_derived_plain_row("MAP (mmHg)", "te-derived-map", "MAP = DBP + (SBP − DBP) / 3"),
            _te_derived_plain_row("Baseline PP (mmHg)", "te-derived-pp", "Baseline PP = SBP − DBP"),
            html.Button(
                [html.I(className="fa-solid fa-rotate-left"), "Restore Original Patient"],
                id="te-restore-btn", n_clicks=0,
                className="scenario-secondary-btn",
            ),
        ],
        className="card te-card",
    )


def build_te_medication_card():
    """
    Build the Test Exploration page's "Medication scenario" card - an
    opioid dropdown with a "None" option (its value is read, client-side
    only, by the fake-logic clientside callbacks in test_exploration.js -
    no server callback of any kind is attached to it).
    """
    return html.Div(
        [
            _te_card_header("fa-pills", "Medication scenario"),
            _te_dropdown_field(
                "Select opioid", "te-opioid-dropdown",
                [
                    {"label": "None", "value": "none"},
                    {"label": "Remifentanil", "value": "remifentanil"},
                    {"label": "Sufentanil (future extension)", "value": "sufentanil", "disabled": True},
                    {"label": "Fentanyl (future extension)", "value": "fentanyl", "disabled": True},
                ],
                "none",
            ),
        ],
        className="card te-card",
    )


def build_te_targets_card():
    """
    Build the Test Exploration page's "Targets" card - BIS/MAP target
    fields, no validation. Reuses the Scenario Exploration page's own
    .scenario-compact-input/.scenario-range-*/.scenario-map-mode-dropdown
    layout classes unchanged (row/column sizing, spacing, dropdown's own
    neutral white look), adding only the TE-only "te-target-input" class
    on top of the two numeric inputs (BIS low/high, MAP value) so they
    pick up the same light-blue "editable" look Patient Scenario's own
    inputs use - the Target MAP dropdown deliberately keeps its plain
    white styling, same as Sex/Select opioid, so dropdowns (selection
    controls) stay visually distinct from blue editable numeric cells.
    Switching te-map-target-mode resets te-map-target-value to that
    mode's own default (reset_te_map_target_value in app.py) - the same
    behavior the Scenario Exploration page's own Target MAP already has.
    """
    return html.Div(
        [
            _te_card_header("fa-bullseye", "Targets"),
            html.Div("Target BIS", className="param-subsection-label param-subsection-label--first"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Lower BIS target", className="field-label"),
                            dcc.Input(
                                id="te-bis-target-low", type="text",
                                value=TEST_EXPLORATION_DEFAULT_BIS_LOW, debounce=True,
                                className="scenario-compact-input te-target-input",
                            ),
                        ],
                        className="te-bis-field",
                    ),
                    html.Div(
                        [
                            html.Label("Upper BIS target", className="field-label"),
                            dcc.Input(
                                id="te-bis-target-high", type="text",
                                value=TEST_EXPLORATION_DEFAULT_BIS_HIGH, debounce=True,
                                className="scenario-compact-input te-target-input",
                            ),
                        ],
                        className="te-bis-field",
                    ),
                ],
                className="te-bis-row",
            ),
            html.Div("Target MAP", className="param-subsection-label"),
            html.Div(
                [
                    dcc.Dropdown(
                        id="te-map-target-mode",
                        options=[
                            {"label": "Absolute (mmHg)", "value": "abs"},
                            {"label": "Relative (% baseline)", "value": "rel"},
                        ],
                        value="abs", clearable=False, searchable=False,
                        className="scenario-map-mode-dropdown",
                    ),
                    _compact_input(
                        "te-map-target-value", TEST_EXPLORATION_DEFAULT_MAP_ABS,
                        extra_class="scenario-range-input te-target-input",
                    ),
                ],
                className="scenario-range-row",
            ),
        ],
        className="card te-card",
    )


def build_te_input_column():
    """
    Build the Test Exploration page's left column: Patient/Medication/
    Targets cards, plus the single "Run Scenario" button at the bottom of
    the form. Every field above this button is pure form state - editing
    age, sex, height, weight, SBP/DBP, opioid choice, or BIS/MAP targets
    never runs any computation. Clicking this button is the ONLY trigger
    for the runScenario clientside callback in test_exploration.js, which
    (re)computes the Scenario Result card, the Dose-Response graph, and
    all 4 prediction graphs from whatever the form currently holds.
    """
    return html.Div(
        [
            build_te_patient_card(),
            build_te_medication_card(),
            build_te_targets_card(),
            html.Button(
                "Run Scenario", id="te-run-scenario-btn", n_clicks=0,
                className="run-recommendation-btn te-run-scenario-btn",
            ),
        ],
        className="input-column",
    )


def build_te_explore_card():
    """
    Build the "Explore opioid strategy" card. Its own opioid dropdown
    (te-explore-opioid-dropdown) is completely independent from the left
    column's Medication Scenario dropdown (te-opioid-dropdown, which only
    feeds the Baseline Recommendation) - this lets the user explore a
    different opioid (or a different rate of the same one) without ever
    re-running the baseline. Selecting an opioid here reveals a rate
    slider (te-explore-slider-wrapper), reconfigured for that opioid by
    onExploreOpioidChange in test_exploration.js and initialized exactly
    at that opioid's own recommended rate - the Explored Scenario column,
    Change column, and every graph's orange trace stay empty/hidden until
    the slider is actually moved away from that initial value
    (updateExploredLive treats "still at the recommended value" as "no
    exploration yet").
    """
    return html.Div(
        [
            _te_card_header("fa-sliders", "Explore opioid strategy"),
            _te_dropdown_field(
                "Opioid to explore", "te-explore-opioid-dropdown",
                [
                    {"label": "None", "value": "none"},
                    {"label": "Remifentanil", "value": "remifentanil"},
                    {"label": "Sufentanil (future extension)", "value": "sufentanil", "disabled": True},
                    {"label": "Fentanyl (future extension)", "value": "fentanyl", "disabled": True},
                ],
                "none",
            ),
            html.Div(
                [
                    html.Div("", id="te-explore-rate-title", className="te-summary-label"),
                    html.Div("", id="te-explore-rate-label", className="te-explore-rate-value"),
                    dcc.Slider(
                        id="te-explore-rate-slider",
                        min=0, max=1, step=0.1, value=0,
                        marks={},
                        tooltip={"placement": "bottom", "always_visible": False},
                        updatemode="drag",
                        allow_direct_input=False,
                        className="scenario-slider te-explore-slider",
                    ),
                ],
                id="te-explore-slider-wrapper",
                style={"display": "none"},
            ),
        ],
        className="card te-card",
    )


def _te_result_header_row():
    """The Scenario Result grid's header row: column titles + a decorative arrow between Baseline Recommendation and Explored Scenario."""
    return html.Div(
        [
            html.Div(className="te-result-cell"),
            html.Div("Baseline Recommendation", className="te-result-cell te-result-col-heading te-result-col-heading--baseline"),
            html.Div("→", className="te-result-cell te-result-arrow-cell"),
            html.Div("Explored Scenario", className="te-result-cell te-result-col-heading te-result-col-heading--explored"),
            html.Div("Change", className="te-result-cell te-result-col-heading te-result-col-heading--change"),
        ],
        className="te-result-row te-result-row--header",
    )


def _te_result_section_header(title_id_or_text, subtitle: str, title_is_id: bool = False):
    """A full-width sub-header row (e.g. "Propofol – Early maintenance") spanning all 5 grid columns."""
    title_kwargs = {"id": title_id_or_text, "children": "-"} if title_is_id else {"children": title_id_or_text}
    return html.Div(
        [
            html.Div(className="te-maint-title", **title_kwargs),
            html.Div(subtitle, className="te-maint-subtitle"),
        ],
        className="te-result-section-header",
    )


def _te_result_data_row(label: str, baseline_children_or_id, explored_id: str, change_id: str | None = None):
    """
    One comparison row: [label] [baseline value] [arrow spacer] [explored
    value] [change]. `baseline_children_or_id` is either static text
    (e.g. "Pause", which never changes) or an id string (for the one
    baseline value that's actually computed, e.g. an mL/h rate) -
    distinguished by whether it's already a rendered value or needs its
    own id for runScenario to fill in.
    """
    baseline_cell = (
        html.Div(baseline_children_or_id, className="te-result-cell te-result-baseline-value")
        if not isinstance(baseline_children_or_id, dict)
        else html.Div("-", id=baseline_children_or_id["id"], className="te-result-cell te-result-baseline-value")
    )
    change_cell = (
        html.Div("–", className="te-result-cell te-result-change-value")
        if change_id is None
        else html.Div("–", id=change_id, className="te-result-cell te-result-change-value")
    )
    return html.Div(
        [
            html.Div(label, className="te-result-cell te-result-row-label"),
            baseline_cell,
            html.Div(className="te-result-cell te-result-arrow-cell"),
            html.Div("–", id=explored_id, className="te-result-cell te-result-explored-value"),
            change_cell,
        ],
        className="te-result-row",
    )


def build_te_scenario_result_card():
    """
    Build the "Scenario result" card - the middle column's primary
    comparison component. Integrates what used to be a separate "Baseline
    recommendation" card (propofol induction/maintenance + opioid
    maintenance, always visible, computed at the opioid's fixed baseline
    rate) side by side with the "Explored scenario" column (placeholder
    text/dashes until "Run Scenario" is pressed) and a "Change" column
    (delta indicators on every row with a real baseline/explored number -
    induction, both propofol maintenance rate rows, and the active-opioid-
    rate row - "Pause" rows have no rate to compare, so stay a plain "–").

    Every cell here starts as a plain "-"/"–" placeholder and is only
    ever filled in by runScenario in test_exploration.js, which fires
    solely on the page's "Run Scenario" button click (see
    build_te_input_column) - never live, never on page load, never from
    editing Patient/Medication/Targets alone.
    """
    return html.Div(
        [
            _te_card_header("fa-code-compare", "Scenario result"),
            html.P(
                "Compare how the explored scenario differs from the baseline recommendation.",
                className="te-card-subtitle",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Span("Baseline Strategy", className="te-strategy-label"),
                            html.Span("No opioid", id="te-strategy-baseline-value", className="te-strategy-value"),
                        ],
                        className="te-strategy-card te-strategy-card--baseline",
                    ),
                    html.Div(html.I(className="fa-solid fa-arrow-right"), className="te-strategy-arrow"),
                    html.Div(
                        [
                            html.Span("Explored Strategy", className="te-strategy-label"),
                            html.Span("Not selected", id="te-strategy-explored-value", className="te-strategy-value"),
                        ],
                        className="te-strategy-card te-strategy-card--explored",
                    ),
                ],
                className="te-strategy-banner",
            ),
            html.Div(
                [
                    _te_result_header_row(),
                    html.Div(
                        [
                            html.Div("Propofol – Induction", className="te-result-cell te-result-row-label"),
                            html.Div(
                                [
                                    html.Span("-", id="te-induction-dose", className="te-result-baseline-induction-value"),
                                    html.Span("-", id="te-induction-total", className="te-result-baseline-induction-total"),
                                ],
                                className="te-result-cell te-result-baseline-value",
                            ),
                            html.Div(className="te-result-cell te-result-arrow-cell"),
                            html.Div(
                                [
                                    html.Span(
                                        "Select an opioid to begin exploring alternative strategies.",
                                        id="te-explored-induction",
                                        className="te-result-explored-induction-value te-result-explored-value--placeholder",
                                    ),
                                    html.Span("", id="te-explored-induction-total", className="te-result-explored-induction-total"),
                                ],
                                className="te-result-cell te-result-explored-value",
                            ),
                            html.Div("–", id="te-result-delta-propofol", className="te-result-cell te-result-change-value"),
                        ],
                        className="te-result-row",
                    ),
                    _te_result_section_header(
                        "Propofol – Early maintenance (0–15 min)", "Infusion regimen relative to induction",
                    ),
                    _te_result_data_row("0–1 min", "Pause", "te-explored-prop-pause"),
                    _te_result_data_row(
                        "1–8 min", {"id": "te-baseline-prop-rate1"}, "te-explored-prop-rate1",
                        change_id="te-result-delta-prop-rate1",
                    ),
                    _te_result_data_row(
                        "8–15 min", {"id": "te-baseline-prop-rate2"}, "te-explored-prop-rate2",
                        change_id="te-result-delta-prop-rate2",
                    ),
                    html.Div(
                        [
                            _te_result_section_header(
                                "Selected Opioid Strategy", "Infusion regimen relative to induction",
                            ),
                            _te_result_data_row("0–2 min", "Pause", "te-explored-remi-pause"),
                            _te_result_data_row(
                                "2–15 min", {"id": "te-baseline-remi-rate"}, "te-explored-remi-rate",
                                change_id="te-result-delta-opioid",
                            ),
                        ],
                        id="te-opioid-maintenance-section",
                        className="te-result-grid-section",
                    ),
                ],
                className="te-result-grid",
            ),
        ],
        className="card te-card te-result-card",
    )


def build_te_dose_response_card():
    """
    Build the "Dose-response interaction" card - reuses graph_card() (the
    same generic card/graph/"Show"-toggle shell every other prediction
    graph in this app uses). No subtitle - the graph's own legend/axis
    labels already say which opioid and variable are being swept. The
    figure itself (swept across opioid infusion rate, not propofol dose)
    is built and refreshed entirely client-side by test_exploration.js.
    """
    card = graph_card(
        "Dose-Response Interaction",
        "te-dose-response-graph",
        tall=True,
    )
    return html.Div(
        [card],
        id="te-dose-response-card-wrapper",
    )


def build_te_center_column():
    """
    Build the Test Exploration page's center column: Scenario Result,
    Explore Opioid Strategy (always shown once a scenario is active,
    regardless of the baseline's own opioid choice - its own dropdown
    lets the user explore any opioid independently), then Dose-Response
    Interaction. Carries an extra "te-center-column" class (on top of the
    shared "recommendation-column" flex-item class every other page's
    center column also uses) so its width can be widened in style.css -
    scoped to this one class, never touching .recommendation-column
    itself - to give the Scenario Result grid enough room for its 5
    columns.
    """
    return html.Div(
        [
            build_te_scenario_result_card(),
            build_te_explore_card(),
            build_te_dose_response_card(),
        ],
        className="recommendation-column te-center-column",
    )


def build_te_predictions_column():
    """
    Build the Test Exploration page's right column: BIS, MAP, Propofol
    PK/PD, and Opioid PK - all fake, client-side-only figures (see
    test_exploration.js). All 4 always render once a scenario has run,
    regardless of opioid selection - for "None", Opioid PK shows a flat
    zero-concentration curve rather than disappearing, since "no opioid
    administered" is itself a valid, displayable result.
    """
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.I(className="fa-solid fa-magnifying-glass-chart te-card-icon"),
                            "Predictions",
                        ],
                        className="section-label te-predictions-label",
                    ),
                    html.Button(
                        ["Learn more about the model", html.Span(" ↗", className="learn-more-link-icon")],
                        id="te-learn-more-link", n_clicks=0,
                        className="learn-more-link",
                    ),
                ],
                className="predictions-header-row",
            ),
            html.Div(
                [
                    graph_card("BIS", "te-bis-graph", default_visible=True),
                    graph_card("MAP", "te-map-graph", default_visible=True),
                    graph_card("Propofol PK/PD", "te-propofol-pk-graph", default_visible=True),
                    # Lowest-priority graph on a 1080p viewport - collapsed by
                    # default (still fully available via its own "Show"
                    # checkbox, graph_card()'s existing generic mechanism) so
                    # the 3 higher-priority graphs above it fit on screen
                    # without it eating vertical space nobody asked to see yet.
                    graph_card("Opioid PK", "te-opioid-pk-graph", default_visible=False),
                ],
                id="te-predictions-grid",
                className="predictions-grid",
            ),
        ],
        className="predictions-column",
    )


def build_test_exploration_page():
    """
    Build the Test Exploration page - a sibling of main-content,
    scenario-exploration-view, and more-info-view, shown/hidden the same
    way (display:none toggling owned by render_active_page in app.py).

    The left column (Patient/Medication/Targets/Run Scenario) is always
    visible. The center+right results area, however, is a genuine
    two-state switch driven by te-scenario-active-store (a plain boolean,
    default False):
      - State A (not run / stale): te-scenario-active-store is False -
        te-results-area is display:none, so Scenario Result, Explore
        Opioid Strategy, Dose-Response, and the 4 prediction graphs are
        not rendered at all.
      - State B (scenario generated): "Run Scenario" sets the store to
        True (see runScenario in test_exploration.js) - te-results-area
        becomes visible and is filled in from te-committed-scenario-store
        (computeBaseline) and the Explore-strategy slider
        (updateExploredLive, live, no button needed).
    Editing ANY Patient/Medication/Targets field while in State B snaps
    the store back to False (markScenarioStale) - immediately hiding the
    results area again, never leaving stale results on screen. Nothing
    here is wired to recommend_regimen2023 or any other model module;
    every computed value is fake and produced client-side.
    """
    return html.Div(
        [
            # The last patient/opioid/target snapshot "Run Scenario" committed -
            # the sole input to the baseline column and to whatever the
            # Explore-strategy slider computes against. Never touched by
            # merely editing the left column (only by clicking the button).
            # Opioid defaults to "none", matching the Medication Scenario
            # dropdown's own default.
            dcc.Store(
                id="te-committed-scenario-store",
                data={
                    "age": TEST_EXPLORATION_DEFAULT_PATIENT["age"],
                    "sex": TEST_EXPLORATION_DEFAULT_PATIENT["sex"],
                    "height": TEST_EXPLORATION_DEFAULT_PATIENT["height"],
                    "weight": TEST_EXPLORATION_DEFAULT_PATIENT["weight"],
                    "sbp": TEST_EXPLORATION_DEFAULT_PATIENT["sbp"],
                    "dbp": TEST_EXPLORATION_DEFAULT_PATIENT["dbp"],
                    "opioid": "none",
                    "bisLow": TEST_EXPLORATION_DEFAULT_BIS_LOW,
                    "bisHigh": TEST_EXPLORATION_DEFAULT_BIS_HIGH,
                    "mapMode": "abs",
                    "mapValue": TEST_EXPLORATION_DEFAULT_MAP_ABS,
                },
            ),
            # True only between a "Run Scenario" click and the next
            # Patient/Medication/Targets edit - the single source of truth
            # for whether te-results-area is shown at all.
            dcc.Store(id="te-scenario-active-store", data=False),
            html.Div(
                [
                    build_te_input_column(),
                    html.Div(
                        html.Div(
                            [
                                build_te_center_column(),
                                build_te_predictions_column(),
                            ],
                            className="content-grid",
                        ),
                        id="te-results-area",
                        className="output-column",
                        style={"display": "none"},
                    ),
                ],
                className="content-grid",
            ),
        ],
        id="test-exploration-view",
        className="main-content",
        style={"display": "none"},
    )


# ============================================================
# Precomputed Remi - real-model outputs for fixed Phase 3 patient cases,
# loaded from a JSON file precomputed offline (see
# scripts/precompute_remi_cases.py) rather than computed live. Visually
# mirrors Test Exploration as closely as possible - reuses its .te-card/
# .te-result-grid/.te-strategy-* CSS classes and generic helper functions
# (_te_card_header, _te_dropdown_field, _te_result_header_row, etc. -
# despite the "_te_" name these only take ids/labels as parameters, so
# reusing them for pr-* ids is safe and does not touch Test Exploration
# itself) directly, for pixel-consistent styling without duplicating that
# CSS. Every id below is pr-*-prefixed and new - nothing here is a
# te-*/se-*/recommendation-page id, so there is no collision risk and no
# shared callback output with any other page. See the "Precomputed Remi"
# section in app.py for the callbacks that read the offline-generated data
# and drive this page; no PK/PD/optimization code is ever called from here
# or from those callbacks.
# ============================================================

PR_CASE_OPTIONS = [
    {"label": "Test Patient 1 (TEST-001)", "value": "1"},
    {"label": "Test Patient 2 (TEST-002)", "value": "2"},
    {"label": "Test Patient 3 (TEST-003)", "value": "3"},
]

# Mirrors scripts/precompute_remi_cases.py's own GRID_MIN/MAX/POINTS -
# used only to draw the slider's static bounds/marks at layout-build time.
# The callback in app.py sources the actual selectable rates from the
# loaded JSON itself, never from this Python-side copy, so if the two
# ever drift this would be a display-only quirk (e.g. a mark not lining
# up exactly), not a crash.
PR_GRID_MIN_MCGKGMIN = 0.02
PR_GRID_MAX_MCGKGMIN = 0.20
PR_GRID_POINTS = 20


def _pr_grid_rates() -> list[float]:
    """Pure-Python linspace (this file has no numpy dependency) matching the precompute script's own grid."""
    step = (PR_GRID_MAX_MCGKGMIN - PR_GRID_MIN_MCGKGMIN) / (PR_GRID_POINTS - 1)
    return [round(PR_GRID_MIN_MCGKGMIN + i * step, 4) for i in range(PR_GRID_POINTS)]


def _pr_slider_marks() -> dict:
    rates = _pr_grid_rates()
    marks = {r: "" for r in rates}
    marks[rates[0]] = f"{rates[0]:.2f}"
    marks[rates[-1]] = f"{rates[-1]:.2f}"
    return marks


def build_pr_case_card():
    """The case-selector card - the only free-choice control on this page (together with the Medication scenario dropdown below). Everything else is read-only, driven by whichever case is selected here."""
    return html.Div(
        [
            _te_card_header("fa-id-card", "Patient case"),
            _te_dropdown_field("Select patient", "pr-case-dropdown", PR_CASE_OPTIONS, None),
        ],
        className="card te-card",
    )


def _pr_locked_input_field(label: str, value_id: str, value):
    """
    Precomputed Remi's own read-only counterpart to Test Exploration's
    _te_input_field: identical markup and the same unmodified
    "te-field-row"/"te-field-label"/"te-field-input-plain" CSS classes (so
    the blue-cell appearance is pixel-identical), but readOnly=True since
    every value here is derived from the selected fixed patient case, not
    typed in. A new, PR-only function - _te_input_field itself is never
    touched, so Test Exploration's own editable inputs are unaffected.
    """
    return html.Div(
        [
            html.Div(label, className="te-field-label"),
            dcc.Input(
                id=value_id, type="text", value=value, readOnly=True,
                className="te-field-input-plain",
            ),
        ],
        className="te-field-row",
    )


def _pr_locked_pair_field(label: str, value_id: str, value, title_attr: str = ""):
    """
    Precomputed Remi's own read-only counterpart to one half of Test
    Exploration's Target BIS row: same "te-bis-field"/"field-label"/
    "scenario-compact-input te-target-input" classes (already shared
    between Scenario Exploration and Test Exploration, unmodified here),
    but readOnly=True - used for all four Targets card values, which are
    locked because the precomputed data is only valid for these exact
    target settings.
    """
    return html.Div(
        [
            html.Label(label, className="field-label"),
            dcc.Input(
                id=value_id, type="text", value=value, readOnly=True,
                className="scenario-compact-input te-target-input",
            ),
        ],
        className="te-bis-field",
        title=title_attr or None,
    )


def build_pr_patient_details_card():
    """
    Read-only "Patient scenario" card for the selected case - structurally
    identical to Test Exploration's own build_te_patient_card (same
    _te_card_header/param_subsection_label section layout, same blue-cell
    field styling via _pr_locked_input_field), but every characteristic/
    vital is locked (readOnly) since it comes from a fixed precomputed
    patient case, not free-form input. MAP/Baseline PP stay as plain-text
    derived rows (_te_derived_plain_row, unmodified, exactly as Test
    Exploration itself displays its own derived values). No Restore
    button - there is nothing to restore to on a locked page.
    """
    return html.Div(
        [
            _te_card_header("fa-user", "Patient scenario"),
            param_subsection_label("Patient characteristics", first=True),
            _pr_locked_input_field("Age (years)", "pr-detail-age", "-"),
            _pr_locked_input_field("Sex", "pr-detail-sex", "-"),
            _pr_locked_input_field("Height (cm)", "pr-detail-height", "-"),
            _pr_locked_input_field("Weight (kg)", "pr-detail-weight", "-"),
            html.Div("Baseline vitals", className="param-subsection-label"),
            _pr_locked_input_field("SBP (mmHg)", "pr-detail-sbp", "-"),
            _pr_locked_input_field("DBP (mmHg)", "pr-detail-dbp", "-"),
            _pr_locked_input_field("HR (bpm)", "pr-detail-hr", "-"),
            html.Div("Derived values", className="param-subsection-label"),
            _te_derived_plain_row("MAP (mmHg)", "pr-detail-map", "MAP = DBP + (SBP − DBP) / 3"),
            _te_derived_plain_row("Baseline PP (mmHg)", "pr-detail-pp", "Baseline PP = SBP − DBP"),
        ],
        className="card te-card",
    )


def build_pr_baseline_card():
    """
    "Medication scenario" card - visually identical to Test Exploration's
    own build_te_medication_card (same header/label/dropdown styling), but
    offering only the two baselines actually precomputed (None/
    Remifentanil). Selects which precomputed baseline is shown as the
    green reference; takes effect on the next Run Scenario click, exactly
    like Test Exploration's own Select opioid dropdown.
    """
    return html.Div(
        [
            _te_card_header("fa-pills", "Medication scenario"),
            _te_dropdown_field(
                "Select opioid", "pr-baseline-dropdown",
                [
                    {"label": "None", "value": "none"},
                    {"label": "Remifentanil", "value": "remifentanil"},
                ],
                "none",
            ),
        ],
        className="card te-card",
    )


def build_pr_targets_card():
    """
    Read-only "Targets" card - structurally identical to Test Exploration's
    own build_te_targets_card (same "Target BIS" te-bis-row pair and
    "Target MAP" pair, same blue-cell classes via _pr_locked_pair_field),
    but every value is locked. These are the fixed BIS/MAP targets the
    offline precompute script actually optimized against (see
    scripts/precompute_remi_cases.py's own metadata block); editing them
    here would silently invalidate every precomputed number, so they are
    shown but never editable.

    The real model has no separate "absolute vs relative" MAP mode toggle
    the way Test Exploration's own (fake) Target MAP does - it always
    enforces both an absolute mmHg floor and a relative %-of-baseline
    floor at once (whichever is higher wins). Rather than the earlier
    "Type"/"Value" pair (which read as unclear next to the Recommendation
    page's own plain Absolute/Relative labeling), both numbers are shown
    directly as their own locked field, matching the Recommendation
    page's input-parameters styling.
    """
    return html.Div(
        [
            _te_card_header("fa-bullseye", "Targets"),
            html.Div("Target BIS", className="param-subsection-label param-subsection-label--first"),
            html.Div(
                [
                    _pr_locked_pair_field("Lower BIS target", "pr-target-bis-low", "-"),
                    _pr_locked_pair_field("Upper BIS target", "pr-target-bis-high", "-"),
                ],
                className="te-bis-row",
            ),
            html.Div("Target MAP", className="param-subsection-label"),
            html.Div(
                [
                    _pr_locked_pair_field(
                        "Absolute (mmHg)", "pr-target-map-abs", "-",
                        title_attr="Absolute MAP floor the model enforces, in mmHg.",
                    ),
                    _pr_locked_pair_field(
                        "Relative (% baseline)", "pr-target-map-rel", "-",
                        title_attr="Relative MAP floor the model enforces, as a percentage of this patient's own baseline MAP.",
                    ),
                ],
                className="te-bis-row",
            ),
        ],
        className="card te-card",
    )


def build_pr_input_column():
    return html.Div(
        [
            build_pr_case_card(),
            build_pr_patient_details_card(),
            build_pr_baseline_card(),
            build_pr_targets_card(),
            html.Button(
                "Run Scenario", id="pr-run-scenario-btn", n_clicks=0,
                className="run-recommendation-btn te-run-scenario-btn",
            ),
        ],
        className="input-column",
    )


def build_pr_scenario_result_card():
    """
    Mirrors Test Exploration's Scenario Result card. Unlike Test
    Exploration, the comparison grid's rows (pr-result-grid) are entirely
    built by the callback in app.py, not pre-declared here with fixed ids -
    real maintenance schedules from compress_minute_schedule() can have a
    different number of rows per case/rate (unlike Test Exploration's
    always-exactly-2-rows fake schedule), so a fixed-row-count layout
    would not fit real data safely.
    """
    return html.Div(
        [
            _te_card_header("fa-code-compare", "Scenario result"),
            html.P(
                "Compare the selected remifentanil grid scenario against the chosen baseline reference - both are real, precomputed model outputs.",
                className="te-card-subtitle",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Span("Baseline Strategy", className="te-strategy-label"),
                            html.Span("-", id="pr-strategy-baseline-value", className="te-strategy-value"),
                        ],
                        className="te-strategy-card te-strategy-card--baseline",
                    ),
                    html.Div(html.I(className="fa-solid fa-arrow-right"), className="te-strategy-arrow"),
                    html.Div(
                        [
                            html.Span("Explored Strategy", className="te-strategy-label"),
                            html.Span("-", id="pr-strategy-explored-value", className="te-strategy-value"),
                        ],
                        className="te-strategy-card te-strategy-card--explored",
                    ),
                ],
                className="te-strategy-banner",
            ),
            html.Div(id="pr-result-grid", className="te-result-grid"),
        ],
        className="card te-card te-result-card",
    )


def build_pr_explore_card():
    """
    "Explore opioid strategy" card - visually and structurally identical
    to Test Exploration's own build_te_explore_card (same
    _te_dropdown_field "Opioid to explore" options, same slider-wrapper-
    hidden-until-Remifentanil-is-picked pattern), but server-side: the
    slider's visibility is toggled by toggle_pr_explore_slider (app.py)
    reading pr-explore-opioid-dropdown, instead of Test Exploration's own
    clientside onExploreOpioidChange. Only "None"/"Remifentanil" are
    selectable - Sufentanil/Fentanyl are listed disabled, exactly like
    Test Exploration's own dropdown, and can never be selected or trigger
    any callback. The slider itself keeps step=None + marks so it can
    only land exactly on one of the 20 precomputed grid rates, matching
    the fixed rates the offline script actually computed.
    """
    marks = _pr_slider_marks()
    rates = _pr_grid_rates()
    return html.Div(
        [
            _te_card_header("fa-sliders", "Explore opioid strategy"),
            _te_dropdown_field(
                "Opioid to explore", "pr-explore-opioid-dropdown",
                [
                    {"label": "None", "value": "none"},
                    {"label": "Remifentanil", "value": "remifentanil"},
                    {"label": "Sufentanil (future extension)", "value": "sufentanil", "disabled": True},
                    {"label": "Fentanyl (future extension)", "value": "fentanyl", "disabled": True},
                ],
                "none",
            ),
            html.Div(
                [
                    html.Div("Remifentanil infusion rate", className="te-summary-label"),
                    html.Div("", id="pr-rate-label", className="te-explore-rate-value"),
                    dcc.Slider(
                        id="pr-rate-slider",
                        min=rates[0], max=rates[-1], step=None, marks=marks, value=rates[0],
                        tooltip={"placement": "bottom", "always_visible": False},
                        className="scenario-slider te-explore-slider",
                    ),
                ],
                id="pr-rate-slider-wrapper",
                style={"display": "none"},
            ),
        ],
        className="card te-card",
    )


def build_pr_dose_response_card():
    """
    Dose-Response Interaction - visually matches the Recommendation
    page's own induction-dose rationale graph (MAP/BIS target bands,
    green "both targets met" region, diamond markers, reversed BIS axis),
    built from each regimen's own precomputed dose-sweep data (see
    scripts/precompute_remi_cases.py's own _dose_sweep()). See
    make_pr_induction_dose_rationale_figure in app.py.
    """
    card = graph_card("Dose-Response Interaction", "pr-dose-response-graph", tall=True)
    return html.Div([card], id="pr-dose-response-card-wrapper")


def build_pr_center_column():
    return html.Div(
        [
            build_pr_scenario_result_card(),
            build_pr_explore_card(),
            build_pr_dose_response_card(),
        ],
        className="recommendation-column te-center-column",
    )


def build_pr_predictions_column():
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.I(className="fa-solid fa-magnifying-glass-chart te-card-icon"),
                            "Predictions",
                        ],
                        className="section-label te-predictions-label",
                    ),
                ],
                className="predictions-header-row",
            ),
            html.Div(
                [
                    graph_card("BIS", "pr-bis-graph", default_visible=True),
                    graph_card("MAP", "pr-map-graph", default_visible=True),
                    graph_card("Propofol PK/PD", "pr-propofol-pk-graph", default_visible=True),
                    graph_card("Remifentanil PK", "pr-remifentanil-pk-graph", default_visible=True),
                ],
                id="pr-predictions-grid",
                className="predictions-grid",
            ),
        ],
        className="predictions-column",
    )


def build_precomputed_remi_page(precomputed_remi_available: bool = True):
    """
    Build the Precomputed Remi page - a sibling of main-content,
    scenario-exploration-view, test-exploration-view, and more-info-view,
    shown/hidden the same way (display:none toggling owned by
    render_active_page in app.py).

    Two guard states, both handled without any callback:
      - precomputed_remi_available=False (the JSON file was missing at
        server startup): pr-missing-data-banner shows instead of the left
        column, and pr-results-area/pr-case-card are never even reachable.
      - A case is selectable but not yet chosen, or chosen but missing
        grid data for some reason: pr-results-area stays hidden
        (display:none, driven by pr-selected-case-store, mirroring Test
        Exploration's own te-results-area/te-scenario-active-store
        pattern) and pr-case-error-banner (inside the left column) shows
        the specific problem instead.
    """
    if not precomputed_remi_available:
        return html.Div(
            html.Div(
                [
                    html.I(className="fa-solid fa-triangle-exclamation"),
                    " Precomputed data has not been generated yet. Run scripts/precompute_remi_cases.py "
                    "and restart the app to use this page.",
                ],
                className="card te-card pr-missing-data-banner",
            ),
            id="precomputed-remi-view",
            className="main-content",
            style={"display": "none"},
        )

    return html.Div(
        [
            html.Div(
                [
                    build_pr_input_column(),
                    html.Div(
                        [
                            html.Div(
                                "This case has no precomputed grid data. Re-run scripts/precompute_remi_cases.py for this patient.",
                                id="pr-case-error-banner",
                                className="card te-card pr-case-error-banner",
                                style={"display": "none"},
                            ),
                            html.Div(
                                html.Div(
                                    [
                                        build_pr_center_column(),
                                        build_pr_predictions_column(),
                                    ],
                                    className="content-grid",
                                ),
                                id="pr-results-area",
                                className="output-column",
                                style={"display": "none"},
                            ),
                        ],
                    ),
                ],
                className="content-grid",
            ),
        ],
        id="precomputed-remi-view",
        className="main-content",
        style={"display": "none"},
    )


def build_layout(precomputed_remi_available: bool = True):
    """
    Build the main layout of the dashboard: a navigation sidebar plus a
    three-column input/recommendation/predictions dashboard, the Scenario
    Exploration page, the Test Exploration page, the Precomputed Remi page,
    or the "More Info" page - all siblings, and only one is ever visible at
    a time (Output("main-content", "style") / Output(
    "scenario-exploration-view", "style") / etc., driven by
    active-page-store and owned by render_active_page in app.py).

    precomputed_remi_available is passed in from app.py (which is the one
    that actually loads src/propofol/data/precomputed_remi_cases.json, at
    import time) purely so build_precomputed_remi_page() can render its
    "no precomputed data yet" message immediately, with no callback and no
    risk of a flash-of-wrong-content - dashboard_layout.py never reads that
    file itself.

    The input column sits outside `dcc.Loading` (as before, inputs never
    show a loading overlay); the recommendation column (including the
    induction-dose rationale graph) and the predictions column are both
    inside one `dcc.Loading`, so every `run_model` output shows the same
    loading feedback it did previously.
    """
    return html.Div(
        [
            dcc.Store(id="sex-store", data=PRESET_TEST_PATIENTS["1"]["sex"]),
            # Which fixed Phase 3 patient case ("1"/"2"/"3") is currently
            # selected on the Precomputed Remi page, if any. The actual
            # precomputed numbers never travel through this store (they
            # stay server-side in app.py's PRECOMPUTED_REMI_DATA, loaded
            # once at startup) - every pr-* callback re-reads that global
            # keyed on this store's case id + the rate slider's own value,
            # so no multi-megabyte payload round-trips to the browser on
            # every case switch.
            dcc.Store(id="pr-selected-case-store", data=None),
            # Which baseline reference ("none"/"remifentanil") was selected
            # at the time Run Scenario was last clicked - committed
            # separately from pr-baseline-dropdown's own live value so that
            # changing the dropdown alone (before Run Scenario) does not
            # affect what pr-results-area currently shows, mirroring Test
            # Exploration's own commit-on-Run-Scenario flow.
            dcc.Store(id="pr-committed-baseline-store", data="none"),
            # Whether pr-results-area should currently be shown - True only
            # right after a successful Run Scenario click, reset to False by
            # any subsequent change to the case or medication (baseline)
            # dropdown. Mirrors Test Exploration's own
            # te-scenario-active-store/markScenarioStale pattern, in Python
            # instead of clientside JS.
            dcc.Store(id="pr-scenario-active-store", data=False),
            # Which preset test patient ("1"/"2"/"3") is currently active -
            # "1" (Test Patient 1) by default, matching the "Select
            # patient" dropdown's own default value and
            # build_patient_parameters_card's own field defaults. Purely a
            # display/tracking aid; the actual patient values it applies
            # live in PRESET_TEST_PATIENTS (dashboard_layout.py) and are
            # written directly into the same age/height/weight/... fields
            # every other input already uses.
            dcc.Store(id="selected-test-patient-store", data="1"),
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
            build_test_exploration_page(),
            build_precomputed_remi_page(precomputed_remi_available),
            build_more_info_view(),
        ],
        className="app-shell",
    )
