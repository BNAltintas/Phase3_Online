"""
Text content for the "More Info" page.

Kept separate from the layout/rendering code (dashboard_layout.py) so the
wording can be edited here without touching how the accordion is built or
styled. Each section's body is Markdown (rendered via dcc.Markdown) -
paragraphs and bullet/numbered lists render as-is; a leading "> " line
renders as a highlighted callout box (see .accordion-body-text blockquote
in style.css), used here for the BIS/MAP priority note.
"""

MORE_INFO_TITLE = "Learn more about the model"

MORE_INFO_SUBTITLE = (
    "Understand how DosePilot generates recommendations and how to interpret model outputs."
)

MORE_INFO_SECTIONS = [
    {
        "id": "what-does-this-tool-do",
        "title": "What does this tool do?",
        "body": """
DosePilot estimates a patient-specific propofol induction dose and early maintenance regimen for the first 15 minutes of anaesthesia.

The recommendation is generated to achieve two clinical targets at the same time:

- an adequate hypnotic effect, reflected by a predicted BIS between 40 and 60
- stable blood pressure, reflected by a predicted MAP that stays above a safety threshold
""",
    },
    {
        "id": "how-are-bis-map-predicted",
        "title": "How are BIS and MAP predicted?",
        "body": """
BIS (depth of anaesthesia) is predicted from the simulated propofol concentration at the site of drug effect, using a validated pharmacodynamic model.

MAP (blood pressure) is predicted using a haemodynamic model that accounts for the cardiovascular effects of propofol and, when used, remifentanil.

Both predictions come from the same patient-specific simulation, so the BIS and MAP graphs you see always describe the same simulated patient response.
""",
    },
    {
        "id": "how-does-model-choose-dose",
        "title": "How does the model choose the recommended dose?",
        "body": """
The model simulates many candidate doses and dosing schedules for the individual patient, and predicts the resulting BIS and MAP response for each one.

The recommended dose is the candidate that best meets the following clinical priorities, in order:

1. Avoid BIS rising above 60
2. Avoid BIS falling below 40
3. Avoid MAP falling below the safety threshold

Only once all three are satisfied does the model use secondary preferences to choose between remaining options - such as keeping BIS closer to the centre of the target range, using less drug overall, or a smoother dosing schedule.

> **Important:** When BIS and MAP targets conflict, the model prioritizes achieving the BIS target before optimizing MAP.
""",
    },
    {
        "id": "confidence",
        "title": "Confidence",
        "body": """
To reflect natural variability between patients, the model repeats its simulation 100 times for the recommended dose.

The confidence score shows how often the BIS and MAP targets were both met across these simulations. A higher confidence means the recommended dose is more consistently effective across likely variation in patient response.
""",
    },
]
