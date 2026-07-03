"""
Text content for the "More Info" page.

Kept separate from the layout/rendering code (dashboard_layout.py) so the
wording can be edited here without touching how the accordion is built or
styled. Each section's body is Markdown (rendered via dcc.Markdown), which
is the only formatting layer applied to the provided text - paragraph
breaks and bullet/numbered lists render as-is, no wording was added.
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
This tool uses a pharmacokinetic–pharmacodynamic model to estimate an optimal patient-specific propofol induction and early maintenance regimen for the first 15 minutes.

The model combines patient characteristics with predicted propofol concentrations, depth of anaesthesia, and haemodynamic response.

The recommended dose is selected by balancing two clinical goals:

- achieving an adequate hypnotic effect, expressed as a predicted BIS between 40 and 60
- avoiding clinically relevant hypotension, expressed as a predicted MAP below the predefined safety threshold. With this threshold, a MAP below 65 mmHg and a MAP decrease of more than 30% from baseline are both avoided.

For each candidate dose, the model predicts:

- the propofol plasma and effect-site concentration over time
- the expected BIS response over time
- the expected MAP response over time

The tool then assigns penalties when the predicted response deviates from the target, with the following priorities:

1. Predicted BIS is above 60
2. Predicted BIS is below 40
3. Predicted MAP falls below the MAP safety threshold
""",
    },
    {
        "id": "what-does-confidence-mean",
        "title": "What does the confidence mean?",
        "body": """
To account for uncertainty in individual patient response, the model repeats the simulation 100 times for the recommended dose.

The confidence estimate indicates how often the predefined treatment targets are met with the given dose when the simulation is repeated.
""",
    },
]
