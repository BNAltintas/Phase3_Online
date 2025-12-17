"""Declaration of protocols for type hints"""

from typing import Protocol


class Patient(Protocol):
    """
    Protocol describing the patient covariates required by the PK/PD model.

    Attributes
    ----------
    age : float
        Patient age in years.
    height : float
        Patient height in centimeters.
    weight : float
        Patient weight in kilograms.
    bmi : float
        Body mass index in kg/m² (weight / height² with height in meters).
    pma : float
        Post‑menstrual age in weeks (used for neonates) or None if not applicable.
    sex : str
        Biological sex, e.g. 'male' or 'female'.
    opiates : bool
        Whether the patient is currently receiving opiate therapy (True/False).
    ffm : float
        Fat‑free mass in kilograms.
    blood_sampling_site : str
        Site of blood sampling, e.g. 'arterial', 'venous'.

    Notes
    -----
    - Both ints and float are acceptable for numeric values where appropriate.
    """
    age = ...
    height = ...
    weight = ...
    bmi = ...
    pma = ...
    sex = ...
    opiates = ...
    ffm = ...
    blood_sampling_site = ...
