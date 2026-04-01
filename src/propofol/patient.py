"""Define patient classes"""


class EleveldPatient:
    """Patient class corresponding to the Eleveld patient in the Eleveld PK/PD model."""

    def __init__(
        self,
        age: float,
        weight: float,
        height: float,
        sex: str,
        opiates: bool,
        pma: float = 40,
        blood_sampling_site: str = "arterial",
        base_sap: float = None,
        base_dap: float = None,
        base_pp: float = None,
        base_hr: float = None,
        base_map: float = None,
    ) -> None:
        """
        Initialize patient parameters for PK/PD modelling.

        Parameters
        ----------
        age : float
            Age in years.
        weight : float
            Body weight in kilograms.
        height : float
            Height in centimeters.
        sex : str
            Biological sex ('male' or 'female', case-insensitive).
        opiates : bool
            Indicator of perioperative/opioid exposure.
        pma : float, optional
            Postmenstrual age baseline in weeks (default: 40).
            Total PMA used by the model is computed as pma + age * 52.
        blood_sampling_site : str, optional
            Blood sampling site ('arterial' or 'venous').
        base_sap : float, optional
            Baseline systolic arterial pressure in mmHg.
        base_dap : float, optional
            Baseline diastolic arterial pressure in mmHg.
        base_pp : float, optional
            Baseline pulse pressure in mmHg.
        base_hr : float, optional
            Baseline heart rate in bpm.
        base_map : float, optional
            Baseline mean arterial pressure in mmHg.

        Notes
        -----
        Priority for derived haemodynamic quantities:
        1. If base_sap and base_dap are provided:
           - base_pp = base_sap - base_dap
           - base_map = (base_sap + 2 * base_dap) / 3
        2. Otherwise, use directly supplied base_pp and/or base_map.
        3. base_sv = 1.5 * base_pp, if base_pp is available.
        4. base_tpr = base_map / (base_sv * base_hr), if all are available.
        """

        # Note: NONMEM control file uses 52, not ~52.18
        self.years_to_weeks = 52.0
        self.weeks_to_years = 1.0 / self.years_to_weeks

        # Individual characteristics
        self.age = age  # years
        self.weight = weight  # kg
        self.height = height  # cm
        self.bmi = self.weight / (self.height / 100.0) ** 2
        self.pma = self.age * self.years_to_weeks + pma  # weeks
        self.sex = sex.lower()
        self.opiates = opiates
        self.blood_sampling_site = blood_sampling_site

        if self.sex not in {"male", "female"}:
            raise ValueError("sex must be 'male' or 'female'")

        if self.blood_sampling_site not in {"arterial", "venous"}:
            raise ValueError("blood_sampling_site must be 'arterial' or 'venous'")

        # Fat free mass (kg)
        self.ffm = self.__f_al_sallami(self.age, self.weight, self.bmi)

        # Baseline pressure inputs
        self.base_sap = base_sap
        self.base_dap = base_dap
        self.base_hr = base_hr

        # Derive PP and MAP from SAP/DAP if available
        if base_sap is not None and base_dap is not None:
            self.base_pp = base_sap - base_dap
            self.base_map = (base_sap + 2.0 * base_dap) / 3.0
        else:
            self.base_pp = base_pp
            self.base_map = base_map

        # Derived baseline stroke volume
        if self.base_pp is not None:
            self.base_sv = self.base_pp * 1.5
        else:
            self.base_sv = None

        # Derived baseline TPR
        if all(x is not None for x in [self.base_map, self.base_hr, self.base_sv]):
            self.base_tpr = self.base_map / (self.base_sv * self.base_hr)
        else:
            self.base_tpr = None

    def __f_al_sallami(self, age, weight, bmi):
        """Fat free mass in kg from Al-Sallami.

        Notes
        -----
        In the NONMEM control files the last part appears slightly different:
        - male:   42.92 * WGT / (30.93 + BMI)
        - female: 37.99 * WGT / (35.98 + BMI)
        """
        if self.sex == "male":
            return (
                (0.88 + (1 - 0.88) / (1 + (age / 13.4) ** (-12.7)))
                * (9270 * weight)
                / (6680 + 216 * bmi)
            )
        elif self.sex == "female":
            return (
                (1.11 + (1 - 1.11) / (1 + (age / 7.1) ** (-1.1)))
                * (9270 * weight)
                / (8780 + 244 * bmi)
            )