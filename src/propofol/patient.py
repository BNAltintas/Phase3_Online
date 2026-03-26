"""Define patient classes"""

class EleveldPatient():
    """Patient class corresponding Eleveld patient in Eleveld pk/pd model. """
    def __init__(self, age: float, weight: float, height: float, sex: str,
                 opiates: bool, pma: float = 40,
                 blood_sampling_site: str = 'arterial',
                 base_pp: float = None,
                 base_hr: float = None,
                 base_map: float = None)  -> None:
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
            Called 'PMW' in the NONMEM control files.
        blood_sampling_site : str, optional
            Blood sampling site ('arterial' (default) or 'venous')).

        Notes
        -----
        - Units: age (years), weight (kg), height (cm), pma (weeks).
        """
        # Note: NONMEM control file uses 52, not (365 * 3 + 366) / 4 / 7 ~ 52.18
        self.years_to_weeks = 52.  # weeks / year
        self.weeks_to_years = 1 / self.years_to_weeks  # year / week

        # individual setting
        self.age = age  # years
        self.weight = weight
        self.height = height
        self.bmi = self.weight / (self.height / 100.)**2
        # Note: if PMA was not recorded it was assumed to be 40 weeks longer than age.
        self.pma = self.age * self.years_to_weeks + pma # weeks
        self.sex = sex.lower()
        self.opiates = opiates
        self.blood_sampling_site = blood_sampling_site

        # Fat free mass (kg)
        self.ffm = self.__f_al_sallami(self.age, self.weight, self.bmi)

        # Baseline map, sv and tpr
        if base_pp is not None:
            self.base_sv = base_pp * 1.5
        else:
            self.base_sv = None
        self.base_hr = base_hr
        self.base_map = base_map

        if all(x is not None for x in [self.base_map, self.base_hr, self.base_sv]):
            self.base_tpr = self.base_map / (self.base_sv * self.base_hr)
        else:
            self.base_tpr = None

    def __f_al_sallami(self, age, weight, bmi):
        """ Fat free mass in kg from Al Sallami

        Notes
        -----
        In the nonmem control files the last part appears slightly different
        # For male: 42.92*wgt/(30.93+BMI)
        # For female 37.99*Wgt/(35.98+BMI)
        """

        if self.sex == 'male':
            return ((0.88 + (1 - 0.88) / (1 + (age / 13.4)**(-12.7)))
                    * (9270 * weight) / (6680 + 216 * bmi))
        elif self.sex == 'female':
            return ((1.11 + (1 - 1.11) / (1 + (age / 7.1)**(-1.1)))
                    * (9270 * weight) / (8780 + 244 * bmi))
