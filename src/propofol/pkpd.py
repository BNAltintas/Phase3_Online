import logging

import numpy as np
from scipy import integrate
from scipy.optimize import brentq

from propofol.protocols import Patient

logger = logging.getLogger(__name__)


class EleveldPatient():
    def __init__(self, age: float, weight: float, height: float, sex: str,
                 opiates: bool, pma: float = 40,
                 blood_sampling_site: str = 'arterial')  -> None:
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


class EleveldPK():
    """Eleveld PK model from https://doi.org/10.1016/j.bja.2018.01.018

    Notes
    -----
    - Theta's from supplementary material 1.
    Note: some of the values in the table of the paper correspond to
    exp(Theta) in the NONMEM control file.
    """
    def __init__(self, patient: Patient, use_bsv: bool = False):
        self.patient = patient
        self.patient_ref = EleveldPatient(age=35, height=170, weight=70, sex='male',
                                          opiates=False)
        self.use_bsv = use_bsv

        self._set_params()


    def _f_ageing(self, x, age, age_ref):
        return np.exp(x * (age - age_ref))

    def _set_params(self):
        self.Theta1 = 6.283 # v1ref(adult) (L)
        self.v1ref = self.Theta1
        self.Theta2 = 25.501 # v2ref (L)
        self.V2ref = self.Theta2
        self.Theta3 = 272.817  # v3ref (L)
        self.V3ref = self.Theta3
        self.Theta4 = 1.790  # clref(male) (L / min)
        self.Theta5 = 1.750  # q2ref(adult) (L / min)
        self.Theta6 = 1.109  # q3ref (L / min)
        self.Theta7 = 0.191  # log-error, note NONMEN says: 0.1913070
        self.Theta8 = 42.3  # maturation CL E50 (weeks)
        self.Theta9 = 9.05  # maturation CL slope
        self.Theta10 = -0.0156  # V2 declines with age. Note control file says -0.15633
        # Note control file says: -.00285709
        self.Theta11 = -0.00286  # CL declines with age with opiate.
        self.Theta12 = 33.6  # V1 sigmoid E50 (kg)
        self.Theta13 = -0.0138  # V3 declines with age with opiates
        self.Theta14 = 68.3  # maturation Q3 E50 (weeks)
        self.Theta15 = 2.100  # clref(female) L / min
        self.Theta16 = 1.304  # q2ref(child)
        self.Theta17 = 1.419  # v1 extra(max,venous) (L)
        self.Theta18 = 0.681  # q2 less for venous (L/min)

        # Set eta
        if self.use_bsv:
            self.draw_eta()
        else:
            self.reset_eta()
        self.reset_epsilon()

    def _update_pk(self):
        """(Re)calculate V1, V2, V3, CL, Q2, Q3 and k10..k31
        given the etas and patient characteristics.
        """
        self.V1_arterial = (self.Theta1
                            * self.f_central(self.patient.weight)
                            / self.f_central(self.patient_ref.weight)
                            * np.exp(self.eta1))  # L
        self.V1_venous =  (self.V1_arterial
                           * (1 + self.Theta17
                              * (1 - self.f_central(self.patient.weight))))
        self.V2 = (self.Theta2
                   * self.patient.weight / self.patient_ref.weight
                   * self._f_ageing(self.Theta10,
                                    self.patient.age,
                                    self.patient_ref.age)
                   * np.exp(self.eta2))  # L
        self.V3 = (self.Theta3
                   * self.patient.ffm / self.patient_ref.ffm
                   * self.f_opiates(self.Theta13)
                   * np.exp(self.eta3))  # L

        self.CL = self._elimination_clearance()  # L / min

        self.Q2_arterial = (self.Theta5
                            * np.float_power(self.V2 / self.V2ref, 0.75)
                            * (1 + self.Theta16
                               * (1 - self.f_q3_maturation(self.patient.age)))
                            * np.exp(self.eta5)
                            )  # L / min
        self.Q2_venous = self.Q2_arterial * self.Theta18
        self.Q3 = (self.Theta6
                   * np.float_power(self.V3 / self.V3ref, 0.75)
                   * self.f_q3_maturation(self.patient.age)
                   / self.f_q3_maturation(self.patient_ref.age)
                   * np.exp(self.eta6)
        )

        # Elimination rate constants
        if self.patient.blood_sampling_site == 'arterial':
            self.k10 = self.CL / self.V1_arterial  # min^-1
            self.k12 = self.Q2_arterial / self.V1_arterial
            self.k13 = self.Q3 / self.V1_arterial
            self.k21 = self.Q2_arterial / self.V2
            self.V1 = self.V1_arterial
        elif self.patient.blood_sampling_site == 'venous':
            self.k10 = self.CL / self.V1_venous
            self.k12 = self.Q2_venous / self.V1_venous
            self.k13 = self.Q3 / self.V1_venous
            self.k21 = self.Q2_venous / self.V2
            self.V1 = self.V1_venous
        self.k31 = self.Q3 / self.V3

    def draw_eta(self):
        """
        Draw independent eta ~ N(0, sqrt(omega^2)) for V1, V2, V3, CL, Q2, Q3
        and recompute PK parameters accordingly.
        """

        # variances (from the NONMEM code)
        omega2 = np.array([
            3.725310e-01,
            3.192030e-01,
            3.563620e-01,
            7.022160e-02,
            1.199800e-01,
            4.360060e-02,
            2.142470e-01
        ])

        n_samples = 1
        eta = np.random.multivariate_normal(np.zeros_like(omega2),
                                            np.diag(omega2),
                                            size=n_samples)
        (self.eta1,
          self.eta2,
          self.eta3,
          self.eta4,
          self.eta5,
          self.eta6,
          self.eta7) = np.squeeze(eta)

        # Update pk parameters after drawing new etas
        self._update_pk()

    def reset_eta(self):
        """
        (Re)set all etas to 0 and (re)compute PK parameters accordingly.
        """
        self.eta1 = 0
        self.eta2 = 0
        self.eta3 = 0
        self.eta4 = 0
        self.eta5 = 0
        self.eta6 = 0
        self.eta7 = 0

        # Update pk parameters after drawing new etas
        self._update_pk()

    def draw_epsilon(self):
        """Residual error in the log-domain with variance fixed to 1"""
        self.epsilon = np.random.normal(loc=0, scale=1)

    def reset_epsilon(self):
        """(Re)set epsilon to 0"""
        self.epsilon = 0

    def f_sigmoid(self, x, E50, lmbda):
        return x**lmbda / (x**lmbda + E50**lmbda)

    def f_central(self, x):
        return self.f_sigmoid(x, self.Theta12, 1)

    def f_opiates(self, x):
        if self.patient.opiates:
            return np.exp(x * self.patient.age)  # TODO: yr or weeks?
        else:
            return 1  # absence of opiates

    def f_cl_maturation(self, pma):
        return self.f_sigmoid(pma, self.Theta8, self.Theta9)

    def _elimination_clearance(self):

        if self.patient.sex == 'male':
            theta = self.Theta4
        elif self.patient.sex == 'female':
            theta = self.Theta15
        return (theta
                * (self.patient.weight / self.patient_ref.weight)**0.75
                * self.f_cl_maturation(self.patient.pma)
                / self.f_cl_maturation(self.patient_ref.pma)
                * self.f_opiates(self.Theta11) * np.exp(self.eta4)
                )
    def f_q3_maturation(self, age):
        # 90% at 11 years
        return self.f_sigmoid(age * 52 + 40, self.Theta14, 1)

    def C_obs(self, x):
        return x * np.exp(self.Theta7 * self.epsilon * np.exp(self.eta7))

class EleveldPD():
    """Eleveld PK model from https://doi.org/10.1016/j.bja.2018.01.018

    Notes
    -----
    - Theta's from supplementary material 1, file 4
    - Note: some of the values in the table of the paper correspond to exp(Theta) in the
    NONMEM control.
    """
    def __init__(self, patient: Patient, use_bsv: bool = False):
        self.patient = patient
        self.patient_ref = EleveldPatient(age=35, height=170, weight=70, sex='male',
                                          opiates=False)
        self.use_bsv = use_bsv

        self._set_params()


    def _f_ageing(self, x, age, age_ref):
        return np.exp(x * (age - age_ref))

    def _set_params(self):

        # Parameters
        self.Theta1 = 3.08  # e50 (microgram / ml)
        self.Theta2 = 0.146  # ke0 (min^-1)
        self.Theta3 = 93.0  # emax (note: NONMEM control file says: 92.98240)
        self.Theta4 = 1.47  # gamma
        self.Theta5 = 8.03  # residual error BIS
        self.Theta6 = 0.0517  # age delay (note: NONMEM control file says: 0.05173990)
        self.Theta7 = -0.00635  # age e50 (note: NONMEM control file says: 0.006348370)
        self.Theta8 = 1.24  # min^-1
        self.Theta9 = 1.89  # gamma

        # Set eta
        if self.use_bsv:
            self.draw_eta()
        else:
             self.reset_eta()

        self.epsilon = 0

    def _update_pd(self):
        """(Re)calculate Ce50, ke0, bis_baseline, bis_delay
        given the current etas and patient characteristics."""

        self.Ce50 = (self.Theta1
                     * self._f_ageing(self.Theta7,
                                      self.patient.age,
                                      self.patient_ref.age)
                     * np.exp(self.eta1))  # microgram / ml (or: mg / L)
        self.ke0 = self._get_ke0()
        self.bis_baseline = self.Theta3
        self.bis_delay = (15. + np.exp(self.Theta6 * self.patient.age)) / 60.  # minutes


    def draw_eta(self):
        """
        Draw independent eta ~ N(0, sqrt(omega^2)) for  Ce50, ke0 and bis_baseline
        and recompute PD parameters accordingly.
        """
        # Variances from the NONMEM code
        omega2 = np.array([
            5.841270e-02,
            4.924930e-01,
            5.287310e-02
        ])

        n_samples = 1  # note: scipy.integrate cannot do multiple realizations at once.
        eta = np.random.multivariate_normal(np.zeros_like(omega2),
                                            np.diag(omega2),
                                            size=n_samples)
        self.eta1, self.eta2, self.eta3 = np.squeeze(eta)

        # Update pd parameters after drawing new etas
        self._update_pd()

    def reset_eta(self):
        """
        (Re)set all etas to 0 and (re)compute PD parameters accordingly.
        """
        self.eta1 = 0
        self.eta2 = 0
        self.eta3 = 0

        # Update pd parameters after drawing new etas
        self._update_pd()

    # TODO: setting epsilon could be moved into a Parent base class
    def draw_epsilon(self):
        """Residual error in the log-domain with variance fixed to 1"""
        self.epsilon = np.random.normal(loc=0, scale=1)

    def reset_epsilon(self):
        """(Re)set epsilon to 0"""
        self.epsilon = 0

    def _get_ke0(self):
        if self.patient.blood_sampling_site == 'arterial':
            theta = self.Theta2
        elif self.patient.blood_sampling_site == 'venous':
            theta = self.Theta8
        return theta * (self.patient.weight / 70)**-0.25 * np.exp(self.eta2)

    def gamma(self, x):
        if x <= self.Ce50:
            return self.Theta4
        elif x > self.Ce50:
            return self.Theta9

    def bis(self, x):
        y = self.gamma(x)

        return (self.bis_baseline
                * (1 - x**y / (self.Ce50**y + x**y))  # IPRED in NONMEM file
                + self.Theta5 * self.epsilon * np.exp(self.eta3)  # RESD in NONMEM file
        )

class PKPD_solver():
    def __init__(self, patient, pk, pd):
        # TODO: patient argument can be removed if we move blood_sampling_site
        self.patient = patient
        self.pk = pk
        self.pd = pd


    def derivative(self, X, t, V1, k10, k12, k13, k21, k31, ke0):
        A1, A2, A3, Ce = X

        dotA1 = (- (k10 + k12 + k13) * A1
                 + k21 * A2 + k31 * A3
        )
        dotA2 = k12 * A1 - k21 * A2
        dotA3 = k13 * A1 - k31 * A3

        # Since Ae does not affect Ax the next line be moved out.
        dotCe = ke0 * (A1 / V1 - Ce)
        return np.array([dotA1, dotA2, dotA3, dotCe])

    def solve_ode(self, t, X0):
        # TODO: replace by integrate.solve_ivp
        res = integrate.odeint(self.derivative, X0, t,
                        args = (self.pk.V1,
                                self.pk.k10,
                                self.pk.k12,
                                self.pk.k13,
                                self.pk.k21,
                                self.pk.k31,
                                self.pd.ke0)
        )
        return res.T  # A1[mg], A2[mg], A3[mg], Ce[mg/L] with units [mg, mg, mg, mg / L]

    def find_dose_drug_effect_50(self) -> float:
        if self.patient.blood_sampling_site == 'venous':
            raise ValueError(
                f"Blood sampling site {self.patient.blood_sampling_site} "
                "is not accepted for predicting effect size concentrations "
                "(see 'Drug transport to the effect site' in Eleveld et al. 2018)."
            )
        def f(x):
            _, _, _, Ce = self.solve_ode(t = np.linspace(0, 15, 15*60+1), X0 = [x, 0, 0, 0])
            return Ce.max() - self.pd.Ce50
        dose50 = brentq(f, 0.1, 10000)
        return dose50 # mg

    def print_model_parameters(self):
        logger.info(f"BW:{self.patient.weight}kg")
        logger.info(f"BH:{self.patient.height}cm")
        logger.info(f"BMI:{self.patient.bmi:.2f}")
        logger.info(f"{self.patient.age}y, {self.patient.sex}")
        logger.info(f"Assume presence of concomitant opioid? {self.patient.opiates}")
        logger.info(f"vc = {self.pk.V1:.3f} L")
        logger.info(f"v2 = {self.pk.V2:.3f} L")
        logger.info(f"v3 = {self.pk.V3:.3f} L")
        logger.info(f"k10 = {self.pk.k10:.8f}")
        logger.info(f"k12 = {self.pk.k12:.8f}")
        logger.info(f"k13 = {self.pk.k13:.8f}")
        logger.info(f"k21 = {self.pk.k21:.8f}")
        logger.info(f"k31 = {self.pk.k31:.8f}")
        logger.info(f"ke0 = {self.pd.ke0:.8f}")

        # Compare to simtiva:
        # AGe 1
        # Correct v1,v2,v3,k10,ke0
        # Incorrect k12, k21,
        # Close: k13, k31

        # Age 35
        # Correct v1,v2,v3,k10,ke0
        # Incorrect k31, k13,
        # Close: k12, k21

        # Age 70
        # Incorrect: k12, k21, k13, k31


    def __call__(self, t = None, X0  = None):
        if t is None:
            t = np.linspace(0, 15, 15*60+1)
        if X0 is None:
            X0 = [1, 0, 0, 0]
        return self.solve_ode(t, X0)


