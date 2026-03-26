import logging
from typing import Any, Callable, Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import integrate

from propofol.patient import EleveldPatient
from propofol.protocols import Patient

logger = logging.getLogger(__name__)


class SuHaemoPD():
    """Su haemodynamic PD model from https://doi.org/10.1016/j.bja.2022.01.022.

    Notes
    -----
    - Theta's from NONMEM file
    """
    def __init__(
        self,
        patient: Patient,
        pk_propofol: Optional[Any] = None,
        pd_propofol: Optional[Any] = None,
    ) -> None:
        """Initialize the haemodynamic PD model.

        Parameters
        ----------
        patient : Patient
            Patient descriptor used for covariates and baseline values.
        pk_propofol : Any, optional
            Propofol PK model instance providing compartment and rate parameters.
        pd_propofol : Any, optional
            Propofol PD model instance providing effect-site parameters.
        """
        self.patient = patient
        self.patient_ref = EleveldPatient(age=35,
                                          height=170,
                                          weight=70,
                                          sex='male',
                                          opiates=False)
        self.pk_propofol = pk_propofol
        self.pd_propofol = pd_propofol

        self._set_params()

    def _set_params(self):

        # Parameters (from NONMEM file)
        self.Theta1 = 0.0754  # kout (min-1)
        self.Theta2 = 80.4  # base_SV (mL)
        self.Theta3 = 53.6  # base_HR (bpm)
        self.Theta4 = 0.0176  # base_TPR (dyn.s.cm-5)
        self.Theta5 = 0.338  # C50_SV_PROP(µg.mL-1)
        self.Theta6 = -0.235  # Emax_SV_PROP
        self.Theta7 = 2.96   # C50_TPR_PROP (µg.mL-1)
        self.Theta8 = -0.851  # EMAX_TPR_PROP
        self.Theta9 = 0.965  # FB
        self.Theta10 = 0.312  # FIX ; HR_SV
        self.Theta11 = 0.0358  # K
        self.Theta12 = 0.188  # ANX
        self.Theta13 = 1.76  # GAMMA
        self.Theta14 = 0.0258  # AGE_Emax_SV

        self.base_sv = self.Theta2 if self.patient.base_sv is None else self.patient.base_sv
        self.base_hr = self.Theta3 if self.patient.base_hr is None else self.patient.base_hr
        self.base_tpr = self.Theta4 if self.patient.base_tpr is None else self.patient.base_tpr

        self.kin_hr = self.Theta1 * self.base_hr
        self.kin_sv = self.Theta1 * self.base_sv
        self.kin_tpr = self.Theta1 * self.base_tpr

        # TODO check
        self.Emax_sv = self.Theta6 * np.exp(self.Theta14 * (self.patient.age - 35))

    def sv(self, sv_ast: float, hr: float) -> float:
        """Compute stroke volume from latent stroke volume and heart rate.

        Parameters
        ----------
        sv_ast : float
            Latent stroke volume state.
        hr : float
            Heart rate in beats per minute.

        Returns
        -------
        float
            Stroke volume.
        """
        return sv_ast * (1 - self.Theta10 * np.log(hr / (self.base_hr * (1 + self.Theta12))))

    def RMAP(self, sv_ast: float, hr: float, tpr: float) -> float:
        """Compute normalized relative mean arterial pressure (RMAP).

        Parameters
        ----------
        sv_ast : float
            Latent stroke volume state.
        hr : float
            Heart rate in beats per minute.
        tpr : float
            Total peripheral resistance state.

        Returns
        -------
        float
            Relative MAP normalized to baseline.
        """
        denominator = self.base_sv * self.base_hr * (1 + self.Theta12) * self.base_tpr
        numerator = self.sv(sv_ast, hr) * hr * tpr
        return numerator / denominator

    def _f_sigmoid(self, x, y, a):
        return np.float_power(x, a) / (np.float_power(y, a) + np.float_power(x, a))

    def eff_sv(self, x: float) -> float:
        """Compute propofol effect on stroke volume dynamics.

        Parameters
        ----------
        x : float
            Plasma propofol concentration.

        Returns
        -------
        float
            Fractional effect term for stroke volume dynamics.
        """
        return self.Emax_sv * self._f_sigmoid(x, self.Theta5, 1)

    def eff_tpr(self, x: float) -> float:
        """Compute propofol effect on total peripheral resistance dynamics.

        Parameters
        ----------
        x : float
            Plasma propofol concentration.

        Returns
        -------
        float
            Fractional effect term for TPR dynamics.
        """
        return self.Theta8 * self._f_sigmoid(x, self.Theta7, self.Theta13)

    def derivative(
        self,
        X: Sequence[float],
        t: float,
        V1: float,
        k10: float,
        k12: float,
        k13: float,
        k21: float,
        k31: float,
        ke0: float,
        dotA0: Callable[[float], float],
    ) -> NDArray[np.float64]:
        """Derivatives for coupled propofol PK and haemodynamic PD states for use in ode solver.

        Parameters
        ----------
        X : Sequence[float]
            State vector ``[A1, A2, A3, Ce, sv_ast, hr_ast, tpr, tde]``.
        t : float
            Time point in minutes.
        V1 : float
            Central compartment volume.
        k10 : float
            Elimination rate constant from central compartment.
        k12 : float
            Distribution rate from central to peripheral compartment 2.
        k13 : float
            Distribution rate from central to peripheral compartment 3.
        k21 : float
            Redistribution rate from compartment 2 to central.
        k31 : float
            Redistribution rate from compartment 3 to central.
        ke0 : float
            Effect-site transfer rate constant.
        dotA0 : Callable[[float], float]
            External input function returning infusion/bolus rate at time ``t``.

        Returns
        -------
        numpy.ndarray
            Derivatives of all state variables in the same order as ``X``.
        """
        (A1, A2, A3, Ce,  # propofol pkpd params
         sv_ast, hr_ast, tpr, tde)  = X   # heamo pd params

        # Drug effect (pk/pd)
        Cp = A1 / self.pk_propofol.V1
        dotA1 = (- (k10 + k12 + k13) * A1
                 + k21 * A2 + k31 * A3
                 + dotA0(t)
        )
        dotA2 = k12 * A1 - k21 * A2
        dotA3 = k13 * A1 - k31 * A3

        dotCe = ke0 * (A1 / V1 - Ce)

        # Guard HR and SV from going into extreme values
        hr_ast = max(hr_ast, 1)
        sv_ast = max(sv_ast, 1)

        # Heamo effects (pd)
        hr = hr_ast + tde
        rmap = self.RMAP(sv_ast, hr, tpr)


        dot_sv_ast = (self.kin_sv
                      * np.float_power(rmap, -self.Theta9)
                      * (1 + self.eff_sv(Cp))
                      - self.Theta1 * sv_ast
        )
        dot_hr_ast = (self.kin_hr
                      * np.float_power(rmap, -self.Theta9)
                      - self.Theta1 * hr_ast
        )
        dot_tpr = (self.kin_tpr
                   * np.float_power(rmap, -self.Theta9)
                   * (1 + self.eff_tpr(Cp))
                   - self.Theta1 * tpr
        )
        dot_tde = -self.Theta11 * tde

        return np.array([dotA1, dotA2, dotA3, dotCe, dot_sv_ast, dot_hr_ast, dot_tpr, dot_tde])

    def solve_ode(
        self,
        t: Optional[NDArray[np.float64]] = None,
        y0: Optional[Sequence[float]] = None,
        dosing: Optional[Any] = None,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """Solve the coupled propofol PK and haemodynamic PD ODE system.

        Parameters
        ----------
        t : numpy.ndarray, optional
            Time grid in minutes. If ``None``, defaults to 15 minutes at 1-second steps.
        y0 : Sequence[float], optional
            Initial state vector.
        dosing : Any, optional
            Dosing object exposing ``dotA0`` and ``tcrit`` attributes.

        Returns
        -------
        tuple of numpy.ndarray
            Time series arrays ``(A1, A2, A3, Ce, sv, hr, MAP, tde)``.
        """
        if t is None:
            t = np.linspace(0, 15, 15*60+1)  # 15 minutes with 1s steps
        if dosing is None:
            def dotA0(x):
                return 0
            logging.debug("dotA0 set to 0")
            tcrit = None
        else:
            dotA0 = dosing.dotA0
            tcrit = dosing.tcrit

        if y0 is None:
            y0 = [0, 0, 0, 0,
                  self.base_sv, # base sv
                  self.base_hr,  # base hr
                  self.base_tpr, # base tpr
                  self.Theta12 * self.base_hr]

        res = integrate.odeint(self.derivative, y0, t,
                               args = (self.pk_propofol.V1,
                                       self.pk_propofol.k10,
                                       self.pk_propofol.k12,
                                       self.pk_propofol.k13,
                                       self.pk_propofol.k21,
                                       self.pk_propofol.k31,
                                       self.pd_propofol.ke0,
                                       dotA0),
                                tcrit=tcrit
        )
        (A1, A2, A3, Ce,  # units [mg, mg, mg, mg / L]
         sv_ast, hr_ast, tpr, tde) = res.T
        hr = hr_ast + tde
        sv = self.sv(sv_ast, hr)

        MAP = sv * tpr * hr
        return A1, A2, A3, Ce, sv, hr, MAP, tde  # [ml], [min^-1], [mm Hg]

