import logging
from typing import Any, Callable, Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import integrate

from propofol.patient import EleveldPatient
from propofol.protocols import Patient

logger = logging.getLogger(__name__)


class SuHaemoPD():
    """Su haemodynamic interaction PD model for propofol + remifentanil.
    Source: DOI: 10.1016/j.bja.2023.04.043

    Notes
    -----
    - Based on the NONMEM model structure.
    """
    def __init__(
        self,
        patient: Patient,
        pk_propofol: Optional[Any] = None,
        pd_propofol: Optional[Any] = None,
        pk_remifentanil: Optional[Any] = None,
        use_bsv: bool = False,
    ):
        """Initialize the haemodynamic interaction PD model.

        Parameters
        ----------
        patient : Patient
            Patient descriptor used for covariates and baseline values.
        pk_propofol : Any, optional
            Propofol PK model instance providing compartment and rate parameters.
        pd_propofol : Any, optional
            Propofol PD model instance providing effect-site parameters.
        pk_remifentanil : Any, optional
            Remifentanil PK model instance providing compartment and rate parameters.
        use_bsv : bool, optional
            Flag indicating whether to use between-subject variability (default: False).
        """
        self.patient = patient
        self.patient_ref = EleveldPatient(
            age=35,
            height=170,
            weight=70,
            sex="male",
            opiates=False,
        )
        self.use_bsv = use_bsv
        self.pk_propofol = pk_propofol
        self.pd_propofol = pd_propofol
        self.pk_remifentanil = pk_remifentanil
        self.rng = np.random.default_rng()

        self._set_params()

    def _set_params(self):
        # Parameters (from NONMEM file)
        self.Theta1 = 0.072     # kout
        self.Theta2 = 82.2      # base_SV
        self.Theta3 = 56.1      # base_HR
        self.Theta4 = 0.0163    # base_TPR
        self.Theta5 = 0.067     # k (anxiety decay)
        self.Theta6 = 0.121     # ltde_hr
        self.Theta7 = 0.0899    # ltde_sv
        self.Theta8 = 0.44      # C50_SV_PROP
        self.Theta9 = 3.21      # C50_TPR_PROP
        self.Theta10 = -0.154   # EMAX_SV_PROP
        self.Theta11 = -0.778   # EMAX_TPR_PROP
        self.Theta12 = 1.83     # GAMP
        self.Theta13 = 4.59     # C50_TPR_REMI
        self.Theta14 = -1.0     # EMAX_TPR_REMI
        self.Theta15 = 0.0327   # SL_HR_REMI
        self.Theta16 = 0.0581   # SL_SV_REMI
        self.Theta17 = 1.0      # GAMR
        self.Theta18 = 0.661    # FB
        self.Theta19 = 0.312    # HR_SV
        self.Theta20 = 0.0333   # AGE_Emax_SV
        self.Theta21 = -0.119   # INT_HR
        self.Theta22 = 0.196    # C50_INT_HR
        self.Theta23 = 1.0      # INT_TPR
        self.Theta24 = -0.212   # INT_SV_PROP

        if self.use_bsv:
            self.draw_eta()
        else:
            self.reset_eta()

    def draw_eta(self):
        """Draw PD etas from the NONMEM OMEGA structure."""
        omega_block_3 = np.array([
            [0.0328, -0.0244,  0.0],
            [-0.0244, 0.0528, -0.0233],
            [0.0,    -0.0233,  0.0242],
        ], dtype=float)

        omega_block_2 = np.array([
            [0.00382, 0.00329],
            [0.00329, 0.00868],
        ], dtype=float)

        self.eta1, self.eta2, self.eta3 = self.rng.multivariate_normal(
            mean=np.zeros(3, dtype=float),
            cov=omega_block_3,
        )

        self.eta4 = self.rng.normal(0.0, np.sqrt(0.44))
        self.eta5 = self.rng.normal(0.0, np.sqrt(0.449))

        self.eta6, self.eta7 = self.rng.multivariate_normal(
            mean=np.zeros(2, dtype=float),
            cov=omega_block_2,
        )

        self._update_pd()

    def reset_eta(self):
        """Reset PD etas to zero."""
        self.eta1 = 0.0
        self.eta2 = 0.0
        self.eta3 = 0.0
        self.eta4 = 0.0
        self.eta5 = 0.0
        self.eta6 = 0.0
        self.eta7 = 0.0

        self._update_pd()

    def _update_pd(self):
        """Update PD parameters and derived values based on current etas and patient covariates."""
        # ------------------------
        # Baseline haemodynamics
        # ------------------------
        # baseline SV
        if self.patient.base_sv is not None:
            self.base_sv = self.patient.base_sv * np.exp(self.eta1)
        else:
            self.base_sv = self.Theta2 * np.exp(self.eta1)

        # baseline HR
        if self.patient.base_hr is not None:
            self.base_hr = self.patient.base_hr * np.exp(self.eta3)
        else:
            self.base_hr = self.Theta3 * np.exp(self.eta3)

        # baseline TPR
        if self.patient.base_tpr is not None:
            self.base_tpr = self.patient.base_tpr * np.exp(self.eta2)
        else:
            self.base_tpr = self.Theta4 * np.exp(self.eta2)

        self.kin_sv = self.Theta1 * self.base_sv
        self.kin_hr = self.Theta1 * self.base_hr
        self.kin_tpr = self.Theta1 * self.base_tpr

        # ------------------------
        # Anxiety effects
        # ------------------------
        self.k = self.Theta5
        self.ltde_hr = self.Theta6
        self.ltde_sv = self.Theta7

        self.anxsv = self.base_sv * (1.0 + self.ltde_sv)
        self.anxhr = self.base_hr * (1.0 + self.ltde_hr)
        self.base_map = self.base_sv * self.base_hr * self.base_tpr

        # ------------------------
        # Propofol effects
        # ------------------------
        self.C50_SV_PROP = self.Theta8
        self.C50_TPR_PROP = self.Theta9 * np.exp(self.eta4)
        self.EMAX_SV_PROP = self.Theta10 * np.exp(self.Theta20 * (self.patient.age - 35.0))
        self.EMAX_TPR_PROP = self.Theta11
        self.GAMP = self.Theta12

        # ------------------------
        # Remifentanil effects
        # ------------------------
        self.C50_TPR_REMI = self.Theta13
        self.EMAX_TPR_REMI = self.Theta14 + self.eta5
        self.SL_HR_REMI = self.Theta15 + self.eta6
        self.SL_SV_REMI = self.Theta16 + self.eta7
        self.GAMR = self.Theta17

        # ------------------------
        # Feedback effects
        # ------------------------
        self.FB = self.Theta18
        self.HR_SV = self.Theta19

        # ------------------------
        # Interaction effects
        # ------------------------
        self.INT_HR = self.Theta21
        self.C50_INT_HR = self.Theta22
        self.INT_TPR = self.Theta23
        self.INT_SV_PROP = self.Theta24

    def sv(self, sv_ast: float, hr_ast: float) -> float:
        """Calculate stroke volume (SV) based on the model's feedback mechanism."""
        return sv_ast * (1.0 - self.HR_SV * np.log(hr_ast / self.base_hr))

    def RMAP(self, sv_ast: float, hr_ast: float, tpr: float) -> float:
        """Calculate the relative change of MAP to baseline MAP (RMAP) based on the model's
        feedback mechanism."""
        amap = self.sv(sv_ast, hr_ast) * hr_ast * tpr
        return amap / self.base_map

    def _f_sigmoid(self, x, y, a):
        """Generalized sigmoid function for drug effects."""
        return np.float_power(x, a) / (np.float_power(y, a) + np.float_power(x, a))

    def eff_sv_prop(self, cp_prop):
        """Compute effect of propofol on stroke volume (SV)."""
        result = self.EMAX_SV_PROP * cp_prop / (cp_prop + self.C50_SV_PROP)
        return max(result, -0.999)

    def eff_tpr_prop(self, cp_prop, cp_remi):
        """Compute effect of propofol on total peripheral resistance (TPR)
        including interaction with remifentanil."""
        tpr_int = (self.EMAX_TPR_PROP + (self.INT_TPR * cp_remi) / (self.C50_TPR_REMI + cp_remi))
        sigmoid = self._f_sigmoid(cp_prop, self.C50_TPR_PROP, self.GAMP)
        result = tpr_int * sigmoid
        return max(float(result), -0.999)

    def eff_tpr_remi(self, cp_remi):
        """Compute effect of remifentanil on total peripheral resistance (TPR)."""
        result = self.EMAX_TPR_REMI * self._f_sigmoid(cp_remi, self.C50_TPR_REMI, self.GAMR)
        return min(float(result), 0.999)

    def eff_hr_remi(self, cp_prop, cp_remi):
        """Compute effect of remifentanil on heart rate (HR) including interaction with propofol."""
        result = (self.SL_HR_REMI + (self.INT_HR * cp_prop) / (self.C50_INT_HR + cp_prop)) * cp_remi
        return min(float(result), 0.999)

    def eff_sv_remi(self, cp_prop, cp_remi):
        """Compute effect of remifentanil on stroke volume (SV),
        including interaction with propofol."""
        result = (self.SL_SV_REMI + (self.INT_SV_PROP * cp_prop) / (self.C50_SV_PROP + cp_prop)
                  ) * cp_remi
        return min(float(result), 0.999)

    def derivative(
        self,
        X: Sequence[float],
        t: float,
        V1_prop: float, k10_prop: float, k12_prop: float, k13_prop: float,
        k21_prop: float, k31_prop: float, ke0_prop: float, dotA0_prop: Callable[[float], float],
        k10_remi: float, k12_remi: float, k13_remi: float,
        k21_remi: float, k31_remi: float, dotA0_remi: Callable[[float], float],
    )-> NDArray[np.float64]:
        """Derivatives for coupled propofol/ remifentanil PK and haemodynamic PD states for use
        in ode solver.

        Parameters
        ----------
        X : Sequence[float]
            State vector ``[A1, A2, A3, Ce_prop, A4, A5, A6, sv_ast, hr_ast, tpr, tde_decay]``
            where ``tde_decay`` mirrors NONMEM A(10): starts at 1, decays to 0 at rate k.
        t : float
            Time point in minutes.
        V1_prop : float
            Volume of central compartment for propofol.
        k10_prop : float
            Elimination rate constant for propofol from central compartment.
        k12_prop : float
            Distribution rate from central to peripheral compartment 2 for propofol.
        k13_prop : float
            Distribution rate from central to peripheral compartment 3 for propofol.
        k21_prop : float
            Redistribution rate from compartment 2 to central for propofol.
        k31_prop : float
            Redistribution rate from compartment 3 to central for propofol.
        ke0_prop : float
            Effect-site transfer rate constant for propofol.
        dotA0_prop : Callable[[float], float]
            External input function returning infusion/bolus rate at time ``t`` for propofol.
        k10_remi : float
            Elimination rate constant for remifentanil from central compartment.
        k12_remi : float
            Distribution rate from central to peripheral compartment 2 for remifentanil.
        k13_remi : float
            Distribution rate from central to peripheral compartment 3 for remifentanil.
        k21_remi : float
            Redistribution rate from compartment 2 to central for remifentanil.
        k31_remi : float
            Redistribution rate from compartment 3 to central for remifentanil.
        dotA0_remi : Callable[[float], float]
            External input function returning infusion/bolus rate at time ``t`` for remifentanil.

        Returns
        -------
        numpy.ndarray
            Derivatives of all state variables in the same order as ``X``.
        """
        (
            A1, A2, A3, Ce_prop, # propofol PKPD states
            A4, A5, A6, # remifentanil PK states
            sv_ast, hr_ast, tpr, # haemodynamic states
            tde_decay, # time-dependent effect decay factor (1→0, mirrors NONMEM A(10))
        ) = X

        # ------------------------
        # PK
        # ------------------------
        cp_prop = A1 / self.pk_propofol.V1
        cp_remi = A4 / self.pk_remifentanil.V1

        dotA1 = (
            - (k10_prop + k12_prop + k13_prop) * A1
            + k21_prop * A2
            + k31_prop * A3
            + dotA0_prop(t)
        )
        dotA2 = k12_prop * A1 - k21_prop * A2
        dotA3 = k13_prop * A1 - k31_prop * A3
        dotCe_prop = ke0_prop * (A1 / V1_prop - Ce_prop)

        dotA4 = (
            - (k10_remi + k12_remi + k13_remi) * A4
            + k21_remi * A5
            + k31_remi * A6
            + dotA0_remi(t)
        )
        dotA5 = k12_remi * A4 - k21_remi * A5
        dotA6 = k13_remi * A4 - k31_remi * A6

        # ------------------------
        # Guard HR and SV from going into extreme values
        # ------------------------
        hr_ast = max(hr_ast, 1)
        sv_ast = max(sv_ast, 1)

        # ------------------------
        # Derived haemodynamics
        # ------------------------

        rmap = self.RMAP(sv_ast, hr_ast, tpr)

        # ------------------------
        # Haemodynamic ODEs
        # ------------------------
        dot_sv_ast = (
            self.kin_sv
            * np.float_power(rmap, -self.FB)
            * (1.0 + self.eff_sv_prop(cp_prop))
            * np.exp(tde_decay * self.ltde_sv)   # NONMEM: EXP(A(10)*TDE_SV)
            - sv_ast * self.Theta1 * (1.0 - self.eff_sv_remi(cp_prop, cp_remi))
        )

        dot_hr_ast = (
            self.kin_hr
            * np.float_power(rmap, -self.FB)
            * np.exp(tde_decay * self.ltde_hr)   # NONMEM: EXP(A(10)*TDE_HR)
            - hr_ast * self.Theta1 * (1.0 - self.eff_hr_remi(cp_prop, cp_remi))
        )

        dot_tpr = (
            self.kin_tpr
            * np.float_power(rmap, -self.FB)
            * (1.0 + self.eff_tpr_prop(cp_prop, cp_remi))
            - tpr * self.Theta1 * (1.0 - self.eff_tpr_remi(cp_remi))
        )

        dot_tde_decay = -self.k * tde_decay   # NONMEM: DADT(10) = -k*A(10)

        return np.array([
            dotA1, dotA2, dotA3, dotCe_prop,
            dotA4, dotA5, dotA6,
            dot_sv_ast, dot_hr_ast, dot_tpr,
            dot_tde_decay,
        ], dtype=float)

    @staticmethod
    def _resolve_dosing(
        dosing: Optional[Any],
        default_label: str,
    ) -> tuple[Callable[[float], float], Optional[Any]]:
        """Return input-rate function and critical times for a dosing object."""
        if dosing is None:
            def dot_a0(_: float) -> float:
                return 0.0

            logger.debug("%s set to 0", default_label)
            return dot_a0, None

        return dosing.dotA0, dosing.tcrit

    @staticmethod
    def _merge_tcrit(
        tcrit_prop: Optional[Any],
        tcrit_remi: Optional[Any],
    ) -> Optional[list[Any]]:
        """Merge and deduplicate critical ODE times from two dosing schedules."""
        tcrit = []
        if tcrit_prop is not None:
            tcrit.extend(tcrit_prop if isinstance(tcrit_prop, (list, np.ndarray)) else [tcrit_prop])
        if tcrit_remi is not None:
            tcrit.extend(tcrit_remi if isinstance(tcrit_remi, (list, np.ndarray)) else [tcrit_remi])
        return sorted(set(tcrit)) if tcrit else None

    def _default_initial_state(self) -> list[float]:
        """Build the default initial state vector for the coupled PK/PD model."""
        return [
            0.0, 0.0, 0.0, 0.0,                        # propofol PK/PD
            0.0, 0.0, 0.0,                              # remifentanil PK
            self.base_sv * np.exp(self.ltde_sv),        # sv_ast — starts elevated (NONMEM A_0(7))
            self.base_hr * np.exp(self.ltde_hr),        # hr_ast — starts elevated (NONMEM A_0(8))
            self.base_tpr,                              # tpr
            1.0,                                        # tde_decay (NONMEM A(10), decays 1 → 0)
        ]

    def solve_ode(
        self,
        t: Optional[NDArray[np.float64]] = None,
        y0: Optional[Sequence[float]] = None,
        dosing_prop: Optional[Any] = None,
        dosing_remi: Optional[Any] = None,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        ]:
        """Solve the coupled PK/PD ODEs for propofol and remifentanil with interactions.

        Parameters
        ----------
        t : numpy.ndarray, optional
            Time grid in minutes. If ``None``, defaults to 15 minutes at 1-second steps.
        y0 : Sequence[float], optional
            Initial state vector.
        dosing_prop : Any, optional
            Dosing object for propofol exposing ``dotA0`` and ``tcrit`` attributes.
        dosing_remi : Any, optional
            Dosing object for remifentanil exposing ``dotA0`` and ``tcrit`` attributes.

        Returns
        -------
        tuple of numpy.ndarray
            (A1, A2, A3, Ce_prop, A4, A5, A6, sv_ast, hr_ast, tpr, tde_decay, sv, MAP)
        """
        if self.pk_propofol is None:
            raise ValueError("pk_propofol must be provided.")
        if self.pk_remifentanil is None:
            raise ValueError("pk_remifentanil must be provided.")

        if t is None:
            t = np.linspace(0, 15, 15*60+1)  # 15 minutes with 1s steps

        dotA0_prop, tcrit_prop = self._resolve_dosing(dosing_prop, "dotA0_prop")
        dotA0_remi, tcrit_remi = self._resolve_dosing(dosing_remi, "dotA0_remi")
        tcrit = self._merge_tcrit(tcrit_prop, tcrit_remi)

        # ------------------------
        # Initial conditions
        # ------------------------
        if y0 is None:
            y0 = self._default_initial_state()

        res = integrate.odeint(
            self.derivative,
            y0,
            t,
            args=(
                self.pk_propofol.V1,
                self.pk_propofol.k10,
                self.pk_propofol.k12,
                self.pk_propofol.k13,
                self.pk_propofol.k21,
                self.pk_propofol.k31,
                self.pd_propofol.ke0,
                dotA0_prop,
                self.pk_remifentanil.k10,
                self.pk_remifentanil.k12,
                self.pk_remifentanil.k13,
                self.pk_remifentanil.k21,
                self.pk_remifentanil.k31,
                dotA0_remi,
            ),
            tcrit=tcrit,
        )

        (
            A1, A2, A3, Ce,
            A4, A5, A6,
            sv_ast, hr_ast, tpr,
            tde_decay,
        ) = res.T

        hr = hr_ast   # hr_ast already includes the TDE effect (starts elevated, decays)
        sv = self.sv(sv_ast, hr_ast)
        MAP = sv * hr * tpr

        return A1, A2, A3, Ce, A4, A5, A6, sv_ast, hr_ast, tpr, tde_decay, sv, MAP
