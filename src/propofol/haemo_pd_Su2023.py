import logging
from typing import Any, Optional

import numpy as np
from scipy import integrate

from propofol.patient import EleveldPatient
from propofol.protocols import Patient

logger = logging.getLogger(__name__)


class SuHaemoPD2023:
    """Su haemodynamic interaction PD model for propofol + remifentanil.
    Source: DOI: 10.1016/j.bja.2023.04.043 

    Based on the NONMEM model structure.

    Notes
    -----
    State vector
    ------------
    X = (
        A1, A2, A3,           # propofol PK compartment amounts
        A4, A5, A6,           # remifentanil PK compartment amounts
        sv_ast, hr_ast, tpr,  # haemodynamic turnover states
        anx_sv, anx_hr        # transient anxiety states
    )

    Derived variables
    -----------------
    Cp_prop = A1 / V1_prop
    Cp_remi = A4 / V1_remi

    DSV = sv_ast + anx_sv
    DHR = hr_ast + anx_hr

    SV  = DSV * (1 - HR_SV * log(DHR / Abase_HR))
    HR  = DHR
    MAP = SV * HR * TPR

    Random effects
    --------------
    ETA1 : base_SV
    ETA2 : base_TPR
    ETA3 : base_HR
    ETA4 : C50_TPR_PROP
    ETA5 : EMAX_TPR_REMI
    ETA6 : SL_HR_REMI
    ETA7 : SL_SV_REMI
    """

    def __init__(
        self,
        patient: Patient,
        pk_propofol: Optional[Any] = None,
        pd_propofol: Optional[Any] = None,
        pk_remifentanil: Optional[Any] = None,
        use_bsv: bool = False,
    ):
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
        # ------------------------
        # Fixed-effect THETAs
        # ------------------------
        self.Theta1 = 0.072     # kout
        self.Theta2 = 82.2      # base_SV
        self.Theta3 = 56.1      # base_HR
        self.Theta4 = 0.0163    # base_TPR
        self.Theta5 = 0.067     # k (anxiety decay)
        self.Theta6 = 0.121     # ANXHR
        self.Theta7 = 0.0899    # ANXSV
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
        # ------------------------
        # Baseline haemodynamics
        # ------------------------
        # baseline SV
        if getattr(self.patient, "base_sv", None) is not None:
            self.base_sv = float(self.patient.base_sv)
        elif getattr(self.patient, "base_pp", None) is not None:
            self.base_sv = 1.5 * float(self.patient.base_pp)
        else:
            self.base_sv = self.Theta2 * np.exp(self.eta1)

         # baseline HR
        if getattr(self.patient, "base_hr", None) is not None:
            self.base_hr = float(self.patient.base_hr)
        else:
            self.base_hr = self.Theta3 * np.exp(self.eta3)

        # ------------------------
        # Anxiety / feedback effects
        # ------------------------
        self.k = self.Theta5
        self.ANXHR = self.Theta6
        self.ANXSV = self.Theta7
        self.FB = self.Theta18
        self.HR_SV = self.Theta19

        self.Abase_HR = self.base_hr * (1.0 + self.ANXHR)
        self.Abase_SV = self.base_sv * (1.0 + self.ANXSV)

        # baseline TPR / MAP
        if getattr(self.patient, "base_tpr", None) is not None:
            self.base_tpr = float(self.patient.base_tpr)
            self.base_MAP = self.Abase_SV * self.Abase_HR * self.base_tpr
        elif getattr(self.patient, "base_map", None) is not None:
            self.base_MAP = float(self.patient.base_map)
            self.base_tpr = self.base_MAP / (self.Abase_SV * self.Abase_HR)
        else:
            self.base_tpr = self.Theta4 * np.exp(self.eta2)
            self.base_MAP = self.Abase_SV * self.Abase_HR * self.base_tpr

        self.kin_sv = self.Theta1 * self.base_sv
        self.kin_hr = self.Theta1 * self.base_hr
        self.kin_tpr = self.Theta1 * self.base_tpr

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
        # Interaction effects
        # ------------------------
        self.INT_HR = self.Theta21
        self.C50_INT_HR = self.Theta22
        self.INT_TPR = self.Theta23
        self.INT_SV_PROP = self.Theta24

    def sv(self, dsv, dhr):
        ratio = np.asarray(dhr, dtype=float) / self.Abase_HR
        return np.asarray(dsv, dtype=float) * (1.0 - self.HR_SV * np.log(ratio))

    def RMAP(self, dsv, dhr, tpr):
        numerator = self.sv(dsv, dhr) * np.asarray(dhr, dtype=float) * np.asarray(tpr, dtype=float)
        return numerator / self.base_MAP

    def _f_sigmoid(self, x, y, a):
        x = np.maximum(np.asarray(x, dtype=float), 0.0)
        return np.float_power(x, a) / (np.float_power(y, a) + np.float_power(x, a))

    # ------------------------
    # Drug effect functions
    # ------------------------
    def eff_sv_prop(self, cp_prop):
        cp_prop = max(float(cp_prop), 0.0)
        result = self.EMAX_SV_PROP * cp_prop / (cp_prop + self.C50_SV_PROP) if cp_prop > 0 else 0.0
        return max(result, -0.999)

    def eff_tpr_prop(self, cp_prop, cp_remi):
        cp_prop = max(float(cp_prop), 0.0)
        cp_remi = max(float(cp_remi), 0.0)

        result = (self.EMAX_TPR_PROP + self.INT_TPR * cp_remi / (self.C50_TPR_REMI + cp_remi)) * self._f_sigmoid(cp_prop, self.C50_TPR_PROP, self.GAMP) if cp_prop > 0 else 0.0
        return max(float(result), -0.999)

    def eff_tpr_remi(self, cp_remi):
        cp_remi = max(float(cp_remi), 0.0)
        result = (
            self.EMAX_TPR_REMI * self._f_sigmoid(cp_remi, self.C50_TPR_REMI, self.GAMR)
            if cp_remi > 0 else 0.0
        )
        return min(float(result), 0.999)

    def eff_hr_remi(self, cp_prop, cp_remi):
        cp_prop = max(float(cp_prop), 0.0)
        cp_remi = max(float(cp_remi), 0.0)
        result = (
            self.SL_HR_REMI + self.INT_HR * cp_prop / (self.C50_INT_HR + cp_prop)
        ) * cp_remi if cp_remi > 0 else 0.0
        return min(float(result), 0.999)

    def eff_sv_remi(self, cp_prop, cp_remi):
        cp_prop = max(float(cp_prop), 0.0)
        cp_remi = max(float(cp_remi), 0.0)
        result = (
            self.SL_SV_REMI + self.INT_SV_PROP * cp_prop / (self.C50_SV_PROP + cp_prop)
        ) * cp_remi if cp_remi > 0 else 0.0
        return min(float(result), 0.999)

    def derivative(
        self,
        X,
        t,
        V1_prop, k10_prop, k12_prop, k13_prop, k21_prop, k31_prop, dotA0_prop,
        V1_remi, k10_remi, k12_remi, k13_remi, k21_remi, k31_remi, dotA0_remi,
    ):
        (
            A1, A2, A3,
            A4, A5, A6,
            sv_ast, hr_ast, tpr,
            anx_sv, anx_hr,
        ) = X

        # ------------------------
        # PK
        # ------------------------
        cp_prop = A1 / V1_prop
        cp_remi = A4 / V1_remi

        dotA1 = (
            - (k10_prop + k12_prop + k13_prop) * A1
            + k21_prop * A2
            + k31_prop * A3
            + dotA0_prop(t)
        )
        dotA2 = k12_prop * A1 - k21_prop * A2
        dotA3 = k13_prop * A1 - k31_prop * A3

        dotA4 = (
            - (k10_remi + k12_remi + k13_remi) * A4
            + k21_remi * A5
            + k31_remi * A6
            + dotA0_remi(t)
        )
        dotA5 = k12_remi * A4 - k21_remi * A5
        dotA6 = k13_remi * A4 - k31_remi * A6

        # ------------------------
        # Derived haemodynamics
        # ------------------------

        dsv = sv_ast + anx_sv
        dhr = hr_ast + anx_hr
        rmap = self.RMAP(dsv, dhr, tpr)

        # ------------------------
        # Drug effects
        # ------------------------
        sv_prop = self.eff_sv_prop(cp_prop)
        tpr_prop = self.eff_tpr_prop(cp_prop, cp_remi)
        tpr_remi = self.eff_tpr_remi(cp_remi)
        hr_remi = self.eff_hr_remi(cp_prop, cp_remi)
        sv_remi = self.eff_sv_remi(cp_prop, cp_remi)

        # ------------------------
        # Haemodynamic ODEs
        # ------------------------
        dot_sv_ast = (
            self.kin_sv
            * np.float_power(rmap, -self.FB)
            * (1.0 + sv_prop)
            - sv_ast * self.Theta1 * (1.0 - sv_remi)
        )

        dot_hr_ast = (
            self.kin_hr
            * np.float_power(rmap, -self.FB)
            - hr_ast * self.Theta1 * (1.0 - hr_remi)
        )

        dot_tpr = (
            self.kin_tpr
            * np.float_power(rmap, -self.FB)
            * (1.0 + tpr_prop)
            - tpr * self.Theta1 * (1.0 - tpr_remi)
        )

        dot_anx_sv = -self.k * anx_sv
        dot_anx_hr = -self.k * anx_hr

        return np.array([
            dotA1, dotA2, dotA3,
            dotA4, dotA5, dotA6,
            dot_sv_ast, dot_hr_ast, dot_tpr,
            dot_anx_sv, dot_anx_hr,
        ], dtype=float)

    def solve_ode(
        self,
        t=np.linspace(0, 15, 15 * 60 + 1),
        X0=None,
        dosing_prop=None,
        dosing_remi=None,
    ):
        if self.pk_propofol is None:
            raise ValueError("pk_propofol must be provided.")
        if self.pk_remifentanil is None:
            raise ValueError("pk_remifentanil must be provided.")

        # ------------------------
        # Dosing inputs
        # ------------------------
        if dosing_prop is None:
            dotA0_prop = lambda x: 0.0
            tcrit_prop = None
            logger.debug("dotA0_prop set to 0")
        else:
            dotA0_prop = dosing_prop.dotA0
            tcrit_prop = getattr(dosing_prop, "tcrit", None)

        if dosing_remi is None:
            dotA0_remi = lambda x: 0.0
            tcrit_remi = None
            logger.debug("dotA0_remi set to 0")
        else:
            dotA0_remi = dosing_remi.dotA0
            tcrit_remi = getattr(dosing_remi, "tcrit", None)

        if tcrit_prop is None and tcrit_remi is None:
            tcrit = None
        else:
            crit_list = []
            if tcrit_prop is not None:
                crit_list.extend(np.atleast_1d(tcrit_prop).tolist())
            if tcrit_remi is not None:
                crit_list.extend(np.atleast_1d(tcrit_remi).tolist())
            tcrit = np.array(sorted(set(crit_list)), dtype=float)

        # ------------------------
        # Initial conditions
        # ------------------------
        if X0 is None:
            X0 = [
                0.0, 0.0, 0.0,                  # propofol PK
                0.0, 0.0, 0.0,                  # remifentanil PK
                self.base_sv,                   # sv_ast
                self.base_hr,                   # hr_ast
                self.base_tpr,                  # tpr
                self.base_sv * self.ANXSV,      # anx_sv
                self.base_hr * self.ANXHR,      # anx_hr
            ]

        res = integrate.odeint(
            self.derivative,
            X0,
            t,
            args=(
                self.pk_propofol.V1,
                self.pk_propofol.k10,
                self.pk_propofol.k12,
                self.pk_propofol.k13,
                self.pk_propofol.k21,
                self.pk_propofol.k31,
                dotA0_prop,
                self.pk_remifentanil.V1,
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
            A1, A2, A3,
            A4, A5, A6,
            sv_ast, hr_ast, tpr,
            anx_sv, anx_hr,
        ) = res.T

        cp_prop = A1 / self.pk_propofol.V1
        cp_remi = A4 / self.pk_remifentanil.V1

        dsv = sv_ast + anx_sv
        dhr = hr_ast + anx_hr

        sv = self.sv(dsv, dhr)
        hr = dhr
        MAP = sv * tpr * hr
        RMAP = MAP / self.base_MAP

        sv_prop = np.array([self.eff_sv_prop(cp) for cp in cp_prop], dtype=float)
        tpr_prop = np.array(
            [self.eff_tpr_prop(cp, cr) for cp, cr in zip(cp_prop, cp_remi)],
            dtype=float,
        )
        tpr_remi = np.array([self.eff_tpr_remi(cr) for cr in cp_remi], dtype=float)
        hr_remi = np.array(
            [self.eff_hr_remi(cp, cr) for cp, cr in zip(cp_prop, cp_remi)],
            dtype=float,
        )
        sv_remi = np.array(
            [self.eff_sv_remi(cp, cr) for cp, cr in zip(cp_prop, cp_remi)],
            dtype=float,
        )

        return (
            A1, A2, A3,                # propofol PK states
            A4, A5, A6,                # remifentanil PK states
            cp_prop, cp_remi,          # plasma concentrations
            sv, hr, MAP, tpr, RMAP,    # haemodynamics
            anx_sv, anx_hr,            # anxiety states
            sv_prop, tpr_prop,         # propofol effects
            sv_remi, hr_remi, tpr_remi # remifentanil effects
        )