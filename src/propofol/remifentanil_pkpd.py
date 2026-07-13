from __future__ import annotations

import logging
from typing import Optional, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import integrate

from propofol.patient import EleveldPatient
from propofol.protocols import Patient

logger = logging.getLogger(__name__)


class DosingProtocol(Protocol):
    """Minimal protocol for external dosing objects used by the solvers."""

    tcrit: Sequence[float]

    def dotA0(self, t: float) -> float:
        """Return drug input rate in microgram/min at time t in minutes."""
        ...


class EleveldPK:
    """Eleveld-style remifentanil PK model.

    Units
    -----
    Amounts
        microgram (µg)
    Volumes
        litre (L)
    Clearances
        litre/min (L/min)
    Concentrations
        µg/L, numerically equal to ng/mL

    Notes
    -----
    This class is PK-only. The effect-site compartment is implemented in
    :class:`EleveldPD` and :class:`PKPDSolver`.
    """

    def __init__(self, patient: Patient, use_bsv: bool = False) -> None:
        self.patient = patient
        self.patient_ref = EleveldPatient(
            age=35,
            height=170,
            weight=70,
            sex="male",
            opiates=False,
        )
        self.use_bsv = bool(use_bsv)

        self._set_params()

    @staticmethod
    def _f_ageing(x: float, age: float, age_ref: float) -> float:
        return float(np.exp(x * (age - age_ref)))

    def _set_params(self) -> None:
        """Set fixed-effect parameters and initialize random effects."""
        self.Theta1 = 5.81    # V1 (L)
        self.Theta2 = 8.82    # V2 (L)
        self.Theta3 = 5.03    # V3 (L)
        self.Theta4 = 2.58    # CL (L/min)
        self.Theta5 = 1.72    # Q2 (L/min)
        self.Theta6 = 0.124   # Q3 (L/min)
        self.Theta7 = 2.88    # weight at 50% maturation of CL
        self.Theta8 = -0.00554
        self.Theta9 = -0.00327
        self.Theta10 = -0.0315
        self.Theta11 = 0.470
        self.Theta12 = -0.0260

        if self.use_bsv:
            self.draw_eta()
        else:
            self.reset_eta()

        self.reset_epsilon()

    def _update_pk(self) -> None:
        """Recompute PK parameters from patient characteristics and etas."""
        size = self.size()

        self.V1 = (
            self.Theta1
            * size
            * self._f_ageing(self.Theta8, self.patient.age, self.patient_ref.age)
            * np.exp(self.eta1)
        )

        self.V2 = (
            self.Theta2
            * size
            * self._f_ageing(self.Theta9, self.patient.age, self.patient_ref.age)
            * self.ksex()
            * np.exp(self.eta2)
        )

        self.V3 = (
            self.Theta3
            * size
            * self._f_ageing(self.Theta10, self.patient.age, self.patient_ref.age)
            * np.exp(self.Theta12 * (self.patient.weight - self.patient_ref.weight))
            * np.exp(self.eta3)
        )

        self.CL = (
            self.Theta4
            * size ** 0.75
            * (self.kmat(self.patient.weight) / self.kmat(self.patient_ref.weight))
            * self.ksex()
            * self._f_ageing(self.Theta9, self.patient.age, self.patient_ref.age)
            * np.exp(self.eta4)
        )

        self.Q2 = (
            self.Theta5
            * np.float_power(self.V2 / self.Theta2, 0.75)
            * self._f_ageing(self.Theta8, self.patient.age, self.patient_ref.age)
            * self.ksex()
            * np.exp(self.eta5)
        )

        self.Q3 = (
            self.Theta6
            * np.float_power(self.V3 / self.Theta3, 0.75)
            * self._f_ageing(self.Theta8, self.patient.age, self.patient_ref.age)
            * np.exp(self.eta6)
        )

        self.k10 = self.CL / self.V1
        self.k12 = self.Q2 / self.V1
        self.k13 = self.Q3 / self.V1
        self.k21 = self.Q2 / self.V2
        self.k31 = self.Q3 / self.V3

    def draw_eta(self) -> None:
        """Draw independent log-normal PK random effects and update PK."""
        omega2 = np.array([
            0.104,   # V1
            0.115,   # V2
            0.810,   # V3
            0.0197,  # CL
            0.0547,  # Q2
            0.285,   # Q3
        ])

        eta = np.random.multivariate_normal(
            mean=np.zeros_like(omega2),
            cov=np.diag(omega2),
            size=1,
        )

        (
            self.eta1,
            self.eta2,
            self.eta3,
            self.eta4,
            self.eta5,
            self.eta6,
        ) = np.squeeze(eta)

        self._update_pk()

    def reset_eta(self) -> None:
        """Reset all PK random effects to zero and update PK."""
        self.eta1 = 0.0
        self.eta2 = 0.0
        self.eta3 = 0.0
        self.eta4 = 0.0
        self.eta5 = 0.0
        self.eta6 = 0.0
        self._update_pk()

    def draw_epsilon(self) -> None:
        """Draw residual error for concentration observations."""
        self.epsilon = float(np.random.normal(loc=0.0, scale=1.0))

    def reset_epsilon(self) -> None:
        """Reset residual error term to zero."""
        self.epsilon = 0.0

    @staticmethod
    def f_sigmoid(x: float, e50: float, lmbda: float) -> float:
        """Compute sigmoidal scaling function."""
        return float(x**lmbda / (x**lmbda + e50**lmbda))

    def size(self) -> float:
        """Compute body-size scaling using Al-Sallami fat-free mass."""
        patient_ffm = self.f_al_sallami(
            age=self.patient.age,
            weight=self.patient.weight,
            bmi=self.patient.bmi,
            sex=self.patient.sex,
        )
        ref_ffm = self.f_al_sallami(
            age=self.patient_ref.age,
            weight=self.patient_ref.weight,
            bmi=self.patient_ref.bmi,
            sex=self.patient_ref.sex,
        )
        return float(patient_ffm / ref_ffm)

    def kmat(self, weight: float) -> float:
        """Compute maturation factor for clearance."""
        return self.f_sigmoid(weight, self.Theta7, 2.0)

    @staticmethod
    def f_al_sallami(age: float, weight: float, bmi: float, sex: str) -> float:
        """Compute Al-Sallami fat-free mass."""
        sex = str(sex).lower()
        if sex == "male":
            return float(
                (0.88 + (1.0 - 0.88) / (1.0 + (age / 13.4) ** (-12.7)))
                * (9720.0 * weight)
                / (6680.0 + 216.0 * bmi)
            )
        if sex == "female":
            return float(
                (1.11 + (1.0 - 1.11) / (1.0 + (age / 7.1) ** (-1.1)))
                * (9720.0 * weight)
                / (8780.0 + 244.0 * bmi)
            )
        raise ValueError(f"Unknown sex: {sex}")

    def ksex(self) -> float:
        """Compute sex effect modifier."""
        sex = str(self.patient.sex).lower()
        if sex == "male":
            return 1.0
        if sex == "female":
            return float(
                1.0
                + self.Theta11
                * self.f_sigmoid(self.patient.age, 12.0, 6.0)
                * (1.0 - self.f_sigmoid(self.patient.age, 45.0, 6.0))
            )
        raise ValueError(f"Unknown sex: {self.patient.sex}")

    def c_obs(self, x: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
        """Apply a simple observation model to latent concentration."""
        return x + np.exp(self.epsilon)


class EleveldPD:
    """Remifentanil effect-site model.

    This class only provides the effect-site equilibration constant ``ke0``.
    It does not predict BIS or MAP.

    Typical model
    -------------
    ``ke0 = 1.09 * exp(-0.0289 * (age - 35))``

    BSV model
    ---------
    ``eta_ke0 ~ N(0, 0.947)``

    ``ke0_i = ke0_typical * exp(eta_ke0)``
    """

    def __init__(self, patient: Patient, use_bsv: bool = False) -> None:
        self.patient = patient
        self.patient_ref = EleveldPatient(
            age=35,
            height=170,
            weight=70,
            sex="male",
            opiates=False,
        )
        self.use_bsv = bool(use_bsv)
        self._set_params()

    @staticmethod
    def _f_ageing(x: float, age: float, age_ref: float) -> float:
        return float(np.exp(x * (age - age_ref)))

    def _set_params(self) -> None:
        self.Theta1 = 1.09       # ke0 at 35 years, min^-1
        self.Theta2 = -0.0289    # age effect on ke0
        self.omega2_ke0 = 0.947  # BSV variance on log(ke0)

        if self.use_bsv:
            self.draw_eta()
        else:
            self.reset_eta()

    def draw_eta(self) -> None:
        """Draw BSV eta for ke0 and update individual ke0."""
        self.eta_ke0 = float(
            np.random.normal(loc=0.0, scale=np.sqrt(self.omega2_ke0))
        )
        self._update_pd()

    def reset_eta(self) -> None:
        """Reset BSV eta to zero and update individual ke0."""
        self.eta_ke0 = 0.0
        self._update_pd()

    def _update_pd(self) -> None:
        """Update remifentanil ke0 and its effect-site half-life."""
        self.ke0_typical = (
            self.Theta1
            * self._f_ageing(self.Theta2, self.patient.age, self.patient_ref.age)
        )
        self.ke0 = float(self.ke0_typical * np.exp(self.eta_ke0))
        self.effect_site_half_life = float(np.log(2.0) / self.ke0)


class PKSolver:
    """Solver class for the 3-compartment remifentanil PK model."""

    def __init__(self, patient: Patient, pk: EleveldPK) -> None:
        self.patient = patient
        self.pk = pk

    @staticmethod
    def derivative(
        X: Sequence[float],
        t: float,
        k10: float,
        k12: float,
        k13: float,
        k21: float,
        k31: float,
        dosing: Optional[DosingProtocol] = None,
    ) -> NDArray[np.float64]:
        """Evaluate ODE derivatives for ``[A1, A2, A3]``.

        Amount states are in microgram. External dosing, if supplied, must
        provide ``dotA0(t)`` in microgram/min.
        """
        A1, A2, A3 = X
        input_rate = 0.0 if dosing is None else float(dosing.dotA0(t))

        dotA1 = input_rate - (k10 + k12 + k13) * A1 + k21 * A2 + k31 * A3
        dotA2 = k12 * A1 - k21 * A2
        dotA3 = k13 * A1 - k31 * A3

        return np.array([dotA1, dotA2, dotA3], dtype=float)

    def solve_ode(
        self,
        t: NDArray[np.float64],
        y0: Sequence[float],
        dosing: Optional[DosingProtocol] = None,
    ) -> NDArray[np.float64]:
        """Solve the PK ODE system.

        Parameters
        ----------
        t : numpy.ndarray
            Time grid in minutes.
        y0 : Sequence[float]
            Initial state ``[A1, A2, A3]`` in microgram.
        dosing : optional
            External dosing object with ``dotA0(t)`` in microgram/min.

        Returns
        -------
        numpy.ndarray
            State trajectories with shape ``(3, n_timepoints)``.
        """
        args = (
            self.pk.k10,
            self.pk.k12,
            self.pk.k13,
            self.pk.k21,
            self.pk.k31,
            dosing,
        )
        try:
            res = integrate.odeint(
                self.derivative,
                y0,
                t,
                args=args,
                tcrit=getattr(dosing, "tcrit", None),
            )
        except TypeError:
            res = integrate.odeint(self.derivative, y0, t, args=args)
        return res.T

    def central_concentration(self, states: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return central concentration, µg/L == ng/mL."""
        return np.asarray(states[0], dtype=float) / float(self.pk.V1)

    def print_model_parameters(self) -> None:
        """Log patient and PK parameter values."""
        logger.info("BW: %.3f kg", self.patient.weight)
        logger.info("BH: %.3f cm", self.patient.height)
        logger.info("BMI: %.3f", self.patient.bmi)
        logger.info("Age/sex: %s y, %s", self.patient.age, self.patient.sex)
        logger.info("V1 = %.6f L", self.pk.V1)
        logger.info("V2 = %.6f L", self.pk.V2)
        logger.info("V3 = %.6f L", self.pk.V3)
        logger.info("CL = %.6f L/min", self.pk.CL)
        logger.info("Q2 = %.6f L/min", self.pk.Q2)
        logger.info("Q3 = %.6f L/min", self.pk.Q3)
        logger.info("k10 = %.8f min^-1", self.pk.k10)
        logger.info("k12 = %.8f min^-1", self.pk.k12)
        logger.info("k13 = %.8f min^-1", self.pk.k13)
        logger.info("k21 = %.8f min^-1", self.pk.k21)
        logger.info("k31 = %.8f min^-1", self.pk.k31)

    def __call__(
        self,
        t: Optional[NDArray[np.float64]] = None,
        y0: Optional[Sequence[float]] = None,
        dosing: Optional[DosingProtocol] = None,
    ) -> NDArray[np.float64]:
        """Run the PK simulation.

        Defaults to a 1 µg initial bolus over a 15-min grid.
        """
        if t is None:
            t = np.linspace(0.0, 15.0, 15 * 60 + 1)
        if y0 is None:
            y0 = [1.0, 0.0, 0.0]
        return self.solve_ode(t=t, y0=y0, dosing=dosing)


class PKPDSolver:
    """Solver class for the 3-compartment remifentanil PK/effect-site model."""

    def __init__(self, patient: Patient, pk: EleveldPK, pd: EleveldPD) -> None:
        self.patient = patient
        self.pk = pk
        self.pd = pd

    @staticmethod
    def derivative(
        X: Sequence[float],
        t: float,
        V1: float,
        k10: float,
        k12: float,
        k13: float,
        k21: float,
        k31: float,
        ke0: float,
        dosing: Optional[DosingProtocol] = None,
    ) -> NDArray[np.float64]:
        """Evaluate ODE derivatives for ``[A1, A2, A3, Ce]``.

        Amount states are in microgram. ``Ce`` is in µg/L, numerically equal
        to ng/mL. External dosing, if supplied, must provide ``dotA0(t)`` in
        microgram/min.
        """
        A1, A2, A3, Ce = X
        input_rate = 0.0 if dosing is None else float(dosing.dotA0(t))

        dotA1 = input_rate - (k10 + k12 + k13) * A1 + k21 * A2 + k31 * A3
        dotA2 = k12 * A1 - k21 * A2
        dotA3 = k13 * A1 - k31 * A3

        cp = A1 / V1  # µg/L == ng/mL
        dotCe = ke0 * (cp - Ce)

        return np.array([dotA1, dotA2, dotA3, dotCe], dtype=float)

    def solve_ode(
        self,
        t: NDArray[np.float64],
        y0: Sequence[float],
        dosing: Optional[DosingProtocol] = None,
    ) -> NDArray[np.float64]:
        """Solve the coupled PK/effect-site ODE system.

        Parameters
        ----------
        t : numpy.ndarray
            Time grid in minutes.
        y0 : Sequence[float]
            Initial state ``[A1, A2, A3, Ce]``. Amount states are in µg;
            ``Ce`` is in µg/L == ng/mL.
        dosing : optional
            External dosing object with ``dotA0(t)`` in µg/min.

        Returns
        -------
        numpy.ndarray
            State trajectories with shape ``(4, n_timepoints)``.
        """
        args = (
            self.pk.V1,
            self.pk.k10,
            self.pk.k12,
            self.pk.k13,
            self.pk.k21,
            self.pk.k31,
            self.pd.ke0,
            dosing,
        )
        try:
            res = integrate.odeint(
                self.derivative,
                y0,
                t,
                args=args,
                tcrit=getattr(dosing, "tcrit", None),
            )
        except TypeError:
            res = integrate.odeint(self.derivative, y0, t, args=args)
        return res.T

    def central_concentration(self, states: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return remifentanil central concentration, µg/L == ng/mL."""
        return np.asarray(states[0], dtype=float) / float(self.pk.V1)

    @staticmethod
    def effect_site_concentration(states: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return remifentanil effect-site concentration, µg/L == ng/mL."""
        return np.asarray(states[3], dtype=float)

    def print_model_parameters(self) -> None:
        """Log patient and PK/PD parameter values."""
        logger.info("BW: %.3f kg", self.patient.weight)
        logger.info("BH: %.3f cm", self.patient.height)
        logger.info("BMI: %.3f", self.patient.bmi)
        logger.info("Age/sex: %s y, %s", self.patient.age, self.patient.sex)
        logger.info("V1 = %.6f L", self.pk.V1)
        logger.info("V2 = %.6f L", self.pk.V2)
        logger.info("V3 = %.6f L", self.pk.V3)
        logger.info("CL = %.6f L/min", self.pk.CL)
        logger.info("Q2 = %.6f L/min", self.pk.Q2)
        logger.info("Q3 = %.6f L/min", self.pk.Q3)
        logger.info("ke0 typical = %.8f min^-1", self.pd.ke0_typical)
        logger.info("ke0 individual = %.8f min^-1", self.pd.ke0)
        logger.info("effect-site half-life = %.6f min", self.pd.effect_site_half_life)

    def __call__(
        self,
        t: Optional[NDArray[np.float64]] = None,
        y0: Optional[Sequence[float]] = None,
        dosing: Optional[DosingProtocol] = None,
    ) -> NDArray[np.float64]:
        """Run the coupled PK/effect-site simulation.

        Defaults to a 1 µg initial bolus over a 15-min grid. For infusion
        schedules, pass a dosing object and set ``y0=[0, 0, 0, 0]``.
        """
        if t is None:
            t = np.linspace(0.0, 15.0, 15 * 60 + 1)
        if y0 is None:
            y0 = [1.0, 0.0, 0.0, 0.0]
        return self.solve_ode(t=t, y0=y0, dosing=dosing)
