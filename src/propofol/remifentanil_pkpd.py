import logging
from typing import Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import integrate

from propofol.patient import EleveldPatient
from propofol.protocols import Patient

logger = logging.getLogger(__name__)


class EleveldPK:
    """Eleveld-style remifentanil PK model.

    Notes
    -----
    - This is a structural rewrite of your remifentanil code to match the
      organization of the propofol PK/PD module.
    - The parameter values are kept from your original remifentanil code.
    - This class is PK-only.
    """

    def __init__(self, patient: Patient, use_bsv: bool = False) -> None:
        """Initialize PK model parameters for a patient.

        Parameters
        ----------
        patient : Patient
            Patient descriptor used to parameterize the PK model.
        use_bsv : bool, default=False
            If ``True``, draw between-subject variability terms at initialization.
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

        self._set_params()

    def _f_ageing(self, x: float, age: float, age_ref: float) -> float:
        return np.exp(x * (age - age_ref))

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
        self.V1 = (
            self.Theta1
            * self.SIZE()
            * self._f_ageing(self.Theta8, self.patient.age, self.patient_ref.age)
            * np.exp(self.eta1)
        )  # L

        self.V2 = (
            self.Theta2
            * self.SIZE()
            * self._f_ageing(self.Theta9, self.patient.age, self.patient_ref.age)
            * self.KSEX()
            * np.exp(self.eta2)
        )  # L

        self.V3 = (
            self.Theta3
            * self.SIZE()
            * self._f_ageing(self.Theta10, self.patient.age, self.patient_ref.age)
            * np.exp(self.Theta12 * (self.patient.weight - self.patient_ref.weight))
            * np.exp(self.eta3)
        )  # L

        self.CL = (
            self.Theta4
            * self.SIZE() ** 0.75
            * (self.KMAT(self.patient.weight) / self.KMAT(self.patient_ref.weight))
            * self.KSEX()
            * self._f_ageing(self.Theta9, self.patient.age, self.patient_ref.age)
            * np.exp(self.eta4)
        )  # L/min

        self.Q2 = (
            self.Theta5
            * np.float_power(self.V2 / self.Theta2, 0.75)
            * self._f_ageing(self.Theta8, self.patient.age, self.patient_ref.age)
            * self.KSEX()
            * np.exp(self.eta5)
        )  # L/min

        self.Q3 = (
            self.Theta6
            * np.float_power(self.V3 / self.Theta3, 0.75)
            * self._f_ageing(self.Theta8, self.patient.age, self.patient_ref.age)
            * np.exp(self.eta6)
        )  # L/min

        self.k10 = self.CL / self.V1
        self.k12 = self.Q2 / self.V1
        self.k13 = self.Q3 / self.V1
        self.k21 = self.Q2 / self.V2
        self.k31 = self.Q3 / self.V3

    def draw_eta(self) -> None:
        """Draw independent eta values and recompute PK parameters."""
        omega2 = np.array([
            0.104,
            0.115,
            0.810,
            0.0197,
            0.0547,
            0.285,
        ])

        n_samples = 1
        eta = np.random.multivariate_normal(
            np.zeros_like(omega2),
            np.diag(omega2),
            size=n_samples,
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
        """Reset all PK random effects to zero and recompute parameters."""
        self.eta1 = 0.0
        self.eta2 = 0.0
        self.eta3 = 0.0
        self.eta4 = 0.0
        self.eta5 = 0.0
        self.eta6 = 0.0

        self._update_pk()

    def draw_epsilon(self) -> None:
        """Draw residual error for concentration observations."""
        self.epsilon = np.random.normal(loc=0.0, scale=1.0)

    def reset_epsilon(self) -> None:
        """Reset residual error term to zero."""
        self.epsilon = 0.0

    def f_sigmoid(self, x: float, e50: float, lmbda: float) -> float:
        """Compute sigmoidal scaling function."""
        return x**lmbda / (x**lmbda + e50**lmbda)

    def SIZE(self) -> float:
        """Compute body-size scaling using Al-Sallami fat-free mass."""
        return (
            self.f_al_sallami(self.patient.age, self.patient.weight, self.patient.bmi)
            / self.f_al_sallami(
                self.patient_ref.age,
                self.patient_ref.weight,
                self.patient_ref.bmi,
            )
        )

    def KMAT(self, weight: float) -> float:
        """Compute maturation factor for clearance."""
        return self.f_sigmoid(weight, self.Theta7, 2.0)

    def f_al_sallami(self, age: float, weight: float, bmi: float) -> float:
        """Compute Al-Sallami fat-free mass."""
        if self.patient.sex == "male":
            return (
                (0.88 + (1.0 - 0.88) / (1.0 + (age / 13.4) ** (-12.7)))
                * (9720.0 * weight)
                / (6680.0 + 216.0 * bmi)
            )
        elif self.patient.sex == "female":
            return (
                (1.11 + (1.0 - 1.11) / (1.0 + (age / 7.1) ** (-1.1)))
                * (9720.0 * weight)
                / (8780.0 + 244.0 * bmi)
            )
        raise ValueError(f"Unknown sex: {self.patient.sex}")

    def KSEX(self) -> float:
        """Compute sex effect modifier."""
        if self.patient.sex == "male":
            return 1.0
        elif self.patient.sex == "female":
            return (
                1.0
                + self.Theta11
                * self.f_sigmoid(self.patient.age, 12.0, 6.0)
                * (1.0 - self.f_sigmoid(self.patient.age, 45.0, 6.0))
            )
        raise ValueError(f"Unknown sex: {self.patient.sex}")

    def c_obs(self, x: float) -> float:
        """Apply observation model to latent concentration."""
        return x + np.exp(self.epsilon)


class PKSolver:
    """Solver class for the 3-compartment remifentanil PK model."""

    def __init__(self, patient: Patient, pk: EleveldPK) -> None:
        """Initialize a coupled PK solver.

        Parameters
        ----------
        patient : Patient
            Patient descriptor.
        pk : EleveldPK
            Pharmacokinetic model instance.
        """
        self.patient = patient
        self.pk = pk

    def derivative(
        self,
        X: Sequence[float],
        t: float,
        k10: float,
        k12: float,
        k13: float,
        k21: float,
        k31: float,
    ) -> NDArray[np.float64]:
        """Evaluate ODE derivatives for the 3-compartment PK system.

        Parameters
        ----------
        X : Sequence[float]
            State vector ``[A1, A2, A3]``.
        t : float
            Time point in minutes.
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

        Returns
        -------
        numpy.ndarray
            Time derivatives ``[dA1/dt, dA2/dt, dA3/dt]``.
        """
        A1, A2, A3 = X

        dotA1 = -(k10 + k12 + k13) * A1 + k21 * A2 + k31 * A3
        dotA2 = k12 * A1 - k21 * A2
        dotA3 = k13 * A1 - k31 * A3

        return np.array([dotA1, dotA2, dotA3], dtype=float)

    def solve_ode(
        self,
        t: NDArray[np.float64],
        y0: Sequence[float],
    ) -> NDArray[np.float64]:
        """Solve the PK ODE system.

        Parameters
        ----------
        t : numpy.ndarray
            Time grid in minutes.
        y0 : Sequence[float]
            Initial state ``[A1, A2, A3]``.

        Returns
        -------
        numpy.ndarray
            State trajectories with shape ``(3, n_timepoints)``.
        """
        res = integrate.odeint(
            self.derivative,
            y0,
            t,
            args=(
                self.pk.k10,
                self.pk.k12,
                self.pk.k13,
                self.pk.k21,
                self.pk.k31,
            ),
        )
        return res.T

    def print_model_parameters(self) -> None:
        """Log patient and PK parameter values."""
        logger.info(f"BW:{self.patient.weight}kg")
        logger.info(f"BH:{self.patient.height}cm")
        logger.info(f"BMI:{self.patient.bmi:.2f}")
        logger.info(f"{self.patient.age}y, {self.patient.sex}")
        logger.info(f"vc = {self.pk.V1:.3f} L")
        logger.info(f"v2 = {self.pk.V2:.3f} L")
        logger.info(f"v3 = {self.pk.V3:.3f} L")
        logger.info(f"k10 = {self.pk.k10:.8f}")
        logger.info(f"k12 = {self.pk.k12:.8f}")
        logger.info(f"k13 = {self.pk.k13:.8f}")
        logger.info(f"k21 = {self.pk.k21:.8f}")
        logger.info(f"k31 = {self.pk.k31:.8f}")

    def __call__(
        self,
        t: Optional[NDArray[np.float64]] = None,
        y0: Optional[Sequence[float]] = None,
    ) -> NDArray[np.float64]:
        """Run the PK simulation using optional custom initial conditions.

        Parameters
        ----------
        t : numpy.ndarray, optional
            Time grid in minutes. If ``None``, uses ``np.linspace(0, 15, 15*60+1)``.
        y0 : Sequence[float], optional
            Initial state ``[A1, A2, A3]``. If ``None``, uses ``[1, 0, 0]``.

        Returns
        -------
        numpy.ndarray
            Simulated state trajectories with shape ``(3, n_timepoints)``.
        """
        if t is None:
            t = np.linspace(0, 15, 15 * 60 + 1)
        if y0 is None:
            y0 = [1.0, 0.0, 0.0]
        return self.solve_ode(t, y0)