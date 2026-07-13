from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from propofol.protocols import Patient


class ResponseSurfaceLaryngoscopy:
    """Synergistic propofol-remifentanil response-surface model for tolerance to laryngoscopy.

    The model predicts the probability of tolerance to laryngoscopy as a percentage.

    The normalized drug effects are:

        u_prop = Ce_prop / EC50_prop
        u_remi = Ce_remi / EC50_remi

    The synergistic interaction model is:

        U = u_prop + u_remi + alpha * u_prop * u_remi

    The probability of tolerance is then:

        P_tolerance = 100 * U**gamma / (1 + U**gamma)

    where alpha > 0 represents synergy.

    Units
    -----
    Ce_prop : microgram/mL
        Propofol effect-site concentration.

    Ce_remi : ng/mL
        Remifentanil effect-site concentration.
        If your remifentanil PK/PD solver returns Ce in microgram/L, this can
        be passed directly because microgram/L = ng/mL.

    Notes
    -----
    This implementation assumes a Greco-style normalized interaction term:
        alpha * u_prop * u_remi

    If the source model defines the interaction term differently, only the
    line that computes U in compute_probability() needs to be changed.
    """

    def __init__(
        self,
        patient: Optional[Patient] = None,
        use_bsv: bool = False,
    ) -> None:
        self.patient = patient
        self.use_bsv = bool(use_bsv)

        self._set_params()

    def _set_params(self) -> None:
        """Set typical parameters for the laryngoscopy-tolerance response surface."""
        self.P_min = 0.0
        self.P_max = 100.0

        self.Theta_EC50_prop = 5.63    # microgram/mL
        self.Theta_EC50_remi = 19.0    # ng/mL
        self.Theta_gamma = 7.94
        self.Theta_interaction = 2.13  # synergistic interaction, alpha

        if self.use_bsv:
            self.draw_eta()
        else:
            self.reset_eta()

    @staticmethod
    def cv_to_omega2(cv: float) -> float:
        """Convert coefficient of variation to log-normal eta variance.

        Parameters
        ----------
        cv : float
            Coefficient of variation as a fraction, not a percentage.
            Example: 37% should be passed as 0.37.

        Returns
        -------
        float
            Log-normal eta variance.
        """
        cv = float(cv)
        if cv < 0:
            raise ValueError("CV must be non-negative.")
        return float(np.log1p(cv**2))

    def draw_eta(self) -> None:
        """Draw between-subject variability for response-surface parameters.

        Currently provided CV:
        - EC50_prop: 37%

        No CV was provided for EC50_remi, gamma, or the interaction term, so
        their eta values are fixed at zero by default. If you later find CVs
        for these parameters, add them here.
        """
        cv_ec50_prop = 0.37

        omega2_ec50_prop = self.cv_to_omega2(cv_ec50_prop)

        self.eta_ec50_prop = np.random.normal(
            loc=0.0,
            scale=np.sqrt(omega2_ec50_prop),
        )

        self.eta_ec50_remi = 0.0
        self.eta_gamma = 0.0
        self.eta_interaction = 0.0

        self._update_pd()

    def reset_eta(self) -> None:
        """Reset eta values to zero."""
        self.eta_ec50_prop = 0.0
        self.eta_ec50_remi = 0.0
        self.eta_gamma = 0.0
        self.eta_interaction = 0.0

        self._update_pd()

    def _update_pd(self) -> None:
        """Update individual response-surface parameters."""
        self.EC50_prop = self.Theta_EC50_prop * np.exp(self.eta_ec50_prop)
        self.EC50_remi = self.Theta_EC50_remi * np.exp(self.eta_ec50_remi)
        self.gamma = self.Theta_gamma * np.exp(self.eta_gamma)
        self.interaction = self.Theta_interaction * np.exp(self.eta_interaction)

    def compute_probability(
        self,
        Ce_prop: ArrayLike,
        Ce_remi: ArrayLike,
    ) -> NDArray[np.float64]:
        """Compute probability of tolerance to laryngoscopy.

        Parameters
        ----------
        Ce_prop : array-like
            Propofol effect-site concentration in microgram/mL.

        Ce_remi : array-like
            Remifentanil effect-site concentration in ng/mL.

        Returns
        -------
        numpy.ndarray
            Predicted probability of tolerance to laryngoscopy, expressed as
            a percentage from 0 to 100.
        """
        Ce_prop = np.asarray(Ce_prop, dtype=float)
        Ce_remi = np.asarray(Ce_remi, dtype=float)

        if np.any(~np.isfinite(Ce_prop)) or np.any(~np.isfinite(Ce_remi)):
            raise ValueError("Effect-site concentrations must be finite.")

        if np.any(Ce_prop < 0) or np.any(Ce_remi < 0):
            raise ValueError("Effect-site concentrations must be non-negative.")

        u_prop = Ce_prop / self.EC50_prop
        u_remi = Ce_remi / self.EC50_remi

        # Synergistic normalized interaction.
        # alpha > 0 increases the combined effect beyond additivity.
        U = u_prop + u_remi + self.interaction * u_prop * u_remi

        U = np.clip(U, 0.0, None)

        U_gamma = np.power(U, self.gamma)
        effect_fraction = U_gamma / (1.0 + U_gamma)

        probability = self.P_min + (self.P_max - self.P_min) * effect_fraction

        return np.asarray(np.clip(probability, self.P_min, self.P_max), dtype=np.float64)

    def compute_failure_probability(
        self,
        Ce_prop: ArrayLike,
        Ce_remi: ArrayLike,
    ) -> NDArray[np.float64]:
        """Compute probability of non-tolerance to laryngoscopy.

        Returns
        -------
        numpy.ndarray
            Predicted probability of not tolerating laryngoscopy, expressed as
            a percentage from 0 to 100.
        """
        return 100.0 - self.compute_probability(Ce_prop=Ce_prop, Ce_remi=Ce_remi)

    def __call__(
        self,
        Ce_prop: ArrayLike,
        Ce_remi: ArrayLike,
    ) -> NDArray[np.float64]:
        """Alias for compute_probability()."""
        return self.compute_probability(Ce_prop=Ce_prop, Ce_remi=Ce_remi)