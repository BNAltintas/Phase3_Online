from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from propofol.protocols import Patient

class ResponseSurfaceBIS:
    """Additive propofol-remifentanil response-surface model for BIS.

    The model is:

        U = (Ce_prop / EC50_prop) + (Ce_remi / EC50_remi)

        BIS = BIS0 - (BIS0 - BIS_min) * U**gamma / (1 + U**gamma)

    Units
    -----
    Ce_prop : µg/mL
        Propofol effect-site concentration.

    Ce_remi : ng/mL
        Remifentanil effect-site concentration.
        If your remifentanil PK/PD solver returns Ce in µg/L, this can be
        passed directly because µg/L = ng/mL.
    """

    def __init__(
        self,
        patient: Optional[Patient] = None,
        use_bsv: bool = False,
    ) -> None:
        self.patient = patient
        self.use_bsv = use_bsv

        self._set_params()

    def _set_params(self) -> None:
        """Set typical parameters for the additive response-surface model."""
        self.BIS0 = 97.4
        self.BIS_min = 0.0

        self.Theta_EC50_prop = 4.47   # µg/mL
        self.Theta_EC50_remi = 19.3   # ng/mL
        self.Theta_gamma = 1.43

        if self.use_bsv:
            self.draw_eta()
        else:
            self.reset_eta()

    @staticmethod
    def cv_to_omega2(cv: float) -> float:
        """Convert coefficient of variation to log-normal eta variance."""
        if cv < 0:
            raise ValueError("CV must be non-negative.")
        return float(np.log1p(cv**2))

    def draw_eta(self) -> None:
        """Draw between-subject variability for response-surface parameters.

        CVs:
        - EC50_prop: 18.2%
        - EC50_remi: 88.8%
        - gamma: 30.4%
        """
        cv_ec50_prop = 0.182
        cv_ec50_remi = 0.888
        cv_gamma = 0.304

        omega2_ec50_prop = self.cv_to_omega2(cv_ec50_prop)
        omega2_ec50_remi = self.cv_to_omega2(cv_ec50_remi)
        omega2_gamma = self.cv_to_omega2(cv_gamma)

        self.eta_ec50_prop = np.random.normal(
            loc=0.0,
            scale=np.sqrt(omega2_ec50_prop),
        )
        self.eta_ec50_remi = np.random.normal(
            loc=0.0,
            scale=np.sqrt(omega2_ec50_remi),
        )
        self.eta_gamma = np.random.normal(
            loc=0.0,
            scale=np.sqrt(omega2_gamma),
        )

        self._update_pd()

    def reset_eta(self) -> None:
        """Reset eta values to zero."""
        self.eta_ec50_prop = 0.0
        self.eta_ec50_remi = 0.0
        self.eta_gamma = 0.0

        self._update_pd()

    def _update_pd(self) -> None:
        """Update individual response-surface parameters."""
        self.EC50_prop = self.Theta_EC50_prop * np.exp(self.eta_ec50_prop)
        self.EC50_remi = self.Theta_EC50_remi * np.exp(self.eta_ec50_remi)
        self.gamma = self.Theta_gamma * np.exp(self.eta_gamma)

    def compute_bis(
        self,
        Ce_prop: ArrayLike,
        Ce_remi: ArrayLike,
    ) -> NDArray[np.float64]:
        """Compute BIS from propofol and remifentanil effect-site concentrations.

        Parameters
        ----------
        Ce_prop : array-like
            Propofol effect-site concentration in µg/mL.

        Ce_remi : array-like
            Remifentanil effect-site concentration in ng/mL.

        Returns
        -------
        numpy.ndarray
            Predicted BIS.
        """
        Ce_prop = np.asarray(Ce_prop, dtype=float)
        Ce_remi = np.asarray(Ce_remi, dtype=float)

        if np.any(Ce_prop < 0) or np.any(Ce_remi < 0):
            raise ValueError("Effect-site concentrations must be non-negative.")

        U = (Ce_prop / self.EC50_prop) + (Ce_remi / self.EC50_remi)

        effect_fraction = U**self.gamma / (1.0 + U**self.gamma)

        BIS = self.BIS0 - (self.BIS0 - self.BIS_min) * effect_fraction

        return BIS

    def __call__(
        self,
        Ce_prop: ArrayLike,
        Ce_remi: ArrayLike,
    ) -> NDArray[np.float64]:
        """Alias for compute_bis()."""
        return self.compute_bis(Ce_prop, Ce_remi)
