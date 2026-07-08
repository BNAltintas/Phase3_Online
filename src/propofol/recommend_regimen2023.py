from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.optimize import Bounds, minimize

from propofol.config import (
    BOLUS_MGKG_BOUNDS,
    INFUSION_MGKGH_BOUNDS,
)
from propofol.haemo_pd_su2023 import SuHaemoPD
from propofol.patient import EleveldPatient as Patient
from propofol.propofol_pkpd import EleveldPD
from propofol.propofol_pkpd import EleveldPK as PropofolPK
from propofol.remifentanil_pkpd import EleveldPK as RemifentanilPK

# ============================================================
# Fixed recommendation settings
# ============================================================

SIM_MINUTES = 15
N_INTERVALS = SIM_MINUTES
TIME = np.linspace(0.0, float(SIM_MINUTES), SIM_MINUTES * 60 + 1)

# Target is assessed only after induction onset. This represents the requested
# "after 2-3 minutes" target window.
TARGET_ASSESSMENT_START_MIN = 2.5
TARGET_BIS_LOW = 40.0
TARGET_BIS_HIGH = 60.0
TARGET_BIS_MID = 50.0
MAP_ABS_MIN_TARGET = 65.0
MAP_REL_FRAC_TARGET = 0.70

CONFIDENCE_N_SIMULATIONS = 100
CONFIDENCE_LOW_PERCENTILE = 5.0
CONFIDENCE_HIGH_PERCENTILE = 95.0

# Fast deterministic optimization settings.
# The app can still pass mode="lazy"/"auto"/"accurate". These now control only
# the number of Powell starts, never more than 5.
OPTIMIZATION_MODE = "auto"
OPTIMIZATION_BASE_SEED = 42
POWELL_MAXITER_PROPOFOL_ONLY = 65
POWELL_MAXITER_WITH_REMI = 85
POWELL_MAXFEV_PROPOFOL_ONLY = 260
POWELL_MAXFEV_WITH_REMI = 390

# At most two active maintenance segments are allowed. A zero-rate pause at the
# start is allowed before segment 1.
MAX_MAINTENANCE_RATE_CHANGES = 2
MAX_INITIAL_PAUSE_MIN = 3.0
MIN_SWITCH_MIN = TARGET_ASSESSMENT_START_MIN
MAX_SWITCH_MIN = float(SIM_MINUTES)

# Automatic clinical rounding.
PROPOFOL_BOLUS_STEP_MG = 5.0
PUMP_RATE_STEP_ML_H = 1.0

# Default concentrations. The app can override these.
DEFAULT_PROPOFOL_CONC_MG_ML = 10.0
DEFAULT_REMI_CONC_MCG_ML = 50.0

# Remifentanil is maintenance-only in this version.
REMI_INFUSION_MCGKGMIN_BOUNDS = (0.0, 0.50)

PROPOFOL_BOLUS_DURATION_MIN = 1.0 / 60.0

# Su2023 extraction convention. Your validation script used out[-1] as MAP.
# If your local Su2023 solve output differs, change these constants only here.
SU2023_MAP_OUTPUT_INDEX = -1
SU2023_REMI_A1_OUTPUT_INDEX = 4


# ============================================================
# Generic helpers
# ============================================================

def map_lower_bound_from_baseline(
    baseline_map: float,
    map_abs_min_target: float = MAP_ABS_MIN_TARGET,
    map_rel_frac_target: float = MAP_REL_FRAC_TARGET,
) -> float:
    """MAP lower target: highest of the absolute floor or a fraction of baseline MAP."""
    return float(max(float(map_abs_min_target), float(map_rel_frac_target) * float(baseline_map)))


def assessment_mask(t: Sequence[float]) -> np.ndarray:
    """Return target-assessment mask after the 2-3 min post-induction onset period."""
    t = np.asarray(t, dtype=float)
    return t >= TARGET_ASSESSMENT_START_MIN


def round_to_step(x: float, step: float) -> float:
    """Round a value to the nearest multiple of a given step."""
    if step <= 0:
        raise ValueError("step must be positive")
    return float(step * np.round(float(x) / step))


def round_array_to_step(x: Sequence[float], step: float) -> np.ndarray:
    """Round an array of values to the nearest multiple of a given step."""
    arr = np.asarray(x, dtype=float)
    if step <= 0:
        raise ValueError("step must be positive")
    return step * np.round(arr / step)


def mode_to_n_starts(mode: str) -> int:
    """Map UI mode to a small number of Powell starts."""
    mode = str(mode or OPTIMIZATION_MODE).lower()
    if mode == "lazy":
        return 3
    if mode == "auto":
        return 4
    if mode == "accurate":
        return 5
    raise ValueError("mode must be one of: 'lazy', 'auto', 'accurate'")


def count_rate_changes(rates: Optional[Sequence[float]], tol: float = 1e-8) -> int:
    """Count the number of rate changes in a sequence of rates."""
    if rates is None:
        return 0
    rates = np.asarray(rates, dtype=float)
    if len(rates) <= 1:
        return 0
    return int(np.sum(np.abs(np.diff(rates)) > tol))


def set_patient_opiates(patient: Patient, opiates: bool) -> Patient:
    """Copy the patient and set the opiates flag before constructing PK/PD objects."""
    patient_copy = deepcopy(patient)
    patient_copy.opiates = bool(opiates)
    return patient_copy


def validate_concentrations(
    propofol_conc_mg_ml: float,
    remifentanil_conc_mcg_ml: float,
) -> tuple[float, float]:
    """Validate and return propofol and remifentanil concentrations."""
    prop = float(propofol_conc_mg_ml)
    remi = float(remifentanil_conc_mcg_ml)

    if not np.isfinite(prop) or prop <= 0:
        raise ValueError("propofol_conc_mg_ml must be positive.")
    if not np.isfinite(remi) or remi <= 0:
        raise ValueError("remifentanil_conc_mcg_ml must be positive.")

    return prop, remi


# ============================================================
# Unit conversions
# ============================================================

def prop_mgkgh_to_ml_h(
    rate_mgkg_h: Sequence[float],
    weight_kg: float,
    conc_mg_ml: float,
) -> np.ndarray:
    """Convert propofol infusion rates from mg/kg/h to mL/h."""
    rate_mgkg_h = np.asarray(rate_mgkg_h, dtype=float)
    return rate_mgkg_h * float(weight_kg) / float(conc_mg_ml)


def prop_ml_h_to_mgkgh(
    rate_ml_h: Sequence[float],
    weight_kg: float,
    conc_mg_ml: float,
) -> np.ndarray:
    """Convert propofol infusion rates from mL/h to mg/kg/h."""
    rate_ml_h = np.asarray(rate_ml_h, dtype=float)
    return rate_ml_h * float(conc_mg_ml) / float(weight_kg)


def prop_mgkgh_to_mcgkgmin(rate_mgkg_h: Sequence[float]) -> np.ndarray:
    """Convert propofol infusion rates from mg/kg/h to mcg/kg/min."""
    rate_mgkg_h = np.asarray(rate_mgkg_h, dtype=float)
    return rate_mgkg_h * 1000.0 / 60.0


def remi_mcgkgmin_to_ml_h(
    rate_mcgkg_min: Sequence[float],
    weight_kg: float,
    conc_mcg_ml: float,
) -> np.ndarray:
    """Convert remifentanil infusion rates from mcg/kg/min to mL/h."""
    rate_mcgkg_min = np.asarray(rate_mcgkg_min, dtype=float)
    return rate_mcgkg_min * float(weight_kg) * 60.0 / float(conc_mcg_ml)


def remi_ml_h_to_mcgkgmin(
    rate_ml_h: Sequence[float],
    weight_kg: float,
    conc_mcg_ml: float,
) -> np.ndarray:
    """Convert remifentanil infusion rates from mL/h to mcg/kg/min."""
    rate_ml_h = np.asarray(rate_ml_h, dtype=float)
    return rate_ml_h * float(conc_mcg_ml) / (float(weight_kg) * 60.0)


def remi_mcgkgmin_to_ngkgmin(rate_mcgkg_min: Sequence[float]) -> np.ndarray:
    """Convert remifentanil infusion rates from mcg/kg/min to ng/kg/min."""
    rate_mcgkg_min = np.asarray(rate_mcgkg_min, dtype=float)
    return rate_mcgkg_min * 1000.0


# ============================================================
# Compact two-segment schedule helper
# ============================================================

def make_pause_two_segment_schedule(
    pause_min: float,
    switch_min: float,
    rate_1: float,
    rate_2: float,
    n_minutes: int = N_INTERVALS,
) -> np.ndarray:
    """
    Make a minute-wise schedule with:
        1. optional initial pause, rate = 0
        2. active segment 1, rate_1
        3. active segment 2, rate_2

    This permits at most two active maintenance segments while allowing a pause
    immediately after induction. Minute intervals are assigned using their
    midpoint, so a pause of 1.5 min pauses approximately the first two minutes.
    """
    pause_min = float(np.clip(pause_min, 0.0, float(n_minutes)))
    switch_min = float(np.clip(switch_min, 0.0, float(n_minutes)))
    switch_min = max(switch_min, pause_min + 1.0)
    switch_min = min(switch_min, float(n_minutes))

    rate_1 = max(float(rate_1), 0.0)
    rate_2 = max(float(rate_2), 0.0)

    midpoints = np.arange(n_minutes, dtype=float) + 0.5
    rates = np.zeros(n_minutes, dtype=float)
    rates[(midpoints >= pause_min) & (midpoints < switch_min)] = rate_1
    rates[midpoints >= switch_min] = rate_2
    return rates


# ============================================================
# Dosing object
# ============================================================

class PiecewiseWeightScaledDosing:
    """
    Supplies a drug input rate dotA0(t) in amount/min.

    Propofol amount unit: mg.
    Remifentanil amount unit: microgram.
    """

    def __init__(
        self,
        bolus_amount: float,
        infusion_rates: Sequence[float],
        weight_kg: float,
        bolus_duration_min: float,
        rate_unit: str,
    ) -> None:
        self.bolus_amount = max(float(bolus_amount), 0.0)
        self.infusion_rates = np.clip(np.asarray(infusion_rates, dtype=float), 0.0, None)
        self.weight_kg = float(weight_kg)
        self.bolus_duration_min = float(bolus_duration_min)
        self.rate_unit = str(rate_unit)

        if self.weight_kg <= 0:
            raise ValueError("weight_kg must be positive")
        if self.bolus_duration_min <= 0:
            raise ValueError("bolus_duration_min must be positive")
        if self.rate_unit not in {"mgkg_h", "mcgkg_min"}:
            raise ValueError("rate_unit must be 'mgkg_h' or 'mcgkg_min'")

        critical_times = [0.0]
        if self.bolus_amount > 0:
            critical_times.append(self.bolus_duration_min)
        critical_times += [float(i) for i in range(1, len(self.infusion_rates) + 1)]
        self.tcrit = sorted(set(critical_times))

    def _maintenance_amount_per_min(self, rate: float) -> float:
        """
        Convert a maintenance infusion rate to the amount per minute.

        Args:
            rate: The infusion rate in the specified rate_unit.

        Returns:
            The amount of drug administered per minute.
        """
        if self.rate_unit == "mgkg_h":
            return float(rate) * self.weight_kg / 60.0
        if self.rate_unit == "mcgkg_min":
            return float(rate) * self.weight_kg
        raise RuntimeError("Unsupported rate_unit")

    def dotA0(self, t: float) -> float:
        """
        Compute the drug input rate at a given time.

        Args:
            t: Time in minutes.

        Returns:
            The drug input rate in amount/min.
        """
        t = float(t)

        if self.bolus_amount > 0 and 0.0 <= t < self.bolus_duration_min:
            return self.bolus_amount / self.bolus_duration_min

        if len(self.infusion_rates) == 0:
            return 0.0

        idx = int(np.floor(t))
        idx = max(0, min(idx, len(self.infusion_rates) - 1))
        return self._maintenance_amount_per_min(float(self.infusion_rates[idx]))

    def __call__(self, t: float) -> float:
        """
        Compute the drug input rate at a given time.

        Args:
            t: Time in minutes.

        Returns:
            The drug input rate in amount/min.
        """
        return self.dotA0(t)


# ============================================================
# Result containers
# ============================================================

@dataclass
class DecodedRegimen:
    """
    Decoded regimen containing detailed information about the drug administration.

    Attributes:
        propofol_bolus_mg: The propofol bolus amount in mg.
        propofol_bolus_mgkg: The propofol bolus amount in mg/kg.
        propofol_rates_mgkgh: The propofol infusion rates in mg/kg/h.
        propofol_rates_ml_h: The propofol infusion rates in ml/h.
        propofol_rates_mcgkgmin: The propofol infusion rates in mcg/kg/min.
    """
    propofol_bolus_mg: float
    propofol_bolus_mgkg: float
    propofol_rates_mgkgh: np.ndarray
    propofol_rates_ml_h: np.ndarray
    propofol_rates_mcgkgmin: np.ndarray

    remifentanil_selected: bool
    remifentanil_bolus_mcg: float = 0.0
    remifentanil_bolus_mcgkg: float = 0.0
    remifentanil_rates_mcgkgmin: Optional[np.ndarray] = None
    remifentanil_rates_ml_h: Optional[np.ndarray] = None
    remifentanil_rates_ngkgmin: Optional[np.ndarray] = None


@dataclass
class ConfidenceResult:
    """
    Confidence result containing detailed information about the simulation outcomes.

    Attributes:
        n_simulations: The total number of simulations run.
        n_completed: The number of simulations that completed successfully.
        n_failed: The number of simulations that failed.
        n_target_met: The number of simulations that met the target criteria.
        confidence_percent: The confidence percentage based on the simulation outcomes.
    """
    n_simulations: int
    n_completed: int
    n_failed: int
    n_target_met: int
    confidence_percent: float

    time_min: np.ndarray

    bis_p05: np.ndarray
    bis_p50: np.ndarray
    bis_p95: np.ndarray

    map_p05: np.ndarray
    map_p50: np.ndarray
    map_p95: np.ndarray

    cp_propofol_p05: np.ndarray
    cp_propofol_p50: np.ndarray
    cp_propofol_p95: np.ndarray

    ce_propofol_p05: np.ndarray
    ce_propofol_p50: np.ndarray
    ce_propofol_p95: np.ndarray

    cp_remifentanil_p05: Optional[np.ndarray] = None
    cp_remifentanil_p50: Optional[np.ndarray] = None
    cp_remifentanil_p95: Optional[np.ndarray] = None


@dataclass
class RecommendationResult:
    """
    Recommendation result containing detailed information about the recommended regimen.

    Attributes:
        propofol_bolus_mg: The propofol bolus amount in mg.
        propofol_bolus_mgkg: The propofol bolus amount in mg/kg.
        propofol_inf_rates_mgkgh: The propofol infusion rates in mg/kg/h.
        propofol_inf_rates_ml_h: The propofol infusion rates in ml/h.
        propofol_inf_rates_mcgkgmin: The propofol infusion rates in mcg/kg/min.
        remifentanil_selected: Whether remifentanil is selected.
        remifentanil_bolus_mcg: The remifentanil bolus amount in mcg.
        remifentanil_bolus_mcgkg: The remifentanil bolus amount in mcg/kg.
    """
    propofol_bolus_mg: float
    propofol_bolus_mgkg: float
    propofol_inf_rates_mgkgh: np.ndarray
    propofol_inf_rates_ml_h: np.ndarray
    propofol_inf_rates_mcgkgmin: np.ndarray

    remifentanil_selected: bool
    remifentanil_bolus_mcg: float
    remifentanil_bolus_mcgkg: float
    remifentanil_inf_rates_mcgkgmin: Optional[np.ndarray]
    remifentanil_inf_rates_ml_h: Optional[np.ndarray]
    remifentanil_inf_rates_ngkgmin: Optional[np.ndarray]

    propofol_conc_mg_ml: float
    remifentanil_conc_mcg_ml: float

    time_min: np.ndarray
    cp_propofol: np.ndarray
    ce_propofol: np.ndarray
    cp_remifentanil: Optional[np.ndarray]
    bis: np.ndarray
    map_mmhg: np.ndarray

    map_lower_bound_mmhg: float
    feasible_bis: bool
    feasible_map: bool
    feasible: bool

    # Actual target values used for this specific run (may differ from the
    # module defaults if the user customized them in the UI). Downstream
    # display code should read these rather than the module constants, so
    # plots/summaries always match what was actually optimized against.
    target_bis_low: float
    target_bis_high: float
    map_abs_min_target_mmhg: float
    map_rel_frac_target: float

    confidence: ConfidenceResult
    confidence_percent: float

    objective_value: float
    mode: str
    opiates_for_bis_pd: bool
    n_propofol_rate_changes: int
    n_remifentanil_rate_changes: int
    max_rate_changes_allowed: int
    n_deterministic_optimization_runs: int

    # True when confidence was intentionally not computed (the fixed-bolus
    # manual-override path). Downstream display code should key off this
    # flag - not off confidence_percent being NaN - to decide whether to
    # show a confidence UI at all.
    confidence_skipped: bool = False


# ============================================================
# Model solve/extraction helpers
# ============================================================

def solve_su2023_model(
    model: SuHaemoPD,
    t_grid: np.ndarray,
    dosing_prop: PiecewiseWeightScaledDosing,
    dosing_remi: Optional[PiecewiseWeightScaledDosing],
):
    """Call Su2023 solve_ode while tolerating minor local signature differences."""
    try:
        return model.solve_ode(
            t=t_grid,
            y0=None,
            dosing_prop=dosing_prop,
            dosing_remi=dosing_remi,
        )
    except TypeError:
        try:
            return model.solve_ode(t_grid, None, dosing_prop, dosing_remi)
        except TypeError:
            return model.solve_ode(t_grid, dosing_prop, dosing_remi)


def extract_su2023_output(
    out,
    map_output_index: int = SU2023_MAP_OUTPUT_INDEX,
    remi_a1_output_index: int = SU2023_REMI_A1_OUTPUT_INDEX,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Extract propofol a1, ce, MAP, and remifentanil a1 from Su2023 solve_ode output."""
    if len(out) < 4:
        raise ValueError("Su2023 solve_ode output has fewer than 4 elements.")

    a1_prop = np.asarray(out[0], dtype=float)
    ce_prop = np.asarray(out[3], dtype=float)
    map_model = np.asarray(out[map_output_index], dtype=float)

    a1_remi = None
    if 0 <= remi_a1_output_index < len(out):
        candidate = np.asarray(out[remi_a1_output_index], dtype=float)
        if candidate.shape == a1_prop.shape:
            a1_remi = candidate

    return a1_prop, ce_prop, map_model, a1_remi


def trajectory_target_flags(
    t: Sequence[float],
    bis: Sequence[float],
    map_mmhg: Sequence[float],
    map_lower_bound: float,
    bis_low: float = TARGET_BIS_LOW,
    bis_high: float = TARGET_BIS_HIGH,
) -> tuple[bool, bool, bool]:
    """
    Target definition after the post-induction onset period:
        BIS trajectory must stay between bis_low and bis_high.
        MAP trajectory must stay >= map_lower_bound.
    """
    t = np.asarray(t, dtype=float)
    bis = np.asarray(bis, dtype=float)
    map_mmhg = np.asarray(map_mmhg, dtype=float)
    mask = assessment_mask(t)

    if not np.any(mask):
        return False, False, False

    bis_eval = bis[mask]
    map_eval = map_mmhg[mask]

    if len(bis_eval) == 0 or len(map_eval) == 0:
        return False, False, False

    if np.any(~np.isfinite(bis_eval)) or np.any(~np.isfinite(map_eval)):
        return False, False, False

    feasible_bis = bool(np.all((bis_eval >= bis_low) & (bis_eval <= bis_high)))
    feasible_map = bool(np.all(map_eval >= float(map_lower_bound)))
    feasible = bool(feasible_bis and feasible_map)

    return feasible_bis, feasible_map, feasible


def percentile_band(values: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the 5th, 50th, and 95th percentiles of a list of arrays."""
    if len(values) == 0:
        nan = np.full_like(TIME, np.nan, dtype=float)
        return nan, nan, nan

    arr = np.vstack([np.asarray(v, dtype=float) for v in values])
    return (
        np.nanpercentile(arr, CONFIDENCE_LOW_PERCENTILE, axis=0),
        np.nanpercentile(arr, 50.0, axis=0),
        np.nanpercentile(arr, CONFIDENCE_HIGH_PERCENTILE, axis=0),
    )


def _skipped_confidence_result(time_min: np.ndarray) -> ConfidenceResult:
    """
    Build a placeholder ConfidenceResult for when confidence is intentionally
    not computed (the fixed-bolus manual-override path). All bands are NaN
    so any code that plots them draws nothing rather than a misleading band.
    """
    t = np.asarray(time_min, dtype=float)
    nan_arr = np.full_like(t, np.nan, dtype=float)
    return ConfidenceResult(
        n_simulations=0,
        n_completed=0,
        n_failed=0,
        n_target_met=0,
        confidence_percent=float("nan"),
        time_min=t.copy(),
        bis_p05=nan_arr.copy(),
        bis_p50=nan_arr.copy(),
        bis_p95=nan_arr.copy(),
        map_p05=nan_arr.copy(),
        map_p50=nan_arr.copy(),
        map_p95=nan_arr.copy(),
        cp_propofol_p05=nan_arr.copy(),
        cp_propofol_p50=nan_arr.copy(),
        cp_propofol_p95=nan_arr.copy(),
        ce_propofol_p05=nan_arr.copy(),
        ce_propofol_p50=nan_arr.copy(),
        ce_propofol_p95=nan_arr.copy(),
        cp_remifentanil_p05=nan_arr.copy(),
        cp_remifentanil_p50=nan_arr.copy(),
        cp_remifentanil_p95=nan_arr.copy(),
    )


# ============================================================
# Core recommender
# ============================================================

class Su2023PropofolRemifentanilRecommender:
    """Optimize a 15-min Su2023 propofol-only or propofol-remifentanil regimen."""

    def __init__(
        self,
        patient: Patient,
        use_remifentanil: bool = False,
        use_bsv: bool = False,
        mode: str = OPTIMIZATION_MODE,
        propofol_conc_mg_ml: float = DEFAULT_PROPOFOL_CONC_MG_ML,
        remifentanil_conc_mcg_ml: float = DEFAULT_REMI_CONC_MCG_ML,
        target_bis_low: float = TARGET_BIS_LOW,
        target_bis_high: float = TARGET_BIS_HIGH,
        map_abs_min_target: float = MAP_ABS_MIN_TARGET,
        map_rel_frac_target: float = MAP_REL_FRAC_TARGET,
        fixed_bolus_mgkg: Optional[float] = None,
        fixed_remi_rate_mcgkgmin: Optional[float] = None,
        map_output_index: int = SU2023_MAP_OUTPUT_INDEX,
        remi_a1_output_index: int = SU2023_REMI_A1_OUTPUT_INDEX,
        seed: int = OPTIMIZATION_BASE_SEED,
    ) -> None:
        # When set, the propofol bolus is pinned to this value (mg/kg) and
        # only the maintenance schedule is optimized - see build_bounds().
        # None (the default) preserves the original, fully free-bolus search
        # used by every existing call site.
        self.fixed_bolus_mgkg = (
            float(fixed_bolus_mgkg) if fixed_bolus_mgkg is not None else None
        )

        # When set (and use_remifentanil is True), the remifentanil
        # maintenance rate is pinned to this single constant value (mcg/kg/
        # min, no pause, one rate for the full 15 minutes) and only the
        # propofol bolus + propofol maintenance are optimized - see
        # build_bounds(). None (the default) preserves the original, fully
        # free remifentanil-schedule search used by every existing call site.
        self.fixed_remi_rate_mcgkgmin = (
            float(fixed_remi_rate_mcgkgmin) if fixed_remi_rate_mcgkgmin is not None else None
        )

        self.use_remifentanil = bool(use_remifentanil)
        self.use_bsv = bool(use_bsv)
        self.mode = str(mode or OPTIMIZATION_MODE).lower()
        self.n_starts = mode_to_n_starts(self.mode)
        self.propofol_conc_mg_ml, self.remifentanil_conc_mcg_ml = validate_concentrations(
            propofol_conc_mg_ml,
            remifentanil_conc_mcg_ml,
        )

        self.target_bis_low = float(target_bis_low)
        self.target_bis_high = float(target_bis_high)
        if self.target_bis_low >= self.target_bis_high:
            raise ValueError("target_bis_low must be lower than target_bis_high.")
        self.target_bis_mid = (self.target_bis_low + self.target_bis_high) / 2.0
        self.map_abs_min_target = float(map_abs_min_target)
        self.map_rel_frac_target = float(map_rel_frac_target)

        self.map_output_index = int(map_output_index)
        self.remi_a1_output_index = int(remi_a1_output_index)
        self.seed = int(seed)
        self._objective_cache: dict[tuple[float, ...], float] = {}

        self.opiates_for_bis_pd = bool(self.use_remifentanil)
        self.patient = set_patient_opiates(patient, self.opiates_for_bis_pd)

        self.weight_kg = float(self.patient.weight)
        self.baseline_map = float(self.patient.base_map)
        self.map_lower_bound = map_lower_bound_from_baseline(
            self.baseline_map, self.map_abs_min_target, self.map_rel_frac_target,
        )

        self.pk_prop = PropofolPK(patient=self.patient, use_bsv=self.use_bsv)
        self.pd_prop = EleveldPD(patient=self.patient, use_bsv=self.use_bsv)
        self.pk_remi = RemifentanilPK(patient=self.patient, use_bsv=self.use_bsv)
        self.haemo = self._instantiate_haemo()

    def _instantiate_haemo(self) -> SuHaemoPD:
        """
        Instantiate the SuHaemoPD model with the appropriate parameters.
        """
        try:
            return SuHaemoPD(
                patient=self.patient,
                pk_propofol=self.pk_prop,
                pd_propofol=self.pd_prop,
                pk_remifentanil=self.pk_remi,
                use_bsv=self.use_bsv,
            )
        except TypeError:
            try:
                return SuHaemoPD(
                    patient=self.patient,
                    pk_propofol=self.pk_prop,
                    pd_propofol=self.pd_prop,
                    pk_remifentanil=self.pk_remi,
                )
            except TypeError:
                return SuHaemoPD(
                    patient=self.patient,
                    pk_propofol=self.pk_prop,
                    pd_propofol=self.pd_prop,
                )

    @property
    def n_parameters(self) -> int:
        """Return the number of optimization parameters for the active model."""
        # Propofol: bolus, pause, switch, rate1, rate2.
        # Remifentanil, if selected: pause, switch, rate1, rate2.
        return 5 + (4 if self.use_remifentanil else 0)

    def build_bounds(self) -> list[tuple[float, float]]:
        """
        Build the bounds for the optimization parameters.

        When self.fixed_bolus_mgkg is set, the bolus bound collapses to a
        single point. When self.fixed_remi_rate_mcgkgmin is set, the
        remifentanil pause bound collapses to 0 (no pause - a single
        constant rate from t=0) and both remifentanil rate bounds collapse
        to that same point (so rate_1 == rate_2, a flat rate for the whole
        window regardless of the free "switch" time). clip_x_to_bounds()
        (used by every downstream consumer of the parameter vector:
        _split_x, initial_vectors, the Powell search itself) then pins
        those parameters to their fixed values on every iteration, so only
        the remaining free parameters are actually searched.
        """
        bolus_bounds = (
            (self.fixed_bolus_mgkg, self.fixed_bolus_mgkg)
            if self.fixed_bolus_mgkg is not None
            else BOLUS_MGKG_BOUNDS
        )

        bounds: list[tuple[float, float]] = [
            bolus_bounds,
            (0.0, MAX_INITIAL_PAUSE_MIN),
            (MIN_SWITCH_MIN, MAX_SWITCH_MIN),
            INFUSION_MGKGH_BOUNDS,
            INFUSION_MGKGH_BOUNDS,
        ]

        if self.use_remifentanil:
            if self.fixed_remi_rate_mcgkgmin is not None:
                remi_rate_bounds = (self.fixed_remi_rate_mcgkgmin, self.fixed_remi_rate_mcgkgmin)
                bounds += [
                    (0.0, 0.0),
                    (MIN_SWITCH_MIN, MAX_SWITCH_MIN),
                    remi_rate_bounds,
                    remi_rate_bounds,
                ]
            else:
                bounds += [
                    (0.0, MAX_INITIAL_PAUSE_MIN),
                    (MIN_SWITCH_MIN, MAX_SWITCH_MIN),
                    REMI_INFUSION_MCGKGMIN_BOUNDS,
                    REMI_INFUSION_MCGKGMIN_BOUNDS,
                ]

        return bounds

    def build_scipy_bounds(self) -> Bounds:
        """
        Build a scipy.optimize.Bounds object for the optimization parameters.
        """
        bounds = self.build_bounds()
        lower = np.asarray([b[0] for b in bounds], dtype=float)
        upper = np.asarray([b[1] for b in bounds], dtype=float)
        return Bounds(lower, upper)

    def clip_x_to_bounds(self, x: Sequence[float]) -> np.ndarray:
        """
        Clip the optimization parameters to their respective bounds.
        """
        x = np.asarray(x, dtype=float)
        bounds = self.build_bounds()
        lower = np.asarray([b[0] for b in bounds], dtype=float)
        upper = np.asarray([b[1] for b in bounds], dtype=float)
        return np.clip(x, lower, upper)

    def _split_x(
        self,
        x: Sequence[float],
    ) -> tuple[float, float, float, float, float, Optional[tuple[float, float, float, float]]]:
        """
        Split the optimization parameter vector into individual components.
        """
        x = self.clip_x_to_bounds(x)

        prop_bolus_mgkg = float(x[0])
        prop_pause_min = float(x[1])
        prop_switch_min = float(x[2])
        prop_rate_1 = float(x[3])
        prop_rate_2 = float(x[4])

        remi_params = None
        if self.use_remifentanil:
            remi_params = (
                float(x[5]),
                float(x[6]),
                float(x[7]),
                float(x[8]),
            )

        return (
            prop_bolus_mgkg,
            prop_pause_min,
            prop_switch_min,
            prop_rate_1,
            prop_rate_2,
            remi_params,
        )

    def decode_regimen(self, x: Sequence[float], apply_final_rounding: bool) -> DecodedRegimen:
        """
        Decode the optimization parameter vector into a human-readable regimen.
        """
        (
            prop_bolus_mgkg,
            prop_pause_min,
            prop_switch_min,
            prop_rate_1,
            prop_rate_2,
            remi_params,
        ) = self._split_x(x)

        # Propofol bolus: final clinical dose in PROPOFOL_BOLUS_STEP_MG steps.
        # Exception: a fixed (manually entered) bolus is never re-rounded -
        # it is already an exact clinician-entered value, and re-rounding it
        # would silently change the number the user typed.
        prop_bolus_mg_raw = max(prop_bolus_mgkg, 0.0) * self.weight_kg
        prop_bolus_mg = (
            round_to_step(prop_bolus_mg_raw, PROPOFOL_BOLUS_STEP_MG)
            if apply_final_rounding and self.fixed_bolus_mgkg is None
            else prop_bolus_mg_raw
        )
        prop_bolus_mg = max(prop_bolus_mg, 0.0)

        # Propofol maintenance: at most two active segments after optional pause.
        prop_rates_mgkgh_raw = make_pause_two_segment_schedule(
            pause_min=prop_pause_min,
            switch_min=prop_switch_min,
            rate_1=prop_rate_1,
            rate_2=prop_rate_2,
        )
        prop_rates_ml_h_raw = prop_mgkgh_to_ml_h(
            prop_rates_mgkgh_raw,
            self.weight_kg,
            self.propofol_conc_mg_ml,
        )
        prop_rates_ml_h = (
            round_array_to_step(prop_rates_ml_h_raw, PUMP_RATE_STEP_ML_H)
            if apply_final_rounding
            else prop_rates_ml_h_raw
        )
        prop_rates_ml_h = np.clip(prop_rates_ml_h, 0.0, None)
        prop_rates_mgkgh = prop_ml_h_to_mgkgh(
            prop_rates_ml_h,
            self.weight_kg,
            self.propofol_conc_mg_ml,
        )
        prop_rates_mcgkgmin = prop_mgkgh_to_mcgkgmin(prop_rates_mgkgh)

        if not self.use_remifentanil:
            return DecodedRegimen(
                propofol_bolus_mg=prop_bolus_mg,
                propofol_bolus_mgkg=prop_bolus_mg / self.weight_kg,
                propofol_rates_mgkgh=prop_rates_mgkgh,
                propofol_rates_ml_h=prop_rates_ml_h,
                propofol_rates_mcgkgmin=prop_rates_mcgkgmin,
                remifentanil_selected=False,
            )

        if remi_params is None:
            raise RuntimeError("Remifentanil selected but maintenance parameters are missing.")

        remi_pause_min, remi_switch_min, remi_rate_1, remi_rate_2 = remi_params

        # Remifentanil maintenance only. There is deliberately no bolus.
        remi_rates_mcgkgmin_raw = make_pause_two_segment_schedule(
            pause_min=remi_pause_min,
            switch_min=remi_switch_min,
            rate_1=remi_rate_1,
            rate_2=remi_rate_2,
        )
        remi_rates_ml_h_raw = remi_mcgkgmin_to_ml_h(
            remi_rates_mcgkgmin_raw,
            self.weight_kg,
            self.remifentanil_conc_mcg_ml,
        )
        remi_rates_ml_h = (
            round_array_to_step(remi_rates_ml_h_raw, PUMP_RATE_STEP_ML_H)
            if apply_final_rounding
            else remi_rates_ml_h_raw
        )
        remi_rates_ml_h = np.clip(remi_rates_ml_h, 0.0, None)
        remi_rates_mcgkgmin = remi_ml_h_to_mcgkgmin(
            remi_rates_ml_h,
            self.weight_kg,
            self.remifentanil_conc_mcg_ml,
        )
        remi_rates_ngkgmin = remi_mcgkgmin_to_ngkgmin(remi_rates_mcgkgmin)

        return DecodedRegimen(
            propofol_bolus_mg=prop_bolus_mg,
            propofol_bolus_mgkg=prop_bolus_mg / self.weight_kg,
            propofol_rates_mgkgh=prop_rates_mgkgh,
            propofol_rates_ml_h=prop_rates_ml_h,
            propofol_rates_mcgkgmin=prop_rates_mcgkgmin,
            remifentanil_selected=True,
            remifentanil_bolus_mcg=0.0,
            remifentanil_bolus_mcgkg=0.0,
            remifentanil_rates_mcgkgmin=remi_rates_mcgkgmin,
            remifentanil_rates_ml_h=remi_rates_ml_h,
            remifentanil_rates_ngkgmin=remi_rates_ngkgmin,
        )

    def simulate_regimen(
        self,
        regimen: DecodedRegimen,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]:
        """
        Simulate the PK/PD and MAP response for a given regimen.
        """
        dosing_prop = PiecewiseWeightScaledDosing(
            bolus_amount=regimen.propofol_bolus_mg,
            infusion_rates=regimen.propofol_rates_mgkgh,
            weight_kg=self.weight_kg,
            bolus_duration_min=PROPOFOL_BOLUS_DURATION_MIN,
            rate_unit="mgkg_h",
        )

        dosing_remi = None
        if regimen.remifentanil_selected:
            dosing_remi = PiecewiseWeightScaledDosing(
                bolus_amount=0.0,
                infusion_rates=(
                    regimen.remifentanil_rates_mcgkgmin
                    if regimen.remifentanil_rates_mcgkgmin is not None
                    else np.zeros(N_INTERVALS)
                ),
                weight_kg=self.weight_kg,
                bolus_duration_min=PROPOFOL_BOLUS_DURATION_MIN,
                rate_unit="mcgkg_min",
            )

        out = solve_su2023_model(
            model=self.haemo,
            t_grid=TIME,
            dosing_prop=dosing_prop,
            dosing_remi=dosing_remi,
        )

        a1_prop, ce_prop, map_model, a1_remi = extract_su2023_output(
            out,
            map_output_index=self.map_output_index,
            remi_a1_output_index=self.remi_a1_output_index,
        )

        cp_prop = np.asarray(a1_prop, dtype=float) / float(self.pk_prop.V1)
        ce_prop = np.asarray(ce_prop, dtype=float)
        bis = np.asarray(self.pd_prop.bis(ce_prop), dtype=float)

        map_model = np.asarray(map_model, dtype=float)
        if len(map_model) == 0 or not np.isfinite(map_model[0]) or map_model[0] <= 0:
            raise ValueError("Initial Su2023 MAP model output is non-positive or invalid.")
        map_mmhg = self.baseline_map * (map_model / map_model[0])

        cp_remi = None
        if regimen.remifentanil_selected:
            if a1_remi is not None:
                # A1_remi is in microgram and V1 is in L; microgram/L = ng/mL.
                cp_remi = np.asarray(a1_remi, dtype=float) / float(self.pk_remi.V1)
            else:
                cp_remi = np.full_like(TIME, np.nan, dtype=float)

        return TIME, cp_prop, ce_prop, cp_remi, bis, map_mmhg

    def simulate_x(
        self,
        x: Sequence[float],
        apply_final_rounding: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray,
               DecodedRegimen]:
        """
        Simulate the PK/PD and MAP response for a given optimization parameter vector.
        """
        regimen = self.decode_regimen(x, apply_final_rounding=apply_final_rounding)
        t, cp_prop, ce_prop, cp_remi, bis, map_mmhg = self.simulate_regimen(regimen)
        return t, cp_prop, ce_prop, cp_remi, bis, map_mmhg, regimen

    def trajectory_penalty(  # noqa: C901
        self,
        t: np.ndarray,
        bis: np.ndarray,
        map_mmhg: np.ndarray,
        regimen: DecodedRegimen,
    ) -> float:
        """
        Hierarchical objective for the deterministic optimizer.

        Priority order after TARGET_ASSESSMENT_START_MIN (bounds are
        self.target_bis_low/high and self.map_lower_bound, which default to
        the module constants but may be user-configured per run):
            1. Avoid BIS above the upper target at any time point. This is
               the highest-priority violation and receives a near-hard
               penalty.
            2. Avoid BIS below the lower target at any time point. This is
               still strongly penalized, but less strongly than #1.
            3. Avoid MAP below the MAP lower bound.
            4. Among target-satisfying regimens, prefer BIS around the
               target midpoint, lower drug exposure, and simpler pump
               schedules.

        This hierarchy means MAP, dose, and smoothness penalties cannot dominate
        a regimen that leaves the patient too light after the onset period.
        """
        mask = assessment_mask(t)
        if not np.any(mask):
            return 1e12

        bis_eval = np.asarray(bis, dtype=float)[mask]
        map_eval = np.asarray(map_mmhg, dtype=float)[mask]

        if (
            len(bis_eval) == 0
            or len(map_eval) == 0
            or np.any(~np.isfinite(bis_eval))
            or np.any(~np.isfinite(map_eval))
        ):
            return 1e12

        # ----------------------------------------------------
        # 1) Highest priority: BIS > 60 after 2.5 minutes
        # ----------------------------------------------------
        # Any point above 60 means insufficient hypnotic depth. This must be
        # more important than MAP, dose, smoothness, or BIS < 40.
        bis_over = np.clip(bis_eval - self.target_bis_high, 0.0, None)

        if np.any(bis_over > 0.0):
            max_over = float(np.max(bis_over))
            mean_over = float(np.mean(bis_over))
            sumsq_over = float(np.sum(bis_over ** 2))
            n_over = float(np.sum(bis_over > 0.0))
            frac_over = n_over / float(len(bis_eval))

            # Small early-onset term, so the optimizer also learns to reach
            # the upper BIS target before the assessment window begins.
            early_mask = (t >= 1.0) & (t < TARGET_ASSESSMENT_START_MIN)
            early_penalty = 0.0
            if np.any(early_mask):
                early_bis = np.asarray(bis, dtype=float)[early_mask]
                if np.any(np.isfinite(early_bis)):
                    early_min = float(np.nanmin(early_bis))
                    early_penalty = 50_000.0 * max(early_min - self.target_bis_high, 0.0) ** 2

            return float(
                1e11
                + 1_000_000_000.0 * max_over ** 2
                + 250_000_000.0 * mean_over ** 2
                + 25_000_000.0 * sumsq_over
                + 100_000_000.0 * frac_over ** 2
                + early_penalty
            )

        # ----------------------------------------------------
        # 2) Second priority: BIS < 40 after 2.5 minutes
        # ----------------------------------------------------
        # Oversedation is important, but the user explicitly wants BIS > 60 to
        # dominate. Therefore the base penalty is two orders of magnitude lower
        # than the BIS > 60 branch.
        bis_under = np.clip(self.target_bis_low - bis_eval, 0.0, None)

        if np.any(bis_under > 0.0):
            max_under = float(np.max(bis_under))
            mean_under = float(np.mean(bis_under))
            sumsq_under = float(np.sum(bis_under ** 2))
            n_under = float(np.sum(bis_under > 0.0))
            frac_under = n_under / float(len(bis_eval))

            return float(
                1e9
                + 75_000_000.0 * max_under ** 2
                + 20_000_000.0 * mean_under ** 2
                + 2_500_000.0 * sumsq_under
                + 10_000_000.0 * frac_under ** 2
            )

        # ----------------------------------------------------
        # 3) Third priority: MAP target after BIS is fully within target
        # ----------------------------------------------------
        map_violation = np.clip(self.map_lower_bound - map_eval, 0.0, None)

        if np.any(map_violation > 0.0):
            max_map_violation = float(np.max(map_violation))
            mean_map_violation = float(np.mean(map_violation))
            sumsq_map_violation = float(np.sum(map_violation ** 2))
            n_map_violation = float(np.sum(map_violation > 0.0))
            frac_map_violation = n_map_violation / float(len(map_eval))

            # This is intentionally lower than BIS<40 and much lower than BIS>60.
            return float(
                1e7
                + 1_000_000.0 * max_map_violation ** 2
                + 250_000.0 * mean_map_violation ** 2
                + 50_000.0 * sumsq_map_violation
                + 1_000_000.0 * frac_map_violation ** 2
            )

        # ----------------------------------------------------
        # 4) Tie-breakers once BIS and MAP are both feasible
        # ----------------------------------------------------
        # Prefer BIS centered around the midpoint of the target range, but keep
        # this much weaker than any actual target violation.
        bis_target_dev = bis_eval - self.target_bis_mid
        bis_centering_penalty = 2_500.0 * float(np.mean(bis_target_dev ** 2))

        # Encourage reaching the upper BIS target before the assessment
        # window, but only as a weak tie-breaker once post-onset targets are
        # satisfied.
        adequate_idx = np.where(np.asarray(bis, dtype=float) <= self.target_bis_high)[0]
        if len(adequate_idx) == 0:
            onset_penalty = 50_000.0
        else:
            time_to_bis60 = float(t[adequate_idx[0]])
            onset_penalty = 500.0 * max(time_to_bis60 - TARGET_ASSESSMENT_START_MIN, 0.0) ** 2

        # Exposure and complexity penalties remain lowest priority. They only
        # decide between regimens that already meet BIS and MAP targets.
        prop_rates_mgkgh = regimen.propofol_rates_mgkgh
        prop_total_mg = regimen.propofol_bolus_mg + np.sum(prop_rates_mgkgh * self.weight_kg / 60.0)
        prop_penalty = (
            1.8 * np.sum(np.diff(prop_rates_mgkgh) ** 2)
            + 0.45 * np.sum(prop_rates_mgkgh)
            + 0.06 * prop_total_mg
        )

        remi_penalty = 0.0
        if self.use_remifentanil:
            remi_rates = regimen.remifentanil_rates_mcgkgmin
            if remi_rates is None:
                return 1e12
            remi_total_mcg = np.sum(remi_rates * self.weight_kg)  # 1-min intervals
            remi_penalty = (
                120.0 * np.sum(np.diff(remi_rates) ** 2)
                + 30.0 * np.sum(remi_rates)
                + 0.022 * remi_total_mcg
            )

        return float(
            bis_centering_penalty
            + onset_penalty
            + prop_penalty
            + remi_penalty
        )

    def objective(self, x: np.ndarray) -> float:
        """
        Compute the objective function value for a given optimization parameter vector.
        """
        x = self.clip_x_to_bounds(x)
        # Round the cache key lightly. This avoids repeated ODE solves for nearly
        # identical Powell evaluations without changing the practical optimum.
        cache_key = tuple(np.round(x, 5))
        cached = self._objective_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            t, _, _, _, bis, map_mmhg, regimen = self.simulate_x(
                x,
                apply_final_rounding=False,
            )
            value = self.trajectory_penalty(t=t, bis=bis, map_mmhg=map_mmhg, regimen=regimen)
        except Exception:
            value = 1e12

        self._objective_cache[cache_key] = float(value)
        return float(value)

    def rounded_objective(self, x: np.ndarray) -> float:
        """
        Compute the objective function value for a given optimization parameter vector,
        after applying final rounding to the regimen.
        """
        try:
            t, _, _, _, bis, map_mmhg, regimen = self.simulate_x(
                x,
                apply_final_rounding=True,
            )
        except Exception:
            return 1e12

        return self.trajectory_penalty(t=t, bis=bis, map_mmhg=map_mmhg, regimen=regimen)

    def initial_vectors(self) -> list[np.ndarray]:
        """Clinical multi-start guesses for Powell. Always returns at most 5 starts."""
        starts = [
            # bolus, prop_pause, prop_switch, prop_rate1, prop_rate2
            [1.5, 0.0, 7.0, 6.0, 4.0],
            [2.0, 0.0, 6.0, 7.5, 4.5],
            [1.2, 1.0, 8.0, 5.0, 4.0],
            [1.8, 0.5, 10.0, 6.0, 6.0],
            [1.0, 0.0, 5.0, 8.5, 5.0],
        ]

        if self.use_remifentanil:
            remi_starts = [
                # remi_pause, remi_switch, remi_rate1, remi_rate2
                [0.0, 7.0, 0.10, 0.06],
                [0.0, 6.0, 0.15, 0.08],
                [1.0, 8.0, 0.08, 0.06],
                [0.5, 10.0, 0.12, 0.12],
                [0.0, 5.0, 0.20, 0.10],
            ]
            starts = [p + r for p, r in zip(starts, remi_starts, strict=False)]

        rng = np.random.default_rng(self.seed)
        out = []
        for start in starts[: self.n_starts]:
            x = np.asarray(start, dtype=float)
            # Tiny deterministic jitter helps Powell avoid identical local paths
            # while staying clinically interpretable.
            scale = np.asarray([b[1] - b[0] for b in self.build_bounds()], dtype=float)
            jitter = rng.normal(loc=0.0, scale=0.015, size=len(x)) * scale
            out.append(self.clip_x_to_bounds(x + jitter))

        return out

    def optimize_once(self, x0: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Run a single Powell optimization starting from x0 and return the rounded objective value
        and the corresponding parameter vector.
        """
        result = minimize(
            self.objective,
            x0=self.clip_x_to_bounds(x0),
            method="Powell",
            bounds=self.build_scipy_bounds(),
            options={
                "maxiter": (
                    POWELL_MAXITER_WITH_REMI
                    if self.use_remifentanil
                    else POWELL_MAXITER_PROPOFOL_ONLY
                ),
                "maxfev": (
                    POWELL_MAXFEV_WITH_REMI
                    if self.use_remifentanil
                    else POWELL_MAXFEV_PROPOFOL_ONLY
                ),
                "xtol": 0.035,
                "ftol": 0.035,
                "disp": False,
            },
        )

        x = self.clip_x_to_bounds(result.x)
        rounded_loss = self.rounded_objective(x)
        return float(rounded_loss), x

    def optimize(self, skip_confidence: bool = False) -> RecommendationResult:
        """
        Run 3-5 deterministic Powell starts and choose the candidate with the
        lowest rounded-regimen objective. Confidence is then estimated with 100
        BSV simulations of the chosen rounded regimen only once.

        skip_confidence=True bypasses the (expensive, 100-simulation)
        confidence Monte Carlo entirely and returns a placeholder
        ConfidenceResult (confidence_percent=NaN, all bands NaN). Used by the
        fixed-bolus manual-override path, where confidence is not meaningful
        for a hand-picked dose and recomputing it would be wasted work.
        """
        best_loss = np.inf
        best_x = None

        for x0 in self.initial_vectors():
            loss, x = self.optimize_once(x0=x0)
            if loss < best_loss:
                best_loss = loss
                best_x = x

        if best_x is None:
            raise RuntimeError("Optimization failed to produce a candidate regimen.")

        regimen = self.decode_regimen(best_x, apply_final_rounding=True)
        t, cp_prop, ce_prop, cp_remi, bis, map_mmhg = self.simulate_regimen(regimen)

        feasible_bis, feasible_map, feasible = trajectory_target_flags(
            t=t,
            bis=bis,
            map_mmhg=map_mmhg,
            map_lower_bound=self.map_lower_bound,
            bis_low=self.target_bis_low,
            bis_high=self.target_bis_high,
        )

        if skip_confidence:
            confidence = _skipped_confidence_result(t)
        else:
            confidence = simulate_confidence(
                patient=self.patient,
                regimen=regimen,
                use_remifentanil=self.use_remifentanil,
                propofol_conc_mg_ml=self.propofol_conc_mg_ml,
                remifentanil_conc_mcg_ml=self.remifentanil_conc_mcg_ml,
                target_bis_low=self.target_bis_low,
                target_bis_high=self.target_bis_high,
                map_abs_min_target=self.map_abs_min_target,
                map_rel_frac_target=self.map_rel_frac_target,
                map_output_index=self.map_output_index,
                remi_a1_output_index=self.remi_a1_output_index,
                base_seed=self.seed + 10_000,
            )

        return RecommendationResult(
            propofol_bolus_mg=regimen.propofol_bolus_mg,
            propofol_bolus_mgkg=regimen.propofol_bolus_mgkg,
            propofol_inf_rates_mgkgh=regimen.propofol_rates_mgkgh,
            propofol_inf_rates_ml_h=regimen.propofol_rates_ml_h,
            propofol_inf_rates_mcgkgmin=regimen.propofol_rates_mcgkgmin,
            remifentanil_selected=regimen.remifentanil_selected,
            remifentanil_bolus_mcg=regimen.remifentanil_bolus_mcg,
            remifentanil_bolus_mcgkg=regimen.remifentanil_bolus_mcgkg,
            remifentanil_inf_rates_mcgkgmin=regimen.remifentanil_rates_mcgkgmin,
            remifentanil_inf_rates_ml_h=regimen.remifentanil_rates_ml_h,
            remifentanil_inf_rates_ngkgmin=regimen.remifentanil_rates_ngkgmin,
            propofol_conc_mg_ml=self.propofol_conc_mg_ml,
            remifentanil_conc_mcg_ml=self.remifentanil_conc_mcg_ml,
            time_min=t,
            cp_propofol=cp_prop,
            ce_propofol=ce_prop,
            cp_remifentanil=cp_remi,
            bis=bis,
            map_mmhg=map_mmhg,
            map_lower_bound_mmhg=self.map_lower_bound,
            feasible_bis=feasible_bis,
            feasible_map=feasible_map,
            feasible=feasible,
            target_bis_low=self.target_bis_low,
            target_bis_high=self.target_bis_high,
            map_abs_min_target_mmhg=self.map_abs_min_target,
            map_rel_frac_target=self.map_rel_frac_target,
            confidence=confidence,
            confidence_percent=confidence.confidence_percent,
            objective_value=float(best_loss),
            mode=self.mode,
            opiates_for_bis_pd=self.opiates_for_bis_pd,
            n_propofol_rate_changes=count_rate_changes(regimen.propofol_rates_ml_h),
            n_remifentanil_rate_changes=count_rate_changes(regimen.remifentanil_rates_ml_h),
            max_rate_changes_allowed=MAX_MAINTENANCE_RATE_CHANGES,
            n_deterministic_optimization_runs=self.n_starts,
            confidence_skipped=skip_confidence,
        )


# ============================================================
# Confidence simulation
# ============================================================

def simulate_regimen_once(
    patient: Patient,
    regimen: DecodedRegimen,
    use_remifentanil: bool,
    use_bsv: bool,
    propofol_conc_mg_ml: float,
    remifentanil_conc_mcg_ml: float,
    map_output_index: int,
    remi_a1_output_index: int,
    target_bis_low: float = TARGET_BIS_LOW,
    target_bis_high: float = TARGET_BIS_HIGH,
    map_abs_min_target: float = MAP_ABS_MIN_TARGET,
    map_rel_frac_target: float = MAP_REL_FRAC_TARGET,
    seed: Optional[int] = None,
):
    """
    Simulate a single regimen with optional BSV and return the PK/PD and MAP outputs.
    """
    if seed is not None:
        np.random.seed(int(seed))

    runner = Su2023PropofolRemifentanilRecommender(
        patient=patient,
        use_remifentanil=use_remifentanil,
        use_bsv=use_bsv,
        mode=OPTIMIZATION_MODE,
        propofol_conc_mg_ml=propofol_conc_mg_ml,
        remifentanil_conc_mcg_ml=remifentanil_conc_mcg_ml,
        target_bis_low=target_bis_low,
        target_bis_high=target_bis_high,
        map_abs_min_target=map_abs_min_target,
        map_rel_frac_target=map_rel_frac_target,
        map_output_index=map_output_index,
        remi_a1_output_index=remi_a1_output_index,
        seed=seed or OPTIMIZATION_BASE_SEED,
    )

    return runner.simulate_regimen(regimen), runner.map_lower_bound


def simulate_confidence(
    patient: Patient,
    regimen: DecodedRegimen,
    use_remifentanil: bool,
    propofol_conc_mg_ml: float,
    remifentanil_conc_mcg_ml: float,
    target_bis_low: float = TARGET_BIS_LOW,
    target_bis_high: float = TARGET_BIS_HIGH,
    map_abs_min_target: float = MAP_ABS_MIN_TARGET,
    map_rel_frac_target: float = MAP_REL_FRAC_TARGET,
    map_output_index: int = SU2023_MAP_OUTPUT_INDEX,
    remi_a1_output_index: int = SU2023_REMI_A1_OUTPUT_INDEX,
    base_seed: int = 10_000,
) -> ConfidenceResult:
    """
    Simulate multiple regimens to estimate the confidence in meeting the target BIS and MAP values.
    """
    bis_values: list[np.ndarray] = []
    map_values: list[np.ndarray] = []
    cp_prop_values: list[np.ndarray] = []
    ce_prop_values: list[np.ndarray] = []
    cp_remi_values: list[np.ndarray] = []

    n_target_met = 0
    n_failed = 0

    for i in range(CONFIDENCE_N_SIMULATIONS):
        try:
            (t, cp_prop, ce_prop, cp_remi, bis, map_mmhg), map_lower_bound = simulate_regimen_once(
                patient=patient,
                regimen=regimen,
                use_remifentanil=use_remifentanil,
                use_bsv=True,
                propofol_conc_mg_ml=propofol_conc_mg_ml,
                remifentanil_conc_mcg_ml=remifentanil_conc_mcg_ml,
                target_bis_low=target_bis_low,
                target_bis_high=target_bis_high,
                map_abs_min_target=map_abs_min_target,
                map_rel_frac_target=map_rel_frac_target,
                map_output_index=map_output_index,
                remi_a1_output_index=remi_a1_output_index,
                seed=base_seed + i,
            )

            _, _, feasible = trajectory_target_flags(
                t=t,
                bis=bis,
                map_mmhg=map_mmhg,
                map_lower_bound=map_lower_bound,
                bis_low=target_bis_low,
                bis_high=target_bis_high,
            )

            n_target_met += int(feasible)
            bis_values.append(np.asarray(bis, dtype=float))
            map_values.append(np.asarray(map_mmhg, dtype=float))
            cp_prop_values.append(np.asarray(cp_prop, dtype=float))
            ce_prop_values.append(np.asarray(ce_prop, dtype=float))

            if use_remifentanil and cp_remi is not None:
                cp_remi_values.append(np.asarray(cp_remi, dtype=float))

        except Exception:
            n_failed += 1
            continue

    n_completed = len(bis_values)
    confidence_percent = 100.0 * n_target_met / n_completed if n_completed > 0 else np.nan

    bis_p05, bis_p50, bis_p95 = percentile_band(bis_values)
    map_p05, map_p50, map_p95 = percentile_band(map_values)
    cp_prop_p05, cp_prop_p50, cp_prop_p95 = percentile_band(cp_prop_values)
    ce_prop_p05, ce_prop_p50, ce_prop_p95 = percentile_band(ce_prop_values)

    cp_remi_p05 = cp_remi_p50 = cp_remi_p95 = None
    if use_remifentanil:
        cp_remi_p05, cp_remi_p50, cp_remi_p95 = percentile_band(cp_remi_values)

    return ConfidenceResult(
        n_simulations=CONFIDENCE_N_SIMULATIONS,
        n_completed=n_completed,
        n_failed=n_failed,
        n_target_met=n_target_met,
        confidence_percent=float(confidence_percent),
        time_min=TIME.copy(),
        bis_p05=bis_p05,
        bis_p50=bis_p50,
        bis_p95=bis_p95,
        map_p05=map_p05,
        map_p50=map_p50,
        map_p95=map_p95,
        cp_propofol_p05=cp_prop_p05,
        cp_propofol_p50=cp_prop_p50,
        cp_propofol_p95=cp_prop_p95,
        ce_propofol_p05=ce_prop_p05,
        ce_propofol_p50=ce_prop_p50,
        ce_propofol_p95=ce_prop_p95,
        cp_remifentanil_p05=cp_remi_p05,
        cp_remifentanil_p50=cp_remi_p50,
        cp_remifentanil_p95=cp_remi_p95,
    )


# ============================================================
# Public API
# ============================================================

def recommend_su2023_regimen(
    patient: Patient,
    opiate: str = "none",
    use_remifentanil: Optional[bool] = None,
    propofol_conc_mg_ml: float = DEFAULT_PROPOFOL_CONC_MG_ML,
    remifentanil_conc_mcg_ml: float = DEFAULT_REMI_CONC_MCG_ML,
    target_bis_low: float = TARGET_BIS_LOW,
    target_bis_high: float = TARGET_BIS_HIGH,
    map_abs_min_target: float = MAP_ABS_MIN_TARGET,
    map_rel_frac_target: float = MAP_REL_FRAC_TARGET,
    mode: str = OPTIMIZATION_MODE,
    seed: int = OPTIMIZATION_BASE_SEED,
) -> RecommendationResult:
    """
    Recommend a 15-min Su2023 regimen.

    Supported opiate options:
        - "none"
        - "remifentanil"

    Sufentanil and fentanyl are intentionally not implemented yet.

    target_bis_low/target_bis_high and map_abs_min_target/map_rel_frac_target
    default to the module constants (the system defaults shown in the UI) but
    can be overridden per call, e.g. from user-edited UI values.
    """
    opiate = str(opiate or "none").lower()

    if use_remifentanil is None:
        use_remifentanil = opiate == "remifentanil"

    if opiate in {"sufentanil", "fentanyl"}:
        raise NotImplementedError("Only remifentanil is currently supported.")

    if bool(use_remifentanil):
        opiate = "remifentanil"
    else:
        opiate = "none"

    recommender = Su2023PropofolRemifentanilRecommender(
        patient=patient,
        use_remifentanil=bool(use_remifentanil),
        use_bsv=False,
        mode=mode,
        propofol_conc_mg_ml=propofol_conc_mg_ml,
        remifentanil_conc_mcg_ml=remifentanil_conc_mcg_ml,
        target_bis_low=target_bis_low,
        target_bis_high=target_bis_high,
        map_abs_min_target=map_abs_min_target,
        map_rel_frac_target=map_rel_frac_target,
        seed=seed,
    )

    return recommender.optimize()


def recommend_maintenance_for_fixed_propofol_bolus(
    patient: Patient,
    fixed_bolus_mg: float,
    opiate: str = "none",
    use_remifentanil: Optional[bool] = None,
    propofol_conc_mg_ml: float = DEFAULT_PROPOFOL_CONC_MG_ML,
    remifentanil_conc_mcg_ml: float = DEFAULT_REMI_CONC_MCG_ML,
    target_bis_low: float = TARGET_BIS_LOW,
    target_bis_high: float = TARGET_BIS_HIGH,
    map_abs_min_target: float = MAP_ABS_MIN_TARGET,
    map_rel_frac_target: float = MAP_REL_FRAC_TARGET,
    mode: str = OPTIMIZATION_MODE,
    seed: int = OPTIMIZATION_BASE_SEED,
) -> RecommendationResult:
    """
    Re-optimize the maintenance schedule for a fixed, manually-entered
    propofol induction bolus (mg). The bolus is pinned exactly as given
    (never rounded); only the maintenance-rate parameters are searched.
    Confidence is not computed (see Su2023PropofolRemifentanilRecommender
    .optimize(skip_confidence=True)) - the returned result has
    confidence_skipped=True and confidence_percent=NaN.

    This is the manual-override entry point. The original, fully free-bolus
    recommendation path (recommend_su2023_regimen) is untouched by this
    function's existence - it is a separate, additive call.

    Supported opiate options mirror recommend_su2023_regimen: "none" or
    "remifentanil". If remifentanil is selected, its maintenance schedule is
    still optimized freely alongside the fixed propofol bolus - only the
    propofol bolus is fixed, never remifentanil.
    """
    opiate = str(opiate or "none").lower()

    if use_remifentanil is None:
        use_remifentanil = opiate == "remifentanil"

    if opiate in {"sufentanil", "fentanyl"}:
        raise NotImplementedError("Only remifentanil is currently supported.")

    if bool(use_remifentanil):
        opiate = "remifentanil"
    else:
        opiate = "none"

    weight_kg = float(patient.weight)
    if not np.isfinite(weight_kg) or weight_kg <= 0:
        raise ValueError("Patient weight must be positive.")

    fixed_bolus_mg = float(fixed_bolus_mg)
    if not np.isfinite(fixed_bolus_mg) or fixed_bolus_mg <= 0:
        raise ValueError("fixed_bolus_mg must be positive.")

    fixed_bolus_mgkg = fixed_bolus_mg / weight_kg

    recommender = Su2023PropofolRemifentanilRecommender(
        patient=patient,
        use_remifentanil=bool(use_remifentanil),
        use_bsv=False,
        mode=mode,
        propofol_conc_mg_ml=propofol_conc_mg_ml,
        remifentanil_conc_mcg_ml=remifentanil_conc_mcg_ml,
        target_bis_low=target_bis_low,
        target_bis_high=target_bis_high,
        map_abs_min_target=map_abs_min_target,
        map_rel_frac_target=map_rel_frac_target,
        fixed_bolus_mgkg=fixed_bolus_mgkg,
        seed=seed,
    )

    return recommender.optimize(skip_confidence=True)


def recommend_propofol_for_fixed_remifentanil_rate(
    patient: Patient,
    fixed_remi_rate_mcgkgmin: float,
    propofol_conc_mg_ml: float = DEFAULT_PROPOFOL_CONC_MG_ML,
    remifentanil_conc_mcg_ml: float = DEFAULT_REMI_CONC_MCG_ML,
    target_bis_low: float = TARGET_BIS_LOW,
    target_bis_high: float = TARGET_BIS_HIGH,
    map_abs_min_target: float = MAP_ABS_MIN_TARGET,
    map_rel_frac_target: float = MAP_REL_FRAC_TARGET,
    mode: str = OPTIMIZATION_MODE,
    seed: int = OPTIMIZATION_BASE_SEED,
) -> RecommendationResult:
    """
    Re-optimize the propofol induction bolus AND its own maintenance
    schedule for a fixed, user-chosen remifentanil maintenance rate
    (mcg/kg/min, held constant for the full 15-minute window - no pause,
    one rate throughout). This is the inverse of
    recommend_maintenance_for_fixed_propofol_bolus: here remifentanil is
    the fixed input and propofol is the free output, for pages where the
    opioid strategy is what the user manipulates and the propofol
    recommendation is what the model produces in response.

    Confidence IS computed (unlike the manual-override path above, which
    intentionally skips it) - this is meant to stand in as a page's own
    full baseline recommendation, not a cheap live-preview overlay, so it
    pays the same Powell-search + 100-simulation-confidence cost as
    recommend_su2023_regimen itself.
    """
    weight_kg = float(patient.weight)
    if not np.isfinite(weight_kg) or weight_kg <= 0:
        raise ValueError("Patient weight must be positive.")

    fixed_remi_rate_mcgkgmin = float(fixed_remi_rate_mcgkgmin)
    if not np.isfinite(fixed_remi_rate_mcgkgmin) or fixed_remi_rate_mcgkgmin < 0:
        raise ValueError("fixed_remi_rate_mcgkgmin must be non-negative.")

    recommender = Su2023PropofolRemifentanilRecommender(
        patient=patient,
        use_remifentanil=True,
        use_bsv=False,
        mode=mode,
        propofol_conc_mg_ml=propofol_conc_mg_ml,
        remifentanil_conc_mcg_ml=remifentanil_conc_mcg_ml,
        target_bis_low=target_bis_low,
        target_bis_high=target_bis_high,
        map_abs_min_target=map_abs_min_target,
        map_rel_frac_target=map_rel_frac_target,
        fixed_remi_rate_mcgkgmin=fixed_remi_rate_mcgkgmin,
        seed=seed,
    )

    return recommender.optimize()


def compress_minute_schedule(rates: Optional[Sequence[float]]) -> list[dict[str, float]]:
    """Compress minute-wise rates into contiguous segments for reporting."""
    if rates is None:
        return []

    rates = np.asarray(rates, dtype=float)
    if len(rates) == 0:
        return []

    rows: list[dict[str, float]] = []
    start = 0
    current = float(rates[0])

    for i in range(1, len(rates)):
        value = float(rates[i])
        if not np.isclose(value, current, atol=1e-8):
            rows.append({"start_min": float(start), "end_min": float(i), "rate": current})
            start = i
            current = value

    rows.append({"start_min": float(start), "end_min": float(len(rates)), "rate": current})
    return rows
