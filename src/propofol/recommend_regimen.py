
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import differential_evolution

from propofol.config import (
    BIS_HIGH,
    BIS_LOW,
    BIS_TARGET,
    BOLUS_MGKG_BOUNDS,
    INFUSION_MGKGH_BOUNDS,
    MAINTENANCE_RATE_STEP,
    MAP_ABS_MIN,
    MAP_REL_FRAC,
    N_INTERVALS,
    TIME,
)
from propofol.haemo_pd import SuHaemoPD
from propofol.propofol_pkpd import EleveldPD, EleveldPK

# ============================================================
# Helpers
# ============================================================

def round_to_5mg(x_mg: float) -> float:
    """Round bolus to nearest 5 mg."""
    return float(5.0 * np.round(x_mg / 5.0))


def round_rate_vector(rates: Sequence[float], step: float | None) -> np.ndarray:
    """Round a vector of maintenance rates to the nearest step."""
    rates = np.asarray(rates, dtype=float)
    if step is None:
        return rates.copy()
    return step * np.round(rates / step)


def mode_to_max_changes(mode: str) -> int:
    """Convert mode string to maximum number of allowed rate changes."""
    mode = mode.lower()
    if mode == "lazy":
        return 1
    if mode == "auto":
        return 3
    if mode == "accurate":
        return 14
    raise ValueError("mode must be one of: 'lazy', 'auto', 'accurate'")


def count_rate_changes(rates: Sequence[float], tol: float = 1e-8) -> int:
    """Count the number of rate changes in a sequence of rates."""
    rates = np.asarray(rates, dtype=float)
    return int(np.sum(np.abs(np.diff(rates)) > tol))


def decode_segment_schedule(
    x_schedule: Sequence[float],
    mode: str,
    n_minutes: int = N_INTERVALS,
    rate_step: float | None = MAINTENANCE_RATE_STEP,
) -> np.ndarray:
    """
    Convert the optimizer parameter vector into a minute-wise maintenance
    infusion schedule with a hard upper bound on the number of rate changes.

    The schedule is represented as a fixed number of contiguous segments.
    For a mode allowing K rate changes, the schedule is parameterized with
    K + 1 segments. Because each segment has a single constant infusion rate,
    the resulting schedule can never contain more than K transitions between adjacent minutes.

    Parameterization
    ----------------
    x_schedule is split into two parts:

        [rate_1, ..., rate_S, weight_1, ..., weight_S]

    where:
        S = max_changes + 1

    - rate_i:
        Infusion rate assigned to segment i.
    - weight_i:
        Raw positive segment-size parameter used to determine how many minutes
        segment i occupies.

    Segment duration construction
    -----------------------------
    The raw weights are normalized to relative proportions and then converted
    to integer segment lengths that sum exactly to `n_minutes`.

    The conversion is performed in three steps:
    1. Each segment is first assigned a minimum length of 1 minute.
    2. The remaining minutes are distributed across segments according to the
       normalized weights.
    3. Any leftover minutes caused by flooring are assigned to the segments
       with the largest fractional remainders.

    This ensures that:
    - every segment is present,
    - total schedule length is exactly `n_minutes`,
    - the number of possible rate changes is strictly bounded by construction.

    Notes
    -----
    - To do: consider segments of variable length with second-wise infusion rate changes.
             Discuss if this is feasible for clinical implementation.
    """
    mode = mode.lower()
    max_changes = mode_to_max_changes(mode)
    n_segments = max_changes + 1

    x_schedule = np.asarray(x_schedule, dtype=float)
    if len(x_schedule) != 2 * n_segments:
        raise ValueError(
            f"Expected {2 * n_segments} schedule variables for mode '{mode}', got {len(x_schedule)}"
        )

    raw_rates = x_schedule[:n_segments]
    raw_weights = x_schedule[n_segments:]

    segment_rates = round_rate_vector(raw_rates, step=rate_step)

    # Ensure positive weights, then allocate integer minute lengths summing to n_minutes
    weights = np.clip(raw_weights, 1e-6, None)
    frac = weights / np.sum(weights)

    # Give every segment at least 1 minute
    # This is feasible because n_segments = max_changes + 1 <= 15 for all modes
    base_lengths = np.ones(n_segments, dtype=int)
    remaining = n_minutes - n_segments

    if remaining < 0:
        raise ValueError("Too many segments for available minutes.")

    target_extra = frac * remaining
    extra_floor = np.floor(target_extra).astype(int)
    lengths = base_lengths + extra_floor

    leftover = remaining - np.sum(extra_floor)
    if leftover > 0:
        order = np.argsort(-(target_extra - extra_floor))
        lengths[order[:leftover]] += 1

    if np.sum(lengths) != n_minutes:
        raise RuntimeError("Internal error: segment lengths do not sum to n_minutes.")

    minute_rates = np.repeat(segment_rates, lengths)

    if len(minute_rates) != n_minutes:
        raise RuntimeError("Internal error: decoded schedule length mismatch.")

    return minute_rates.astype(float)

# ============================================================
# Dosing object for SuHaemoPD.solve_ode()
# ============================================================

class PiecewisePropofolDosing:
    """
    Supplies propofol input rate dotA0(t) in mg/min.

    - Bolus over first second
    - Maintenance begins after first second
    - Maintenance rate is piecewise constant over each minute
    """

    def __init__(
        self,
        bolus_mg: float,
        infusion_rates_mgkgh: Sequence[float],
        weight_kg: float,
    ) -> None:
        self.bolus_mg = float(bolus_mg)
        self.infusion_rates_mgkgh = np.asarray(infusion_rates_mgkgh, dtype=float)
        self.weight_kg = float(weight_kg)

        self.tcrit = sorted(set(
            [0.0, 1.0 / 60.0] + [float(i) for i in range(1, len(self.infusion_rates_mgkgh) + 1)]
        ))

    def dotA0(self, t: float) -> float:
        """Return propofol input rate in mg/min at time t (in minutes)."""
        if 0.0 <= t < (1.0 / 60.0):
            return self.bolus_mg / (1.0 / 60.0)  # mg/min

        idx = int(np.floor(t))
        idx = max(0, min(idx, len(self.infusion_rates_mgkgh) - 1))
        return self.infusion_rates_mgkgh[idx] * self.weight_kg / 60.0  # mg/min


# ============================================================
# Result container
# ============================================================

@dataclass
class RecommendationResult:
    """Container for recommended regimen and associated simulation results."""
    bolus_mg: float
    bolus_mgkg: float
    infusion_rates_mgkgh: np.ndarray
    mode: str

    time_min: np.ndarray
    cp: np.ndarray
    ce: np.ndarray
    bis: np.ndarray
    map_mmhg: np.ndarray

    objective_value: float
    feasible_bis: bool
    feasible_map: bool
    n_rate_changes: int
    max_rate_changes_allowed: int


# ============================================================
# Core recommender
# ============================================================

class PropofolDoseRecommender:
    """Recommends a propofol regimen by optimizing over bolus dose and piecewise maintenance
    infusion rates with a hard limit on the number of rate changes.
    """
    def __init__(
        self,
        patient,
        baseline_map: float,
        use_bsv: bool = False,
        mode: str = "auto",
        maintenance_rate_step: float | None = MAINTENANCE_RATE_STEP,
    ) -> None:
        self.patient = patient
        self.weight_kg = float(patient.weight)
        self.baseline_map = float(baseline_map)
        self.use_bsv = use_bsv
        self.mode = mode.lower()
        self.maintenance_rate_step = maintenance_rate_step
        self.max_changes = mode_to_max_changes(self.mode)

        self.pk = EleveldPK(patient=patient, use_bsv=use_bsv)
        self.pd = EleveldPD(patient=patient, use_bsv=use_bsv)

        self.haemo = SuHaemoPD(
            patient=patient,
            pk_propofol=self.pk,
            pd_propofol=self.pd,
        )

    def simulate(
        self,
        bolus_mgkg: float,
        schedule_params: Sequence[float],
    ):
        """
        Simulate regimen defined by:
          - bolus (mg/kg)
          - schedule_params decoded into a strict-change-limited 15-min schedule

        Returns
        -------
        t, Cp, Ce, bis, map_mmhg, bolus_mg, minute_rates_used
        """
        bolus_mg_raw = float(bolus_mgkg) * self.weight_kg
        bolus_mg = round_to_5mg(bolus_mg_raw)

        minute_rates_used = decode_segment_schedule(
            schedule_params,
            mode=self.mode,
            n_minutes=N_INTERVALS,
            rate_step=self.maintenance_rate_step,
        )

        dosing = PiecewisePropofolDosing(
            bolus_mg=bolus_mg,
            infusion_rates_mgkgh=minute_rates_used,
            weight_kg=self.weight_kg,
        )

        A1, A2, A3, Ce, sv, hr, MAP_model, tde = self.haemo.solve_ode(
            t=TIME,
            y0=None,
            dosing=dosing,
        )

        Cp = A1 / self.pk.V1
        bis = np.array([self.pd.bis(x) for x in Ce], dtype=float)

        # Convert haemodynamic model output to MAP, scaled to baseline provided by user.
        # To-do: for altered Su2023 model, ensure that baseline scaling is still appropriate.
        map0 = MAP_model[0]
        if map0 <= 0:
            raise ValueError("Initial haemodynamic MAP model output is non-positive.")
        map_mmhg = self.baseline_map * (MAP_model / map0)

        return TIME, Cp, Ce, bis, map_mmhg, bolus_mg, minute_rates_used

    def objective(self, x: np.ndarray) -> float:
        """Objective function for optimization."""
        bolus_mgkg = float(x[0])
        schedule_params = np.asarray(x[1:], dtype=float)

        t, cp, ce, bis, map_mmhg, bolus_mg, minute_rates_used = self.simulate(
            bolus_mgkg=bolus_mgkg,
            schedule_params=schedule_params,
        )

        # BIS penalty: priority is to avoid underdosing (BIS > 60)
        bis_under = np.clip(BIS_LOW - bis, 0.0, None)
        bis_over = np.clip(bis - BIS_HIGH, 0.0, None)
        bis_target_dev = np.abs(bis - BIS_TARGET)

        bis_penalty = (
            1800.0 * np.sum(bis_over ** 2) +
            700.0  * np.sum(bis_under ** 2) +
            2.5    * np.sum(bis_target_dev ** 2)
        )

        # Time to adequate BIS penalty: priority is to achieve adequate sedation quickly
        # To-do: consider adding option in dashboard to choose maximal acceptable time to sedation.
        adequate_idx = np.where(bis <= BIS_HIGH)[0]
        if len(adequate_idx) == 0:
            time_to_adequate_penalty = 25000.0
        else:
            time_to_adequate_penalty = 80.0 * t[adequate_idx[0]]

        # MAP penalty: priority is to avoid hypotension.
        # Increasing penalty for severe hypotension.
        map_min_allowed = max(MAP_ABS_MIN, MAP_REL_FRAC * self.baseline_map)

        map_violation = np.clip(map_min_allowed - map_mmhg, 0.0, None)
        map_penalty = 140.0 * np.sum(map_violation ** 2)

        severe_hypo = np.clip(55.0 - map_mmhg, 0.0, None)
        map_penalty += 600.0 * np.sum(severe_hypo ** 2)

        # Regularization: priority is to ensure smooth infusion rates and limit total dose.
        # To do: consider if penalty on total dose is needed.
        smoothness_penalty = 10.0 * np.sum(np.diff(minute_rates_used) ** 2)
        rate_l1_penalty = 1.5 * np.sum(minute_rates_used)

        total_maint_mg = np.sum(minute_rates_used * self.weight_kg / 60.0)
        total_dose_penalty = 0.15 * (bolus_mg + total_maint_mg)

        # No change-limit penalty needed:
        # the limit is guaranteed by construction
        return float(
            bis_penalty
            + time_to_adequate_penalty
            + map_penalty
            + smoothness_penalty
            + rate_l1_penalty
            + total_dose_penalty
        )

    def optimize(self) -> RecommendationResult:
        """
        Optimize the regimen using differential evolution.

        Optimization workflow
        ---------------------
        1. Construct parameter bounds for:
        - bolus dose,
        - segment rates,
        - segment weights.
        2. Run `scipy.optimize.differential_evolution` on the regimen objective.
        3. Decode the optimal parameter vector into a full minute-wise schedule.
        4. Re-simulate the optimal regimen to obtain PK, BIS, and MAP trajectories.
        5. Evaluate feasibility against BIS and MAP criteria.
        6. Return all results in a `RecommendationResult` object.

        Notes
        - To do: consider alternative (faster) optimization algorithms.
        """
        n_segments = mode_to_max_changes(self.mode) + 1

        # First n_segments vars = rates
        # Next n_segments vars = segment weights
        bounds = [BOLUS_MGKG_BOUNDS]
        bounds += [INFUSION_MGKGH_BOUNDS] * n_segments
        bounds += [(0.1, 10.0)] * n_segments  # segment weights

        result = differential_evolution(
            self.objective,
            bounds=bounds,
            strategy="best1bin",
            maxiter=20,
            popsize=6,
            tol=0.08,
            polish=False,
            init="sobol",
            updating="immediate",
            workers=1,
            seed=42,
        )

        bolus_mgkg = float(result.x[0])
        schedule_params = np.asarray(result.x[1:], dtype=float)

        t, cp, ce, bis, map_mmhg, bolus_mg, minute_rates_used = self.simulate(
            bolus_mgkg=bolus_mgkg,
            schedule_params=schedule_params,
        )

        map_min_allowed = max(MAP_ABS_MIN, MAP_REL_FRAC * self.baseline_map)

        feasible_bis = bool(np.all((bis >= BIS_LOW) & (bis <= BIS_HIGH)))
        feasible_map = bool(np.all(map_mmhg >= map_min_allowed))

        bolus_mgkg_effective = bolus_mg / self.weight_kg
        n_changes = count_rate_changes(minute_rates_used)

        # Sanity check: guaranteed by construction
        if n_changes > self.max_changes:
            raise RuntimeError(
                f"Internal error: produced {n_changes} changes, exceeds allowed {self.max_changes}."
            )

        return RecommendationResult(
            bolus_mg=bolus_mg,
            bolus_mgkg=bolus_mgkg_effective,
            infusion_rates_mgkgh=minute_rates_used,
            mode=self.mode,
            time_min=t,
            cp=cp,
            ce=ce,
            bis=bis,
            map_mmhg=map_mmhg,
            objective_value=float(result.fun),
            feasible_bis=feasible_bis,
            feasible_map=feasible_map,
            n_rate_changes=n_changes,
            max_rate_changes_allowed=self.max_changes,
        )


# ============================================================
# Reporting in terminal
# ============================================================

def print_summary(rec: RecommendationResult, baseline_map: float) -> None:
    """Print a summary of the recommended regimen and its performance."""
    map_min_allowed = max(MAP_ABS_MIN, MAP_REL_FRAC * baseline_map)

    print("\n================ RECOMMENDED REGIMEN ================\n")
    print(f"Mode: {rec.mode}")
    print(f"Induction dose: {rec.bolus_mg:.0f} mg ({rec.bolus_mgkg:.3f} mg/kg)\n")

    print("Minute-wise maintenance schedule (mg/kg/h):")
    for i, r in enumerate(rec.infusion_rates_mgkgh):
        txt = "pause" if np.isclose(r, 0.0, atol=1e-8) else f"{r:.2f}"
        print(f"  minute {i:02d}-{i+1:02d}: {txt}")

    print("\nPerformance:")
    adequate_idx = np.where(rec.bis <= BIS_HIGH)[0]
    if len(adequate_idx) > 0:
        print(f"  time to BIS <= 60: {rec.time_min[adequate_idx[0]]:.2f} min")
    else:
        print("  time to BIS <= 60: not reached")
    print(f"  Chosen threshold for MAP: {map_min_allowed:.1f} mmHg")

# ============================================================
# User-facing function
# ============================================================

def recommend_propofol_regimen(
    patient,
    use_bsv: bool = False,
    mode: str = "auto",
    maintenance_rate_step: float | None = MAINTENANCE_RATE_STEP,
) -> RecommendationResult:
    """
    Parameters
    ----------
    patient :
        Patient object compatible with EleveldPK / EleveldPD / SuHaemoPD.
    use_bsv : bool
        Whether to sample between-subject variability.
    mode : str
        "lazy", "auto", or "accurate".
    maintenance_rate_step : float | None
        Optional maintenance rate rounding step, e.g. 0.5 or 1.0 mg/kg/h.

    Returns
    -------
    RecommendationResult
    """
    recommender = PropofolDoseRecommender(
        patient=patient,
        baseline_map=patient.base_map,
        use_bsv=use_bsv,
        mode=mode,
        maintenance_rate_step=maintenance_rate_step,
    )

    rec = recommender.optimize()
    return rec

