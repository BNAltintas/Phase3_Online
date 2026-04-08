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
    if step <= 0:
        raise ValueError("step must be positive when provided")
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


def build_segment_lengths(weights: Sequence[float], n_minutes: int) -> np.ndarray:
    """
    Convert positive segment weights to integer segment lengths summing exactly to n_minutes.

    Every segment gets at least 1 minute.
    """
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1:
        raise ValueError("weights must be a 1D array")
    if len(weights) == 0:
        raise ValueError("weights must not be empty")

    weights = np.clip(weights, 1e-6, None)
    frac = weights / np.sum(weights)

    n_segments = len(weights)
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

    return lengths


def decode_segment_schedule(
    x_schedule: Sequence[float],
    mode: str,
    n_minutes: int = N_INTERVALS,
    rate_step: float | None = MAINTENANCE_RATE_STEP,
    apply_rounding: bool = False,
) -> np.ndarray:
    """
    Convert the optimizer parameter vector into a minute-wise maintenance
    infusion schedule with a hard upper bound on the number of rate changes.

    x_schedule is split into:
        [rate_1, ..., rate_S, weight_1, ..., weight_S]
    where S = max_changes + 1.

    Parameters
    ----------
    apply_rounding : bool
        If True, round segment rates to the requested clinical step.
        If False, keep them continuous for optimization.
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

    if apply_rounding:
        segment_rates = round_rate_vector(raw_rates, step=rate_step)
    else:
        segment_rates = np.asarray(raw_rates, dtype=float).copy()

    lengths = build_segment_lengths(raw_weights, n_minutes=n_minutes)
    minute_rates = np.repeat(segment_rates, lengths)

    if len(minute_rates) != n_minutes:
        raise RuntimeError("Internal error: decoded schedule length mismatch.")

    return minute_rates.astype(float)


def encode_schedule_from_segment_solution(
    segment_rates: Sequence[float],
    segment_weights: Sequence[float],
) -> np.ndarray:
    """Concatenate segment rates and weights into optimizer format."""
    segment_rates = np.asarray(segment_rates, dtype=float)
    segment_weights = np.asarray(segment_weights, dtype=float)
    if len(segment_rates) != len(segment_weights):
        raise ValueError("segment_rates and segment_weights must have equal length")
    return np.concatenate([segment_rates, segment_weights])


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
            return self.bolus_mg / (1.0 / 60.0)

        idx = int(np.floor(t))
        idx = max(0, min(idx, len(self.infusion_rates_mgkgh) - 1))
        return self.infusion_rates_mgkgh[idx] * self.weight_kg / 60.0


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
    objective_value_continuous: float
    feasible_bis: bool
    feasible_map: bool
    n_rate_changes: int
    max_rate_changes_allowed: int


# ============================================================
# Core recommender
# ============================================================

class PropofolDoseRecommender:
    """
    Recommends a propofol regimen by optimizing over bolus dose and piecewise maintenance
    infusion rates with a hard limit on the number of rate changes.

    Key design choice
    -----------------
    The optimizer searches a continuous space:
    - bolus is continuous during optimization
    - maintenance segment rates are continuous during optimization

    Rounding to clinically implementable increments is applied only after optimization,
    followed by re-simulation of the rounded regimen.

    This avoids unnecessary flat regions in the objective caused by rounding inside
    differential evolution.
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

    # --------------------------------------------------------
    # Regimen decoding / simulation
    # --------------------------------------------------------

    def simulate(
        self,
        bolus_mgkg: float,
        schedule_params: Sequence[float],
        apply_rounding: bool = False,
    ):
        """
        Simulate a regimen defined by:
          - bolus (mg/kg)
          - schedule_params decoded into a strict-change-limited schedule

        Parameters
        ----------
        apply_rounding : bool
            If False:
                use continuous bolus and continuous maintenance rates.
            If True:
                round bolus to nearest 5 mg and maintenance rates to the configured step.

        Returns
        -------
        t, Cp, Ce, bis, map_mmhg, bolus_mg_used, minute_rates_used
        """
        bolus_mg_raw = float(bolus_mgkg) * self.weight_kg
        bolus_mg_used = round_to_5mg(bolus_mg_raw) if apply_rounding else bolus_mg_raw

        minute_rates_used = decode_segment_schedule(
            schedule_params,
            mode=self.mode,
            n_minutes=N_INTERVALS,
            rate_step=self.maintenance_rate_step,
            apply_rounding=apply_rounding,
        )

        dosing = PiecewisePropofolDosing(
            bolus_mg=bolus_mg_used,
            infusion_rates_mgkgh=minute_rates_used,
            weight_kg=self.weight_kg,
        )

        A1, A2, A3, Ce, sv, hr, MAP_model, tde = self.haemo.solve_ode(
            t=TIME,
            y0=None,
            dosing=dosing,
        )

        Cp = A1 / self.pk.V1
        bis = np.vectorize(self.pd.bis)(Ce)

        map0 = MAP_model[0]
        if map0 <= 0:
            raise ValueError("Initial haemodynamic MAP model output is non-positive.")
        map_mmhg = self.baseline_map * (MAP_model / map0)

        return TIME, Cp, Ce, bis, map_mmhg, bolus_mg_used, minute_rates_used

    # --------------------------------------------------------
    # Objective
    # --------------------------------------------------------

    def objective(self, x: np.ndarray) -> float:
        """
        Objective function evaluated on the continuous regimen.
        """
        bolus_mgkg = float(x[0])
        schedule_params = np.asarray(x[1:], dtype=float)

        t, cp, ce, bis, map_mmhg, bolus_mg, minute_rates_used = self.simulate(
            bolus_mgkg=bolus_mgkg,
            schedule_params=schedule_params,
            apply_rounding=False,
        )

        # BIS penalty
        # Priority:
        # 1) avoid BIS > BIS_HIGH (too light)
        # 2) avoid BIS < BIS_LOW (too deep)
        # 3) remain near BIS_TARGET
        bis_under = np.clip(BIS_LOW - bis, 0.0, None)
        bis_over = np.clip(bis - BIS_HIGH, 0.0, None)
        bis_target_dev = np.abs(bis - BIS_TARGET)

        bis_penalty = (
            1800.0 * np.sum(bis_over ** 2) +
            700.0  * np.sum(bis_under ** 2) +
            2.5    * np.sum(bis_target_dev ** 2)
        )

        # Time to adequate BIS penalty
        adequate_idx = np.where(bis <= BIS_HIGH)[0]
        if len(adequate_idx) == 0:
            time_to_adequate_penalty = 25000.0
        else:
            time_to_adequate_penalty = 80.0 * t[adequate_idx[0]]

        # MAP penalty
        map_min_allowed = max(MAP_ABS_MIN, MAP_REL_FRAC * self.baseline_map)

        map_violation = np.clip(map_min_allowed - map_mmhg, 0.0, None)
        map_penalty = 140.0 * np.sum(map_violation ** 2)

        severe_hypo = np.clip(55.0 - map_mmhg, 0.0, None)
        map_penalty += 600.0 * np.sum(severe_hypo ** 2)

        # Regularization
        smoothness_penalty = 10.0 * np.sum(np.diff(minute_rates_used) ** 2)
        rate_l1_penalty = 1.5 * np.sum(minute_rates_used)

        total_maint_mg = np.sum(minute_rates_used * self.weight_kg / 60.0)
        total_dose_penalty = 0.15 * (bolus_mg + total_maint_mg)

        return float(
            bis_penalty
            + time_to_adequate_penalty
            + map_penalty
            + smoothness_penalty
            + rate_l1_penalty
            + total_dose_penalty
        )

    # --------------------------------------------------------
    # Post-optimization rounded refinement
    # --------------------------------------------------------

    def rounded_objective_from_components(
        self,
        bolus_mg: float,
        segment_rates: Sequence[float],
        segment_weights: Sequence[float],
    ) -> float:
        """
        Evaluate the objective after explicit rounding / discretization.

        This is used for local post-DE refinement in a discrete neighborhood.
        """
        bolus_mgkg = float(bolus_mg) / self.weight_kg
        schedule_params = encode_schedule_from_segment_solution(
            segment_rates=segment_rates,
            segment_weights=segment_weights,
        )

        t, cp, ce, bis, map_mmhg, bolus_mg_used, minute_rates_used = self.simulate(
            bolus_mgkg=bolus_mgkg,
            schedule_params=schedule_params,
            apply_rounding=True,
        )

        bis_under = np.clip(BIS_LOW - bis, 0.0, None)
        bis_over = np.clip(bis - BIS_HIGH, 0.0, None)
        bis_target_dev = np.abs(bis - BIS_TARGET)

        bis_penalty = (
            1800.0 * np.sum(bis_over ** 2) +
            700.0  * np.sum(bis_under ** 2) +
            2.5    * np.sum(bis_target_dev ** 2)
        )

        adequate_idx = np.where(bis <= BIS_HIGH)[0]
        if len(adequate_idx) == 0:
            time_to_adequate_penalty = 25000.0
        else:
            time_to_adequate_penalty = 80.0 * t[adequate_idx[0]]

        map_min_allowed = max(MAP_ABS_MIN, MAP_REL_FRAC * self.baseline_map)

        map_violation = np.clip(map_min_allowed - map_mmhg, 0.0, None)
        map_penalty = 140.0 * np.sum(map_violation ** 2)

        severe_hypo = np.clip(55.0 - map_mmhg, 0.0, None)
        map_penalty += 600.0 * np.sum(severe_hypo ** 2)

        smoothness_penalty = 10.0 * np.sum(np.diff(minute_rates_used) ** 2)
        rate_l1_penalty = 1.5 * np.sum(minute_rates_used)

        total_maint_mg = np.sum(minute_rates_used * self.weight_kg / 60.0)
        total_dose_penalty = 0.15 * (bolus_mg_used + total_maint_mg)

        return float(
            bis_penalty
            + time_to_adequate_penalty
            + map_penalty
            + smoothness_penalty
            + rate_l1_penalty
            + total_dose_penalty
        )

    def refine_discrete_solution(
        self,
        bolus_mgkg_cont: float,
        schedule_params_cont: Sequence[float],
    ) -> tuple[float, np.ndarray]:
        """
        Small local refinement around the continuous optimum after rounding.

        Strategy
        --------
        - Round bolus to nearest 5 mg, then try {rounded - 5, rounded, rounded + 5}
        - Round each segment rate to the nearest allowed step, then independently
          try {-step, 0, +step} for each segment with coordinate descent
        - Keep segment weights fixed; they already define the change-limited segmentation

        Returns
        -------
        best_bolus_mgkg, best_schedule_params
            Parameters encoded in the same format as the optimizer uses.
            These parameters are intended for simulation with apply_rounding=True.
        """
        n_segments = self.max_changes + 1
        schedule_params_cont = np.asarray(schedule_params_cont, dtype=float)

        segment_rates_cont = schedule_params_cont[:n_segments]
        segment_weights = schedule_params_cont[n_segments:]

        # Initial rounded values
        bolus_mg_cont = float(bolus_mgkg_cont) * self.weight_kg
        bolus_mg_rounded = round_to_5mg(bolus_mg_cont)

        if self.maintenance_rate_step is None:
            rate_step = None
            segment_rates_rounded = np.asarray(segment_rates_cont, dtype=float).copy()
        else:
            rate_step = float(self.maintenance_rate_step)
            segment_rates_rounded = round_rate_vector(segment_rates_cont, step=rate_step)

        best_bolus_mg = bolus_mg_rounded
        best_segment_rates = segment_rates_rounded.copy()

        best_obj = self.rounded_objective_from_components(
            bolus_mg=best_bolus_mg,
            segment_rates=best_segment_rates,
            segment_weights=segment_weights,
        )

        # Bolus local search: nearest neighbors in 5 mg increments
        bolus_candidates = [best_bolus_mg]
        bolus_candidates += [best_bolus_mg - 5.0, best_bolus_mg + 5.0]

        bolus_min_mg = BOLUS_MGKG_BOUNDS[0] * self.weight_kg
        bolus_max_mg = BOLUS_MGKG_BOUNDS[1] * self.weight_kg

        for b in bolus_candidates:
            if b < bolus_min_mg or b > bolus_max_mg:
                continue
            obj = self.rounded_objective_from_components(
                bolus_mg=b,
                segment_rates=best_segment_rates,
                segment_weights=segment_weights,
            )
            if obj < best_obj:
                best_obj = obj
                best_bolus_mg = b

        # Coordinate descent over rounded segment rates
        if rate_step is not None:
            improved = True
            while improved:
                improved = False
                for j in range(n_segments):
                    current = best_segment_rates[j]
                    candidates = [current - rate_step, current, current + rate_step]

                    seg_min, seg_max = INFUSION_MGKGH_BOUNDS
                    local_best_rate = current
                    local_best_obj = best_obj

                    for cand in candidates:
                        if cand < seg_min or cand > seg_max:
                            continue

                        trial_rates = best_segment_rates.copy()
                        trial_rates[j] = cand

                        obj = self.rounded_objective_from_components(
                            bolus_mg=best_bolus_mg,
                            segment_rates=trial_rates,
                            segment_weights=segment_weights,
                        )
                        if obj < local_best_obj:
                            local_best_obj = obj
                            local_best_rate = cand

                    if not np.isclose(local_best_rate, current):
                        best_segment_rates[j] = local_best_rate
                        best_obj = local_best_obj
                        improved = True

        best_schedule_params = encode_schedule_from_segment_solution(
            segment_rates=best_segment_rates,
            segment_weights=segment_weights,
        )
        best_bolus_mgkg = best_bolus_mg / self.weight_kg

        return float(best_bolus_mgkg), best_schedule_params

    # --------------------------------------------------------
    # Optimization
    # --------------------------------------------------------

    def optimize(self) -> RecommendationResult:
        """
        Optimize the regimen using differential evolution.

        Workflow
        --------
        1. Optimize the continuous regimen.
        2. Round and locally refine the continuous optimum.
        3. Re-simulate the final rounded regimen.
        4. Return the clinically implementable recommendation.
        """
        n_segments = self.max_changes + 1

        bounds = [BOLUS_MGKG_BOUNDS]
        bounds += [INFUSION_MGKGH_BOUNDS] * n_segments
        bounds += [(0.1, 10.0)] * n_segments

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

        # Best continuous solution
        bolus_mgkg_cont = float(result.x[0])
        schedule_params_cont = np.asarray(result.x[1:], dtype=float)

        objective_value_continuous = float(result.fun)

        # Refine in the discrete / rounded neighborhood
        bolus_mgkg_final, schedule_params_final = self.refine_discrete_solution(
            bolus_mgkg_cont=bolus_mgkg_cont,
            schedule_params_cont=schedule_params_cont,
        )

        # Final simulation uses the rounded regimen
        t, cp, ce, bis, map_mmhg, bolus_mg, minute_rates_used = self.simulate(
            bolus_mgkg=bolus_mgkg_final,
            schedule_params=schedule_params_final,
            apply_rounding=True,
        )

        objective_value_final = self.objective(
            np.concatenate([[bolus_mgkg_final], np.asarray(schedule_params_final, dtype=float)])
        )
        objective_value_final = self.rounded_objective_from_components(
            bolus_mg=bolus_mg,
            segment_rates=np.asarray(schedule_params_final[:n_segments], dtype=float),
            segment_weights=np.asarray(schedule_params_final[n_segments:], dtype=float),
        )

        map_min_allowed = max(MAP_ABS_MIN, MAP_REL_FRAC * self.baseline_map)

        feasible_bis = bool(np.all((bis >= BIS_LOW) & (bis <= BIS_HIGH)))
        feasible_map = bool(np.all(map_mmhg >= map_min_allowed))

        bolus_mgkg_effective = bolus_mg / self.weight_kg
        n_changes = count_rate_changes(minute_rates_used)

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
            objective_value=float(objective_value_final),
            objective_value_continuous=objective_value_continuous,
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
        print(f"  time to BIS <= {BIS_HIGH}: {rec.time_min[adequate_idx[0]]:.2f} min")
    else:
        print(f"  time to BIS <= {BIS_HIGH}: not reached")

    print(f"  Chosen threshold for MAP: {map_min_allowed:.1f} mmHg")
    print(f"  Feasible BIS: {rec.feasible_bis}")
    print(f"  Feasible MAP: {rec.feasible_map}")
    print(f"  Rate changes: {rec.n_rate_changes} / {rec.max_rate_changes_allowed}")
    print(f"  Final rounded objective: {rec.objective_value:.2f}")
    print(f"  Best continuous objective: {rec.objective_value_continuous:.2f}")


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
        Use None to keep the final rates continuous.

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
    return recommender.optimize()