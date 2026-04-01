
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

from propofol.propofol_pkpd import EleveldPK, EleveldPD
from propofol.haemo_pd import SuHaemoPD


# ============================================================
# Configuration
# ============================================================

SIM_MIN = 15
TIME = np.linspace(0.0, SIM_MIN, SIM_MIN * 60 + 1)  # 1-second steps, time in minutes
N_INTERVALS = 15  # one maintenance decision per minute

# BIS targets
BIS_LOW = 40.0
BIS_HIGH = 60.0
BIS_TARGET = 50.0

# MAP thresholds
MAP_ABS_MIN = 65.0
MAP_REL_FRAC = 0.70

# Search bounds
BOLUS_MGKG_BOUNDS = (0.2, 3.5)
INFUSION_MGKGH_BOUNDS = (0.0, 20.0)  # includes pause at 0

# Optional clinical rounding for maintenance rates
# e.g. 0.5 or 1.0. Leave None for continuous rates.
MAINTENANCE_RATE_STEP = None


# ============================================================
# Helpers
# ============================================================

def round_to_5mg(x_mg: float) -> float:
    return float(5.0 * np.round(x_mg / 5.0))


def round_rate(rate_mgkgh: float, step: float | None) -> float:
    if step is None:
        return float(rate_mgkgh)
    return float(step * np.round(rate_mgkgh / step))


def round_rate_vector(rates: Sequence[float], step: float | None) -> np.ndarray:
    rates = np.asarray(rates, dtype=float)
    if step is None:
        return rates.copy()
    return step * np.round(rates / step)


def mode_to_max_changes(mode: str) -> int:
    mode = mode.lower()
    if mode == "lazy":
        return 1
    if mode == "auto":
        return 3
    if mode == "accurate":
        return 14
    raise ValueError("mode must be one of: 'lazy', 'auto', 'accurate'")


def count_rate_changes(rates: Sequence[float], tol: float = 1e-8) -> int:
    rates = np.asarray(rates, dtype=float)
    return int(np.sum(np.abs(np.diff(rates)) > tol))


def decode_segment_schedule(
    x_schedule: Sequence[float],
    mode: str,
    n_minutes: int = N_INTERVALS,
    rate_step: float | None = MAINTENANCE_RATE_STEP,
) -> np.ndarray:
    """
    Decode optimizer variables into a 15-minute schedule that can NEVER exceed
    the allowed number of changes.

    Construction:
      - Let max_changes = K
      - Then use exactly K+1 contiguous segments
      - Each segment has one rate
      - Segments may collapse to identical rates, so actual changes can be fewer
      - But actual changes can never exceed K

    Parameterization:
      x_schedule = [rate_1, ..., rate_S, w_1, ..., w_S]
      where S = K + 1

      - rates are segment rates
      - w are positive-like raw segment weights
      - normalized weights determine segment lengths that sum to n_minutes
      - each segment gets at least 1 minute
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


def n_schedule_params_for_mode(mode: str) -> int:
    max_changes = mode_to_max_changes(mode)
    n_segments = max_changes + 1
    return 2 * n_segments


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
    bolus_mg: float
    bolus_mgkg: float
    infusion_rates_mgkgh: np.ndarray
    mode: str

    time_min: np.ndarray
    cp: np.ndarray
    ce: np.ndarray
    bis: np.ndarray
    map_mmHg: np.ndarray

    objective_value: float
    feasible_bis: bool
    feasible_map: bool
    n_rate_changes: int
    max_rate_changes_allowed: int


# ============================================================
# Core recommender
# ============================================================

class PropofolDoseRecommender:
    def __init__(
        self,
        patient,
        baseline_map: float,
        hemo_model: str = "Su2022",
        use_bsv: bool = False,
        mode: str = "auto",
        maintenance_rate_step: float | None = MAINTENANCE_RATE_STEP,
    ) -> None:
        self.patient = patient
        self.weight_kg = float(patient.weight)
        self.baseline_map = float(baseline_map)
        self.hemo_model = hemo_model
        self.use_bsv = use_bsv
        self.mode = mode.lower()
        self.maintenance_rate_step = maintenance_rate_step
        self.max_changes = mode_to_max_changes(self.mode)

        self.pk = EleveldPK(patient=patient, use_bsv=use_bsv)
        self.pd = EleveldPD(patient=patient, use_bsv=use_bsv)

        if self.hemo_model != "Su2022":
            raise ValueError(
                f"Unsupported hemodynamic model '{self.hemo_model}'. "
                "Currently only 'Su2022' is implemented."
            )

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
        t, Cp, Ce, BIS, MAP_mmHg, bolus_mg, minute_rates_used
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
        BIS = np.array([self.pd.bis(x) for x in Ce], dtype=float)

        map0 = MAP_model[0]
        if map0 <= 0:
            raise ValueError("Initial haemodynamic MAP model output is non-positive.")
        MAP_mmHg = self.baseline_map * (MAP_model / map0)

        return TIME, Cp, Ce, BIS, MAP_mmHg, bolus_mg, minute_rates_used

    def objective(self, x: np.ndarray) -> float:
        bolus_mgkg = float(x[0])
        schedule_params = np.asarray(x[1:], dtype=float)

        t, cp, ce, bis, map_mmHg, bolus_mg, minute_rates_used = self.simulate(
            bolus_mgkg=bolus_mgkg,
            schedule_params=schedule_params,
        )

        map_min_allowed = max(MAP_ABS_MIN, MAP_REL_FRAC * self.baseline_map)

        # BIS penalty: prioritized
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

        # MAP penalty
        map_violation = np.clip(map_min_allowed - map_mmHg, 0.0, None)
        map_penalty = 140.0 * np.sum(map_violation ** 2)

        severe_hypo = np.clip(55.0 - map_mmHg, 0.0, None)
        map_penalty += 600.0 * np.sum(severe_hypo ** 2)

        # Regularization
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
        n_sched = n_schedule_params_for_mode(self.mode)
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
            maxiter=80,
            popsize=16,
            seed=42,
            polish=True,
            tol=0.02,
            workers=1,
            updating="deferred",
        )

        bolus_mgkg = float(result.x[0])
        schedule_params = np.asarray(result.x[1:], dtype=float)

        t, cp, ce, bis, map_mmHg, bolus_mg, minute_rates_used = self.simulate(
            bolus_mgkg=bolus_mgkg,
            schedule_params=schedule_params,
        )

        map_min_allowed = max(MAP_ABS_MIN, MAP_REL_FRAC * self.baseline_map)

        feasible_bis = bool(np.all((bis >= BIS_LOW) & (bis <= BIS_HIGH)))
        feasible_map = bool(np.all(map_mmHg >= map_min_allowed))

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
            map_mmHg=map_mmHg,
            objective_value=float(result.fun),
            feasible_bis=feasible_bis,
            feasible_map=feasible_map,
            n_rate_changes=n_changes,
            max_rate_changes_allowed=self.max_changes,
        )


# ============================================================
# Reporting
# ============================================================

def print_summary(rec: RecommendationResult, baseline_map: float) -> None:
    map_min_allowed = max(MAP_ABS_MIN, MAP_REL_FRAC * baseline_map)

    print("\n================ RECOMMENDED REGIMEN ================\n")
    print(f"Mode: {rec.mode}")
    print(f"Bolus: {rec.bolus_mg:.0f} mg ({rec.bolus_mgkg:.3f} mg/kg)\n")

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


# ============================================================
# Plotting
# ============================================================

def plot_recommendation(rec: RecommendationResult, baseline_map: float) -> None:
    map_min_allowed = max(MAP_ABS_MIN, MAP_REL_FRAC * baseline_map)

    plt.figure(figsize=(10, 5))
    plt.plot(rec.time_min, rec.cp, label="Plasma concentration (Cp)")
    plt.plot(rec.time_min, rec.ce, label="Effect-site concentration (Ce)")
    plt.xlabel("Time (min)")
    plt.ylabel("Concentration (mg/L)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(rec.time_min, rec.bis)
    plt.axhline(BIS_LOW, linestyle="--")
    plt.axhline(BIS_HIGH, linestyle="--")
    plt.ylim(0, 100)
    plt.xlabel("Time (min)")
    plt.ylabel("BIS")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(rec.time_min, rec.map_mmHg)
    plt.axhline(map_min_allowed, linestyle="--")
    plt.ylim(0, max(160, float(np.max(rec.map_mmHg)) * 1.1))
    plt.xlabel("Time (min)")
    plt.ylabel("MAP (mmHg)")
    plt.tight_layout()
    plt.show()


# ============================================================
# User-facing function
# ============================================================

def recommend_propofol_regimen(
    patient,
    baseline_map: float,
    hemo_model: str = "Su2022",
    use_bsv: bool = False,
    mode: str = "auto",
    maintenance_rate_step: float | None = MAINTENANCE_RATE_STEP,
) -> RecommendationResult:
    """
    Parameters
    ----------
    patient :
        Patient object compatible with EleveldPK / EleveldPD / SuHaemoPD.
    baseline_map : float
        Baseline MAP in mmHg.
    hemo_model : str
        Currently only "Su2022".
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
        baseline_map=baseline_map,
        hemo_model=hemo_model,
        use_bsv=use_bsv,
        mode=mode,
        maintenance_rate_step=maintenance_rate_step,
    )

    rec = recommender.optimize()
    print_summary(rec, baseline_map)
    return rec

