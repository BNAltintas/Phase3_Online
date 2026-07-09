"""
Offline precompute script for the "Precomputed Remi" page.

Run manually, before a Phase 3 testing session - never imported by the
running app and never invoked during participant interaction. It calls the
same real model the Recommendation page already uses
(recommend_su2023_regimen / Su2023PropofolRemifentanilRecommender from
recommend_regimen2023.py, completely unmodified), for each of the three
fixed Phase 3 patient cases already defined once in app.py
(PRESET_TEST_PATIENTS, reused here rather than re-typing the same numbers
a second time and risking the two definitions drifting apart), and writes
the results to src/propofol/data/precomputed_remi_cases.json.

For each case this computes:
    - a "no opioid" baseline recommendation (opiate="none", confidence
      computed)
    - a "remifentanil" baseline recommendation (opiate="remifentanil",
      confidence computed)
    - a 20-point remifentanil rate grid (0.02-0.20 mcg/kg/min), each point
      a full propofol-bolus + propofol-maintenance re-optimization for that
      fixed remifentanil rate (Su2023PropofolRemifentanilRecommender's
      existing fixed_remi_rate_mcgkgmin parameter), confidence skipped
      (skip_confidence=True) since 20 x 100-simulation Monte Carlo runs
      per case is unnecessary cost for exploratory grid points - only the
      two baselines need a real confidence estimate.
    - for every one of those regimens (both baselines + all 20 grid
      points), a propofol induction-dose sweep (_dose_sweep) - the same
      forward-simulation the Recommendation page's own
      make_induction_dose_rationale_figure computes live, run once here
      instead so the Precomputed Remi page can redraw that exact graph
      style from stored data, never by calling simulate_regimen during
      participant interaction.

Every result is serialized with the app's own existing _rec_to_store_dict
(app.py) - the exact same function the Recommendation page already uses to
put a RecommendationResult into a dcc.Store - so the JSON this script
writes is byte-for-byte the same shape the app already knows how to turn
back into Plotly figures via _store_dict_to_namespace + make_bis_figure /
make_bis_figure_dual / etc. No new serialization format, no drift risk
between "what the script writes" and "what the app expects to read".

Usage:
    python scripts/precompute_remi_cases.py
"""
from __future__ import annotations

import datetime
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

from propofol.app import (
    PRESET_TEST_PATIENTS,
    _clinical_propofol_dose_grid_mgkg,
    _rec_to_store_dict,
)
from propofol.patient import EleveldPatient as Patient
from propofol.recommend_regimen2023 import (
    DEFAULT_PROPOFOL_CONC_MG_ML,
    DEFAULT_REMI_CONC_MCG_ML,
    MAP_ABS_MIN_TARGET,
    MAP_REL_FRAC_TARGET,
    OPTIMIZATION_MODE,
    TARGET_ASSESSMENT_START_MIN,
    TARGET_BIS_HIGH,
    TARGET_BIS_LOW,
    DecodedRegimen,
    Su2023PropofolRemifentanilRecommender,
    recommend_su2023_regimen,
)

OUTPUT_PATH = REPO_ROOT / "src" / "propofol" / "data" / "precomputed_remi_cases.json"

GRID_MIN_MCGKGMIN = 0.02
GRID_MAX_MCGKGMIN = 0.20
GRID_POINTS = 20

# "TEST-001"/"TEST-002"/"TEST-003", matching the Recommendation page's own
# Test Patient card labeling for the same 3 presets.
CASE_PATIENT_IDS = {"1": "TEST-001", "2": "TEST-002", "3": "TEST-003"}


def _build_patient(preset: dict) -> Patient:
    return Patient(
        age=preset["age"],
        height=preset["height"],
        weight=preset["weight"],
        sex=preset["sex"],
        opiates=True,
        blood_sampling_site="arterial",
        base_sap=preset["baseline_sap"],
        base_dap=preset["baseline_dap"],
        base_hr=preset["baseline_hr"],
    )


def _dose_sweep(patient: Patient, rec) -> dict:
    """
    Offline counterpart to app.py's own make_induction_dose_rationale_
    figure (Recommendation page, never modified): vary only the propofol
    induction bolus while holding `rec`'s own maintenance regimen fixed,
    forward-simulating (Su2023PropofolRemifentanilRecommender.
    simulate_regimen - a single PK/PD pass, not a re-optimization) at
    each dose on the exact same local grid
    (_clinical_propofol_dose_grid_mgkg, imported unchanged from app.py)
    the Recommendation page itself would use for this same dose. Stored
    once per regimen (baseline_none/baseline_remifentanil/each grid
    point) so the Precomputed Remi page can redraw the identical curve at
    interaction time by reading this array back, never by calling
    simulate_regimen live.
    """
    selected_dose_mgkg = float(rec.propofol_bolus_mgkg)
    dose_grid_mgkg = _clinical_propofol_dose_grid_mgkg(selected_dose_mgkg)

    runner = Su2023PropofolRemifentanilRecommender(
        patient=patient,
        use_remifentanil=bool(rec.remifentanil_selected),
        use_bsv=False,
        propofol_conc_mg_ml=DEFAULT_PROPOFOL_CONC_MG_ML,
        remifentanil_conc_mcg_ml=DEFAULT_REMI_CONC_MCG_ML,
        target_bis_low=rec.target_bis_low,
        target_bis_high=rec.target_bis_high,
        map_abs_min_target=rec.map_abs_min_target_mmhg,
        map_rel_frac_target=rec.map_rel_frac_target,
    )

    min_maps, min_bis_values, max_bis_values = [], [], []
    for dose_mgkg in dose_grid_mgkg:
        regimen = DecodedRegimen(
            propofol_bolus_mg=float(dose_mgkg * patient.weight),
            propofol_bolus_mgkg=float(dose_mgkg),
            propofol_rates_mgkgh=np.asarray(rec.propofol_inf_rates_mgkgh, dtype=float).copy(),
            propofol_rates_ml_h=np.asarray(rec.propofol_inf_rates_ml_h, dtype=float).copy(),
            propofol_rates_mcgkgmin=np.asarray(rec.propofol_inf_rates_mcgkgmin, dtype=float).copy(),
            remifentanil_selected=bool(rec.remifentanil_selected),
            remifentanil_bolus_mcg=float(rec.remifentanil_bolus_mcg),
            remifentanil_bolus_mcgkg=float(rec.remifentanil_bolus_mcgkg),
            remifentanil_rates_mcgkgmin=(
                np.asarray(rec.remifentanil_inf_rates_mcgkgmin, dtype=float).copy()
                if rec.remifentanil_inf_rates_mcgkgmin is not None else None
            ),
            remifentanil_rates_ml_h=(
                np.asarray(rec.remifentanil_inf_rates_ml_h, dtype=float).copy()
                if rec.remifentanil_inf_rates_ml_h is not None else None
            ),
            remifentanil_rates_ngkgmin=(
                np.asarray(rec.remifentanil_inf_rates_ngkgmin, dtype=float).copy()
                if rec.remifentanil_inf_rates_ngkgmin is not None else None
            ),
        )
        try:
            t, _, _, _, bis, map_mmhg = runner.simulate_regimen(regimen)
            mask = np.asarray(t, dtype=float) >= TARGET_ASSESSMENT_START_MIN
            if not np.any(mask):
                min_maps.append(float("nan"))
                min_bis_values.append(float("nan"))
                max_bis_values.append(float("nan"))
                continue
            bis_after = np.asarray(bis, dtype=float)[mask]
            map_after = np.asarray(map_mmhg, dtype=float)[mask]
            min_maps.append(float(np.nanmin(map_after)))
            min_bis_values.append(float(np.nanmin(bis_after)))
            max_bis_values.append(float(np.nanmax(bis_after)))
        except Exception:  # noqa: BLE001 - one bad sweep point must not fail the whole case
            min_maps.append(float("nan"))
            min_bis_values.append(float("nan"))
            max_bis_values.append(float("nan"))

    return {
        "dose_grid_mgkg": [float(x) for x in dose_grid_mgkg],
        "min_map_mmhg": min_maps,
        "min_bis": min_bis_values,
        "max_bis": max_bis_values,
    }


def _augmented_store_dict(rec, patient: Patient) -> dict:
    """_rec_to_store_dict, plus this regimen's own precomputed dose_sweep (see _dose_sweep)."""
    d = _rec_to_store_dict(rec)
    d["dose_sweep"] = _dose_sweep(patient, rec)
    return d


def _grid_point(patient: Patient, rate: float) -> dict:
    recommender = Su2023PropofolRemifentanilRecommender(
        patient=patient,
        use_remifentanil=True,
        use_bsv=False,
        fixed_remi_rate_mcgkgmin=float(rate),
    )
    return _augmented_store_dict(recommender.optimize(skip_confidence=True), patient)


def precompute_case(case_id: str, preset: dict) -> dict:
    print(f"[case {case_id}] {preset['label']} - building patient...")
    patient = _build_patient(preset)

    print(f"[case {case_id}] baseline (no opioid)...")
    t0 = time.perf_counter()
    baseline_none = _augmented_store_dict(
        recommend_su2023_regimen(patient=patient, opiate="none"), patient,
    )
    print(f"[case {case_id}] baseline (no opioid) done in {time.perf_counter() - t0:.1f}s")

    print(f"[case {case_id}] baseline (remifentanil)...")
    t0 = time.perf_counter()
    baseline_remifentanil = _augmented_store_dict(
        recommend_su2023_regimen(patient=patient, opiate="remifentanil"), patient,
    )
    print(f"[case {case_id}] baseline (remifentanil) done in {time.perf_counter() - t0:.1f}s")

    rates = np.linspace(GRID_MIN_MCGKGMIN, GRID_MAX_MCGKGMIN, GRID_POINTS)
    grid_results = []
    for i, rate in enumerate(rates):
        t0 = time.perf_counter()
        grid_results.append(_grid_point(patient, rate))
        print(
            f"[case {case_id}] grid {i + 1}/{GRID_POINTS} "
            f"(rate={rate:.4f} mcg/kg/min) done in {time.perf_counter() - t0:.1f}s"
        )

    return {
        "case_id": case_id,
        "label": preset["label"],
        "patient_id": CASE_PATIENT_IDS[case_id],
        "patient": {
            "age": preset["age"],
            "height": preset["height"],
            "weight": preset["weight"],
            "sex": preset["sex"],
            "baseline_sap": preset["baseline_sap"],
            "baseline_dap": preset["baseline_dap"],
            "baseline_hr": preset["baseline_hr"],
        },
        "baseline_none": baseline_none,
        "baseline_remifentanil": baseline_remifentanil,
        "remifentanil_grid": {
            "rates_mcgkgmin": [float(r) for r in rates],
            "results": grid_results,
        },
    }


def main() -> None:
    started = time.perf_counter()
    cases = {}
    failed = []

    for case_id, preset in PRESET_TEST_PATIENTS.items():
        try:
            cases[case_id] = precompute_case(case_id, preset)
        except Exception as exc:  # noqa: BLE001 - a script, not a callback: fail loud, keep going
            print(f"[case {case_id}] FAILED: {exc!r}")
            failed.append(case_id)

    output = {
        "metadata": {
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "model_module": "recommend_regimen2023.Su2023PropofolRemifentanilRecommender",
            "optimization_mode": OPTIMIZATION_MODE,
            "grid_min_mcgkgmin": GRID_MIN_MCGKGMIN,
            "grid_max_mcgkgmin": GRID_MAX_MCGKGMIN,
            "grid_points": GRID_POINTS,
            "target_bis_low": float(TARGET_BIS_LOW),
            "target_bis_high": float(TARGET_BIS_HIGH),
            "map_abs_min_target_mmhg": float(MAP_ABS_MIN_TARGET),
            "map_rel_frac_target": float(MAP_REL_FRAC_TARGET),
            "propofol_conc_mg_ml": float(DEFAULT_PROPOFOL_CONC_MG_ML),
            "remifentanil_conc_mcg_ml": float(DEFAULT_REMI_CONC_MCG_ML),
            "failed_cases": failed,
        },
        "cases": cases,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output), encoding="utf-8")

    elapsed = time.perf_counter() - started
    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print()
    print(f"Wrote {OUTPUT_PATH} ({size_kb:.0f} KB) in {elapsed:.1f}s")
    if failed:
        print(f"WARNING: {len(failed)} case(s) failed and are missing from the output: {failed}")


if __name__ == "__main__":
    main()
