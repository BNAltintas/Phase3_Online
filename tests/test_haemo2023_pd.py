import copy
from functools import lru_cache

import numpy as np

from propofol.haemo_pd_Su2023 import SuHaemoPD
from propofol.patient import EleveldPatient as Patient
from propofol.propofol_pkpd import EleveldPK as PropofolPK
from propofol.propofol_pkpd import EleveldPD as PropofolPD
from propofol.remifentanil_pkpd import EleveldPK as RemifentanilPK


# ============================================================
# Settings
# ============================================================

t = np.linspace(0.0, 180.0, 180 * 60 + 1)


# ============================================================
# Model helpers
# ============================================================

@lru_cache(maxsize=1)
def make_model():
    patient = Patient(
        age=35,
        height=170,
        weight=70,
        sex="male",
        opiates=False,
        base_sv=82.2,
        base_hr=56.0,
        base_tpr=0.016,
    )

    return SuHaemoPD(
        patient=patient,
        pk_propofol=PropofolPK(patient),
        pd_propofol=PropofolPD(patient),
        pk_remifentanil=RemifentanilPK(patient),
        use_bsv=False,
    )


def make_constant_concentration_model():
    m = copy.deepcopy(make_model())

    # Freeze propofol PK
    for attr in ("k10", "k12", "k13", "k21", "k31"):
        if hasattr(m.pk_propofol, attr):
            setattr(m.pk_propofol, attr, 0.0)

    # Freeze propofol PD
    if hasattr(m.pd_propofol, "ke0"):
        m.pd_propofol.ke0 = 0.0

    # Freeze remifentanil PK
    for attr in ("k10", "k12", "k13", "k21", "k31"):
        if hasattr(m.pk_remifentanil, attr):
            setattr(m.pk_remifentanil, attr, 0.0)

    return m


# ============================================================
# Simulation helpers
# ============================================================

@lru_cache(maxsize=None)
def simulate_constant_cp(cp_prop, cp_remi):
    """
    Simulate fixed propofol and remifentanil plasma concentrations.
    Results are cached to avoid repeated ODE solves.
    """
    m = make_constant_concentration_model()

    y0 = [
        cp_prop * m.pk_propofol.V1,        # A1: propofol central amount
        0.0,                               # A2
        0.0,                               # A3
        0.0,                               # Ce_prop
        cp_remi * m.pk_remifentanil.V1,    # A4: remifentanil central amount
        0.0,                               # A5
        0.0,                               # A6
        m.base_sv,                         # sv_ast
        m.base_hr,                         # hr_ast
        m.base_tpr,                        # tpr
        m.base_sv * m.ltde_sv,             # tde_sv
        m.base_hr * m.ltde_hr,             # tde_hr
    ]

    (
        A1, A2, A3, Ce_prop,
        A4, A5, A6,
        sv_ast, hr_ast, tpr,
        tde_sv, tde_hr,
        sv, MAP,
    ) = m.solve_ode(
        t=t,
        y0=y0,
        dosing_prop=None,
        dosing_remi=None,
    )

    return {
        "map": float(MAP[-1]),
        "hr": float((hr_ast + tde_hr)[-1]),
        "sv": float(sv[-1]),
    }


def pct_change(value, baseline):
    return 100.0 * (value - baseline) / baseline


@lru_cache(maxsize=None)
def effects(cp_prop, cp_remi):
    """
    Returns percentage changes for:
    - prop: propofol alone
    - full: propofol + remifentanil interaction model
    - null: propofol + remifentanil additive-null model
    """
    base = simulate_constant_cp(0.0, 0.0)
    remi_base = simulate_constant_cp(0.0, cp_remi)
    prop = simulate_constant_cp(cp_prop, 0.0)
    full = simulate_constant_cp(cp_prop, cp_remi)

    null = {
        key: prop[key] + remi_base[key] - base[key]
        for key in ("map", "hr", "sv")
    }

    return {
        "prop": {
            key: pct_change(prop[key], base[key])
            for key in ("map", "hr", "sv")
        },
        "full": {
            key: pct_change(full[key], remi_base[key])
            for key in ("map", "hr", "sv")
        },
        "null": {
            key: pct_change(null[key], remi_base[key])
            for key in ("map", "hr", "sv")
        },
    }


# ============================================================
# Tests
# ============================================================

def test_reported_map_values():
    """
    At Cpropofol = 1.1 µg/mL, MAP decrease should be:
    - 5.7% for propofol alone
    - 14.6% for propofol + remifentanil 2 ng/mL
    - 21.5% for propofol + remifentanil 4 ng/mL
    """
    assert round(effects(1.1, 0.0)["prop"]["map"], 1) == -5.7
    assert round(effects(1.1, 2.0)["full"]["map"], 1) == -14.6
    assert round(effects(1.1, 4.0)["full"]["map"], 1) == -21.5


def test_reported_hr_values():
    """
    At Cpropofol = 10 µg/mL, HR change should be:
    - 52.3% for propofol alone
    - 12.3% for propofol + remifentanil 2 ng/mL
    - 1.9% for propofol + remifentanil 4 ng/mL
    """
    assert round(effects(10.0, 0.0)["prop"]["hr"], 1) == 52.3
    assert round(effects(10.0, 2.0)["full"]["hr"], 1) == 12.3
    assert round(effects(10.0, 4.0)["full"]["hr"], 1) == 1.9


def test_hr_interaction_vs_null_model():
    """
    At Cpropofol = 10 µg/mL and Cremifentanil = 4 ng/mL:
    - interaction model HR change: 1.9%
    - combination-null model HR change: 24.3%
    """
    assert round(effects(10.0, 4.0)["full"]["hr"], 1) == 1.9
    assert round(effects(10.0, 4.0)["null"]["hr"], 1) == 24.3


def test_sv_interaction_vs_null_model():
    """
    At Cpropofol = 10 µg/mL and Cremifentanil = 4 ng/mL:
    - interaction model SV change: -32.5%
    - combination-null model SV change: -7.2%
    """
    assert round(effects(10.0, 4.0)["full"]["sv"], 1) == -32.5
    assert round(effects(10.0, 4.0)["null"]["sv"], 1) == -7.2


def test_map_deviation_is_maximal():
    """
    The deviation in MAP effect between propofol alone and propofol combined
    with remifentanil should be maximal at Cpropofol = 1.1 µg/mL.
    """
    cps = np.array([0.1, 1.1, 10.0])

    for cp_remi in [2.0, 4.0]:
        deviations = []

        for cp_prop in cps:
            prop_map = effects(float(cp_prop), 0.0)["prop"]["map"]
            full_map = effects(float(cp_prop), cp_remi)["full"]["map"]
            deviations.append(abs(full_map - prop_map))

        cp_max = cps[np.argmax(deviations)]

        assert cp_max == 1.1


def test_base_values():
    patient = Patient(
        age=35,
        height=170,
        weight=70,
        sex="male",
        opiates=False,
        base_sv=82.2,
        base_hr=56.0,
        base_tpr=0.016,
    )

    model = SuHaemoPD(
        patient=patient,
        pk_propofol=PropofolPK(patient),
        pd_propofol=PropofolPD(patient),
        pk_remifentanil=RemifentanilPK(patient),
        use_bsv=False,
    )

    assert model.base_tpr == patient.base_tpr
    assert model.base_sv == patient.base_sv
    assert model.base_hr == patient.base_hr