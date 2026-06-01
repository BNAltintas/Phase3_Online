from functools import lru_cache

import numpy as np

from propofol.haemo_pd_su2023 import SuHaemoPD
from propofol.patient import EleveldPatient as Patient
from propofol.propofol_pkpd import EleveldPD as PropofolPD
from propofol.propofol_pkpd import EleveldPK as PropofolPK
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
