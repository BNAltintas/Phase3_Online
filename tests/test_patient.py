
from propofol.patient import EleveldPatient


def test_base_values():
    patient1 = EleveldPatient(35, 70, 170, 'male', opiates=True)
    patient2 = EleveldPatient(35, 70, 170, 'male', opiates=True,
                              base_hr=4, base_map=6, base_pp=2)
    patient3 = EleveldPatient(35, 70, 170, 'male', opiates=True,
                              base_hr = None, base_map=6, base_pp=2)
    patient4 = EleveldPatient(35, 70, 170, 'male', opiates=True,
                              base_sap=130, base_dap=80)

    assert patient1.base_tpr is None
    assert patient1.base_sv is None
    assert patient2.base_sv == 3
    assert patient2.base_tpr == 0.5  # 6 / (4 * 3)
    assert patient3.base_tpr is None
    assert patient4.base_pp == 50
    assert patient4.base_map == (130 + 2 * 80) / 3.0
    assert patient4.base_sv == 75




