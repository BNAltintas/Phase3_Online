
import numpy as np

from propofol.haemo_pd import SuHaemoPD
from propofol.patient import EleveldPatient
from propofol.propofol_pkpd import EleveldPD, EleveldPK


def test_figure2():
    """Compare output of the code against Fig. 2 of the paper and values on p811."""

    t = np.linspace(0, 180, 180*60+1)  # needs to be longer than one hour to properly converge
    patient = EleveldPatient(35, 70, 170, 'male', opiates=True)
    pd_propofol = EleveldPD(patient)
    pk_propofol = EleveldPK(patient)

    # Set all pk parameters to 0
    pk_propofol.k10 = 0
    pk_propofol.k12 = 0
    pk_propofol.k13 = 0
    pk_propofol.k21 = 0
    pk_propofol.k31 = 0
    pk_propofol.ke0 = 0

    pd_haemo = SuHaemoPD(patient=patient, pk_propofol= pk_propofol, pd_propofol=pd_propofol)

    Cps = [0.1, 1.1, 10.]
    hrs = []
    svs = []
    MAPs = []
    for Cp in Cps:
        y0 = [Cp * pd_haemo.pk_propofol.V1,  # fix A1 for constant Cp
            0,
            0,
            0,
            pd_haemo.base_sv, # base sv
            pd_haemo.base_hr,  # base hr
            pd_haemo.base_tpr, # base tpr
        pd_haemo.Theta12 * pd_haemo.base_hr]

        A1, A2, A3, Ce, sv, hr, MAP, tde = pd_haemo.solve_ode(t=t,
                                                        y0 = y0,
                                                        dosing=None)
        hrs.append(hr[-1])
        svs.append(sv[-1])
        MAPs.append(MAP[-1])

    assert MAPs[0].round(0) == 86  # Cp = 0.1
    assert MAPs[-1].round(0) == 54  # Cp = 10
    assert hrs[0].round(0) == 56  # Cp = 0.1
    assert hrs[-1].round(0) == 87  # Cp = 10
    assert svs[0].round(1) == 82.9  # Cp = 0.1
    assert svs[1].round(1) == 75.6  # Cp = 0.1
    assert svs[-1].round(1) == 91.2  # Cp = 10

def test_base_value():
    patient = EleveldPatient(35, 70, 170, 'male', opiates=True,
                             base_hr=4, base_map=6, base_pp=2)
    pd_propofol = EleveldPD(patient)
    pk_propofol = EleveldPK(patient)

    pd_haemo = SuHaemoPD(patient=patient, pk_propofol= pk_propofol, pd_propofol=pd_propofol)

    assert pd_haemo.base_tpr == patient.base_tpr
    assert pd_haemo.base_sv  == patient.base_sv
    assert pd_haemo.base_hr  == patient.base_hr
