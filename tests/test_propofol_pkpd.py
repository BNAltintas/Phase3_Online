import numpy as np

from propofol.propofol_pkpd import EleveldPatient, EleveldPD, EleveldPK

# Reproducibility
np.random.seed(0)
n_samples = 10000

patient =  EleveldPatient(age=35, height=170, weight=70, sex='male', opiates=False)

def test_eleveldpk_draw_eta():

    pk = EleveldPK(patient, use_bsv=False)
    etas_list = []

    omega = [
            0.610,
            0.565,
            0.597,
            0.265,
            0.346,
            0.209,
            0.463,
        ]

    for _ in range(n_samples):
        pk.draw_eta()
        etas_list.append([pk.eta1, pk.eta2, pk.eta3, pk.eta4,
                          pk.eta5, pk.eta6, pk.eta7])
    etas_list = np.array(etas_list)

    print(np.std(etas_list, axis=0))
    assert np.allclose(np.std(etas_list, axis=0),
                       omega,
                       rtol = 5e-2, atol=3e-2)

    assert np.allclose(np.mean(etas_list, axis=0),
                       np.zeros(len(omega)),
                       rtol=5e-2, atol=3e-2)



def test_eleveldpd_draw_eta():
    pd = EleveldPD(patient, use_bsv=False)
    etas_list = []

    omega = [
            0.242,
            0.702,
            0.230,
        ]

    for _ in range(n_samples):
        pd.draw_eta()
        etas_list.append([pd.eta1, pd.eta2, pd.eta3])
    etas_list = np.array(etas_list)

    print(np.std(etas_list, axis=0))
    assert np.allclose(np.std(etas_list, axis=0),
                       omega, rtol = 5e-2, atol=3e-2)

    assert np.allclose(np.mean(etas_list, axis=0),
                       np.zeros(len(omega)), rtol=5e-2, atol=3e-2)
