

import numpy as np

from propofol.dosing import Dosing


class Patient:
    def __init__(self, weight):
        self.weight = weight

def test_strategy():
    weight = 100

    patient = Patient(weight)

    strategy_titration = {
            'induction_dose': 2.5,  # mg / kg
            'titration_dose': 40,  # mg/10s (defaulgt 40 mg/10s = 240 mg/s)s
            'maintenance':{
            1: {'duration': 15,  # min
                'dose': 12, # mg / kg / h

            },
            2: {'duration': 30, # min
                'dose': 7.2, # mg / kg / h
            }
            }}


    strategy_bolus = {
            'induction_dose': 2.5,  # mg / kg
            'titration_dose': None,
            'maintenance':{
            1: {'duration': 15,  # min
                'dose': 12, # mg / kg / h

            },
            2: {'duration': 30, # min
                'dose': 7.2, # mg / kg / h
            }
            }}

    d_titration = Dosing(patient=patient, strategy=strategy_titration)
    d_bolus = Dosing(patient=patient, strategy=strategy_bolus)

    # Note: we need a resolution of 0.01 second to get this correct for the induction part.
    # For 75kg the resolution even has to be 0.001 second.
    t = np.linspace(-60, 60, 120 * 600 + 1)
    dotA0_titration = [d_titration(x) for x in t]
    total_titration = np.trapezoid(dotA0_titration, t).round(1)
    dotA0_bolus = [d_bolus(x) for x in t]
    total_bolus = np.trapezoid(dotA0_bolus, t).round(1)

    # Check that total mg are the same
    total_dose = round((2.5
                        + 12 * 15 / 60
                        + 7.2 * 30 / 60)
                        * weight, 1)
    assert total_bolus == total_dose
    assert total_titration == total_dose

    # Check cumulative dose computed from strategy
    df = d_titration.get_cumulative_dose_from_strategy()
    assert df['cumulative_dose'].iloc[0].round(1) == 0
    assert df['cumulative_dose'].iloc[-1].round(1) == total_titration
    assert (np.diff(df['cumulative_dose']) >= 0).all()

def test_cumulative_dose():
    # Compare to strategy implementation
    patient = Patient(100)

    d1 = Dosing(patient=patient, strategy='default')
    df_dose = d1.get_cumulative_dose_from_strategy()
    d2 = Dosing(patient=patient,
                cumulative_dose=df_dose)

    # Set t
    # Note: need high enough resolution for strategy (else trapezoid misses changes in rate)
    t = np.linspace(0, 120, 120*600+1)  # 0.1 second resolution when integrating

    integral = np.trapezoid([d2(x) for x in t], t)
    assert np.trapezoid([d1(x) for x in t], t).round(1) == integral.round(1)
    assert integral.round(1) == d2.df_cumdose.iloc[-1]['cumulative_dose'].round(1)

    # Check dose with and without baseline correction
    baseline = 100
    df_baseline = df_dose.copy()
    df_baseline['cumulative_dose'] += baseline
    d3 = Dosing(patient=patient,
            cumulative_dose=df_baseline,
            correct_baseline=False)
    d4 = Dosing(patient=patient,
            cumulative_dose=df_baseline,
            correct_baseline=True)
    integral_baseline_uncorrected = np.trapezoid([d3(x) for x in t], t)
    integral_baseline_corrected = np.trapezoid([d4(x) for x in t], t)
    assert integral_baseline_uncorrected == integral  # integral ignores baseline
    assert integral_baseline_corrected == integral + baseline


