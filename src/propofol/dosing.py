import logging
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid

from propofol.protocols import Patient

logger = logging.getLogger(__name__)


class Dosing():
    """Dosing strategy and rate class."""
    def __init__(
        self,
        patient: Patient,
        strategy: Optional[dict[str, Any] | str] = None,
        cumulative_dose: Optional[pd.DataFrame] = None,
        correct_baseline: bool = True,
    ) -> None:
        """Initialize a dosing model from a strategy or cumulative-dose data.

        Parameters
        ----------
        patient : Patient
            Patient descriptor used for weight-based dosing.
        strategy : dict or str or None, optional
            Dosing strategy definition. If set to ``"default"``, a predefined
            strategy is used.
        cumulative_dose : pandas.DataFrame, optional
            DataFrame containing ``time`` and ``cumulative_dose`` columns used to
            derive dosing rates.
        correct_baseline : bool, default=True
            If ``True``, reset positive baseline cumulative dose to zero.

        Raises
        ------
        ValueError
            If both or neither of ``strategy`` and ``cumulative_dose`` are provided.
        """
        self.patient = patient
        if (strategy is None) == (cumulative_dose is None):
            raise ValueError("Exactly one of 'strategy' or 'cumulative_dose' must be provided.")

        if strategy == 'default':
            strategy = {
                'induction_dose': 2.5,  # mg / kg
                'titration_dose': 40,  # mg/10s (defaulgt 40 mg/10s = 240 mg/min)
                'maintenance':{
                1: {'duration': 15,  # min
                    'dose': 12, # mg / kg / h

                },
                2: {'duration': 30, # min
                    'dose': 7.2, # mg / kg / h
                }
                }}

        self.strategy = strategy
        if strategy is not None:
            self.df_cumdose = None
        else:
            self.df_cumdose = cumulative_dose
            self.correct_baseline = correct_baseline
        self.dotA0, self.tcrit = self.get_dosing_rate_func()


    def _rule(self, duration, dose):
        return (lambda t, m=duration: t <= m,  # min
            lambda t, d=dose: d)  # mg/min

    def _build_rules_from_cumulative_dose_dataframe(self, df, correct_baseline: bool =True):
        df = df.copy()
        baseline = df.iloc[0]['cumulative_dose']
        df.reset_index(drop=True, inplace=True)
        assert df.equals(df.sort_values(by="time")), "DataFrame is not sorted by time"
        assert df.index.equals(pd.RangeIndex(start=0, stop=len(df))), "Index is not 0..n-1"
        assert (np.diff(df['cumulative_dose']) >= 0).all(),\
            "cumulative dose not monototically increasing"


        if baseline > 0:
            logger.info(f"Baseline A0 = {baseline} mg, setting to 0")
            if correct_baseline:
                logger.info("Setting baseline A0 to 0")
                df.loc[0, 'cumulative_dose'] = 0
            else:
                logger.warning("Not correcting baseline A0")

        # create rules
        rules = []
        tcrit = []
        tprev = None
        y_prev = None
        for ix, row in df.iterrows():
            t = row['time']
            y = row['cumulative_dose']
            if ix == 0:
                rules.append(self._rule(t, 0))
            else:
                dy = y - y_prev  # mg
                dt = t - tprev  # min
                rules.append(self._rule(t, dy / dt))
            tcrit.append(t)
            tprev = t
            y_prev = y
        return rules, tcrit

    def _build_rules_from_strategy(self, strategy):
        rules = []
        tcurrent = 0
        tcrit = []

        # t < 0:
        rules.append(self._rule(0, 0))
        tcrit.append(tcurrent)

        # Induction (at t = 0)
        induction_dose_total = self.patient.weight * strategy['induction_dose']  # mg

        if strategy['titration_dose'] is None:  # bolus
            induction_duration = 10 / 60.  # min (default is 10 sec for bolus)
        else:
            induction_duration = (induction_dose_total
                                  / (strategy['titration_dose'])
                                  * 10. / 60.)  # 10s per min min

        rules.append(self._rule(induction_duration,
                           induction_dose_total / induction_duration)  # mg / min
                           )

        tcurrent += induction_duration
        tcrit.append(tcurrent)


        # Maintenance (t > induction duration)
        # sort by duration threshold
        items = sorted(
            (v['duration'], v['dose']) for v in strategy['maintenance'].values()
        )

        for duration, dose in items:
            # note: have to use defaults here or it will use the last value from the loop
            rules.append(self._rule(duration + tcurrent,
                               dose * self.patient.weight / 60  # mg / min
                               ))
            tcurrent += duration
            tcrit.append(tcurrent)
        return rules, tcrit


    def get_dosing_rate_func(self) -> tuple[Callable[[float], float], list[float]]:
        """Build a piecewise dosing-rate function and critical time points.

        Returns
        -------
        tuple
            Tuple ``(dotA0, tcrit)`` where ``dotA0`` maps time in minutes to dosing
            rate in mg/min and ``tcrit`` contains rule transition times.
        """

        if self.strategy is not None:
            rules, tcrit = self._build_rules_from_strategy(self.strategy)
        else:
            rules, tcrit = self._build_rules_from_cumulative_dose_dataframe(self.df_cumdose,
                                                                            self.correct_baseline)

        def func(x: float) -> float:
            """Evaluate dosing rate at time ``x`` in minutes."""
            for condition, action in rules:
                if condition(x):
                    return action(x)
            return 0  # default

        return func, tcrit

    def get_cumulative_dose_from_strategy(
        self,
        tmax: float = 60,
        freq: str = 'm',
    ) -> pd.DataFrame:
        """Integrate dosing-rate function to cumulative dose over time.

        Parameters
        ----------
        tmax : float, default=60
            Maximum simulation time in minutes.
        freq : str, default='m'
            Output sampling frequency. Only ``'m'`` (1 minute) is supported.

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns ``time`` and ``cumulative_dose``.

        Raises
        ------
        NotImplementedError
            If an unsupported output frequency is requested.
        """
        t = np.linspace(0, tmax, tmax * 6000 + 1)  # dt is 0.01 seconds
        if freq == 'm':
            tinterp = np.linspace(0, tmax, tmax + 1)
        else:
            raise NotImplementedError("Other frequencies than 1minute are not implemented.")
        dy = [self.dotA0(x) for x in t]
        cumy = cumulative_trapezoid(dy, t, initial=dy[0])
        return pd.DataFrame(data = {'time': tinterp,
                                    'cumulative_dose': np.interp(tinterp, t, cumy)})


    def __call__(self, t: float) -> float:
        """Return infusion rate at time ``t``.

        Parameters
        ----------
        t : float
            Time in minutes.

        Returns
        -------
        float
            Infusion rate in mg/min.
        """
        return self.dotA0(t)
