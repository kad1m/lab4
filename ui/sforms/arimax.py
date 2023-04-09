from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import pacf
from ui.backend.utils import UtilsCalculation

import numpy as np


class Arimax:

    def __init__(self, endog, order=None, exog=None):
        self.endog = endog
        self.order = order
        self.exog = exog

    def get_model(self):
        if self.order is None:
            self.order = (0, 0, 0)
        if len(self.order) == 3:
            p, q, d = self.order
        elif len(self.order) == 2:
            p, q = self.order
            d = 0
        else:
            p, q, d = 0, 0, 0
        if p == 0:
            pacf_tolerance = 1.96 / np.sqrt(len(self.endog))
            try:
                p = np.where(abs(pacf(self.endog, nlags=10)) >= pacf_tolerance)[0].max() + 1
            except Exception:
                p = 0

        if q == 0:
            ma = UtilsCalculation.ewma(self.endog)
            pacf_tolerance = 1.96 / np.sqrt(len(self.endog))
            try:
                q = np.where(abs(pacf(ma, nlags=10)) >= pacf_tolerance)[0].max() + 1
            except Exception:
                q = 0

        model = ARIMA(self.endog, self.exog, (p, q, d))
        model = model.fit()
        return model
