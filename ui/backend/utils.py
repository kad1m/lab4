from sklearn.ensemble import IsolationForest

import pandas as pd
import numpy as np


class UtilsCalculation:
    RANDOM_STATE = 42

    danger_levels = [
        (11.7, 10.5),
        (1.0, 0.0),
        (11.7, 10.5)
    ]

    TIME_DELTA = 10

    def get_fault_probs(self, input_data, y_dim, pred_samples, pred_steps):
        fault_probs = []
        for i in range(y_dim):
            fault_probs.append(
                self.moving_average_prediction(
                    input_data[:, -y_dim + i],
                    y_emergency=self.danger_levels[i][0],
                    y_fatal=self.danger_levels[i][1],
                    window_size=pred_samples // pred_steps
                )
            )
        return np.array(fault_probs).T

    # FaultProb
    def moving_average_prediction(self, y, y_emergency, y_fatal, window_size):
        y_ma = pd.DataFrame(y).rolling(window_size).mean().values.flatten()
        result = (y_emergency - y_ma) / (y_emergency - y_fatal)
        result[result > 1] = 1
        result[result < 0] = 0
        return result

    def ClassifyEmergency(self, y1, y2, y3):
        res = []
        if self.danger_levels[0][1] < y1 <= self.danger_levels[0][0]:
            res.append('низька напруга в бортовій мережі')
        elif y1 <= self.danger_levels[0][1]:
            res.append('критично низька напруга в бортовій мережі')
        if self.danger_levels[1][1] < y2 <= self.danger_levels[1][0]:
            res.append('мала кількість пального')
        elif y2 <= self.danger_levels[1][1]:
            res.append('немає пального')
        if self.danger_levels[2][1] < y3 <= self.danger_levels[2][0]:
            res.append('низька напруга в АКБ')
        elif y3 <= self.danger_levels[2][1]:
            res.append('критично низька напруга в АКБ')

        if len(res) > 0:
            return ', '.join(res)
        else:
            return '-' + ' ' * 40

    def classify_state(self, y1, y2, y3):
        if (
                self.danger_levels[0][1] < y1 <= self.danger_levels[0][0] or
                self.danger_levels[1][1] < y2 <= self.danger_levels[1][0] or
                self.danger_levels[2][1] < y3 <= self.danger_levels[2][0]):
            return 'Нештатна ситуація'
        elif (
                y1 <= self.danger_levels[0][1] or
                y2 <= self.danger_levels[1][1] or
                y3 <= self.danger_levels[2][1]):
            return 'Аварійна ситуація'
        else:
            return 'Нормальний стан'

    def acceptable_risk(self, y_slice, danger_levels):
        deltas = np.diff(y_slice, axis=0).max(axis=0)
        y_to_danger = np.array([
            y_slice[-1][i] - danger_levels[i][1]
            for i in range(len(danger_levels))
        ])
        return max((y_to_danger / deltas).min(), 0)

    # CheckSensors
    def get_isolation_forest_predict(self, x):
        all_pred = []
        for i in range(x.shape[1]):
            clf = IsolationForest(
                random_state=self.RANDOM_STATE
            ).fit(x[:, i].reshape(-1, 1))
            pred = clf.predict(x[:, i].reshape(-1, 1))
            all_pred.append(pred)
        all_pred = np.array(all_pred)
        return 1 * (all_pred.sum(axis=0) == -2)

    def highlight(self, s, column, vals, colors):
        for val, color in zip(vals, colors):
            if s[column] == val:
                return [f'background-color: {color}'] * len(s)
        else:
            return [f'background-color: white'] * len(s)

    @staticmethod
    def ewma(x, alpha=None):
        if alpha is None:
            alpha = 2 / (len(x) + 1)
        x = np.array(x)
        n = x.size

        w0 = np.ones(shape=(n, n)) * (1 - alpha)
        p = np.vstack([np.arange(i, i - n, -1) for i in range(n)])

        w = np.tril(w0 ** p, 0)

        return np.dot(w, x[::np.newaxis]) / w.sum(axis=1)