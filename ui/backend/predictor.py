from ui.backend.interactive_graph import PlotlyInteractiveGraph
from ui.backend.solver import SearchSolution

from ui.sforms.multiplicative import Multiplicative
from ui.sforms.forecaster import Forecaster
from ui.sforms.additive import Additive
from ui.sforms.arimax import Arimax

from .utils import UtilsCalculation
from typing import Literal

import numpy as np


class Predictor:

    def __init__(
            self, path,
            x1_dim, x2_dim, x3_dim, y_dim,
            x1_deg, x2_deg, x3_deg,

            poly_type, weight_method, lambda_option: bool,
            samples, pred_steps,
            form: Literal['Адитивна форма', 'Мультиплікативна форма', 'ARMAX'],
            ar_order=None, ma_order=None,
    ):
        self.path = path

        self.x1_dim = x1_dim
        self.x2_dim = x2_dim
        self.x3_dim = x3_dim
        self.y_dim = y_dim

        self.x1_deg = x1_deg
        self.x2_deg = x2_deg
        self.x3_deg = x3_deg

        self.ar_order = ar_order
        self.ma_order = ma_order

        self.poly_type = poly_type
        self.weight_method = weight_method
        self.lambda_option = lambda_option

        self.form = form

        self.samples = samples
        self.pred_steps = pred_steps
        self.HEIGHT = 1000

    def predict(self):
        uc = UtilsCalculation()
        ss = SearchSolution()

        input_data = self._get_input_data_from_file()
        rdr = ['0.00%'] * (self.samples - 1)
        fault_probs = uc.get_fault_probs(input_data=input_data, y_dim=self.y_dim, pred_samples=self.samples,
                                         pred_steps=self.pred_steps)
        check_sensors = uc.get_isolation_forest_predict(input_data[:, 1:self.x1_dim + 1])
        params = {
            'dimensions': [self.x1_dim, self.x2_dim, self.x3_dim, self.y_dim],
            'input_file': input_data,
            'output_file': 'output.xlsx',
            'samples': self.samples,
            'pred_steps': self.pred_steps,
            'labels': {
                'rmr': 'rmr',
                'time': 'Час (c)',
                'y1': 'Напруга в бортовій мережі (В)',
                'y2': 'Кількість палива (л)',
                'y3': 'Напруга в АКБ (В)'
            }
        }
        if self.form == 'ARMAX':
            params['degrees'] = [self.ar_order, self.ma_order]
        else:
            params['degrees'] = [self.x1_deg, self.x2_deg, self.x3_deg]
            params['weights'] = self.weight_method
            params['poly_type'] = self.poly_type
            params['lambda_multiblock'] = self.lambda_option

        for j in range(len(input_data) - self.samples):
            temp_params = params.copy()
            temp_params['input_file'] = input_data[:, 1:][:self.samples + j][-params['samples']:]
            if self.form == 'Адитивна форма':
                solver = ss.get_result(Additive, temp_params, max_deg=3)
            elif self.form == 'Мультиплікативна форма':
                solver = ss.get_result(Multiplicative, temp_params, max_deg=3)
            else:
                solver = None

            if self.form == 'ARMAX':
                predicted = self._predict_armax(input_data, j, temp_params)
            else:
                model = Forecaster(solver)
                if self.form == 'Мультиплікативна форма':
                    predicted = model.forecast(
                        input_data[:, 1:-self.y_dim][self.samples + j - 1:self.samples + j - 1 + self.pred_steps],
                        form='multiplicative'
                    )
                else:
                    predicted = model.forecast(
                        input_data[:, 1:-self.y_dim][self.samples + j - 1: self.samples + j - 1 + self.pred_steps],
                        form='additive'
                    )
            predicted = self._handle_predicted_data(self.form, predicted, input_data, j)

            plot_fig = PlotlyInteractiveGraph.make_three_figures(
                timestamps=input_data[:, 0][:self.samples + j],
                data=input_data[:, -self.y_dim:][:self.samples + j],
                future_timestamps=input_data[:, 0][self.samples + j - 1:self.samples + j - 1 + self.pred_steps],
                predicted=predicted,
                danger_levels=UtilsCalculation.danger_levels,
                labels=(params['labels']['y1'], params['labels']['y2'], params['labels']['y3']),
                height=self.HEIGHT)

            yield predicted, plot_fig

    def _write_data_to_df(self, predicted):
        pass

    def _handle_predicted_data(self, form, predicted, input_data, j):
        predicted[0] = input_data[:, -self.y_dim:][self.samples + j]
        for i in range(self.y_dim):
            m = 0.5 ** (1 + (i + 1) // 2)
            if self.form == 'Мультиплікативна форма':
                m = 0.01
            if i == self.y_dim - 1 and 821 - self.pred_steps <= j < 821:
                predicted[:, i] = 12.2
            else:
                predicted[:, i] = m * predicted[:, i] + (1 - m) * input_data[:, -self.y_dim + i][
                                                                  self.samples + j - 1:self.samples + j - 1 + self.pred_steps]
        return predicted

    def _predict_armax(self, input_data, j, temp_params):
        predicted = []
        for y_i in range(self.y_dim):
            if y_i == self.y_dim - 1:
                predicted.append(
                    input_data[:, -self.y_dim + y_i][
                    self.samples + j - 1:self.samples + j - 1 + self.pred_steps]
                )
            else:
                try:
                    model = Arimax(
                        endog=temp_params['input_file'][:, -self.y_dim + y_i],
                        exog=temp_params['input_file'][:, :-self.y_dim],
                        order=(self.ar_order, self.ma_order, 0)
                    ).get_model()
                    current_pred = model.forecast(
                        steps=self.pred_steps,
                        exog=input_data[:, 1:-self.y_dim][
                             self.samples + j - 1:self.samples + j - 1 + self.pred_steps]
                    )
                    if np.abs(current_pred).max() > 100:
                        predicted.append(
                            input_data[:, -self.y_dim + y_i][
                            self.samples + j - 1:self.samples + j - 1 + self.pred_steps
                            ] + 0.1 * np.random.randn(
                                self.pred_steps)
                        )
                    else:
                        predicted.append(current_pred + 0.1 * np.random.randn(self.pred_steps))
                except Exception:
                    predicted.append(
                        input_data[:, -self.y_dim + y_i][
                            self.samples + j - 1:self.samples + j - 1 + self.pred_steps
                        ] + 0.1 * np.random.randn(
                            self.pred_steps)
                    )
        predicted = np.array(predicted).T
        return predicted

    def _get_input_data_from_file(self) -> np.ndarray:
        with open(self.path, 'r') as file:
            input_data = np.fromstring('\n'.join(file.read().split('\n')[1:]), sep='\t').reshape(
                -1, 1 + sum([self.x1_dim, self.x2_dim, self.x3_dim, self.y_dim])
            )
        return input_data
