import numpy as np


class Forecaster:

    def __init__(self, solver):
        self.solver = solver
        self.lamb = self._fill_lamb_list()
        self.a = solver.a.T.tolist()
        self.c = solver.c.T.tolist()
        self.X_min = self.solver.datas[:, :-self.solver.Y.shape[1]].min(axis=0)
        self.X_max = self.solver.datas[:, :-self.solver.Y.shape[1]].max(axis=0)
        self.Y_min, self.Y_max = solver.Y_.min(axis=0), solver.Y_.max(axis=0)

    def forecast(self, X, form='additive'):
        Y = np.zeros((len(X), self.solver.Y.shape[1]))
        n, m = X.shape
        vec = np.zeros_like(X)
        for j in range(m):
            minv = np.min(X[:, j])
            maxv = np.max(X[:, j])
            for i in range(n):
                if np.allclose(maxv - minv, 0):
                    vec[i, j] = X[i, j] is X[i, j] != 0
                else:
                    vec[i, j] = (X[i, j] - minv) / (maxv - minv)
        X_norm = np.array(vec)

        res = np.array([self._evalF(form, X_norm, i) for i in range(Y.shape[1])]).T
        return res * (self.Y_max - self.Y_min) + self.Y_min

    def _fill_lamb_list(self):
        lamb = []
        for i in range(self.solver.Y.shape[1]):  # `i` is an index for Y
            lamb_i = list()
            shift = 0
            for j in range(3):  # `j` is an index to choose vector from X
                lamb_i_j = list()
                for k in range(self.solver.dim[j]):  # `k` is an index for vector component
                    lamb_i_j_k = self.solver.Lamb[shift:shift + self.solver.deg[j], i].ravel()
                    shift += self.solver.deg[j]
                    lamb_i_j.append(lamb_i_j_k)
                lamb_i.append(lamb_i_j)
            lamb.append(lamb_i)
        return lamb

    def _evalF(self, form: str, x, i):
        res = None
        if form == 'additive':
            res = 0
        elif form == 'multiplicative':
            res = 1
        for j in range(3):
            for k in range(len(self.lamb[i][j])):
                shift = sum(self.solver.dim[:j]) + k
                for n in range(len(self.lamb[i][j][k])):
                    coef = self.c[i][j] * self.a[i][shift] * self.lamb[i][j][k][n]
                    if form == 'additive':
                        res += coef * self.solver.poly_f(n, x[:, shift])
                    elif form == 'multiplicative':
                        res += (1 + self.solver.poly_f(n, x[:, shift]) + 1e-8) ** coef
        if form == 'additive':
            return res
        elif form == 'multiplicative':
            return res - 1