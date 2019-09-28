import numpy as np
from sklearn.linear_model import LinearRegression 

from it import IT


class Fitness:

    def __init__(self, X, y, model=LinearRegression):
        self.X = X
        self.y = y
        self._model = model()
        self.n_vars = X.shape[1]

    def fitness(self, it):
        Z = it._eval(self.X)

        if np.isinf(Z).any() or np.isnan(Z).any() or np.any(Z > 1e+300) or np.any(Z < -1e+300):
            #return 1e+300, np.ones(self.X.shape[1]), 0.0
            return 1e+300, np.ones(it.len), 0.0 #notificar pro fabrício!
        
        self._model.fit(Z, self.y)
        y_hat = self._model.predict(Z)
        err = np.square(y_hat - self.y)

        return np.sqrt(err.mean()), self._model.coef_.tolist(), self._model.intercept_
