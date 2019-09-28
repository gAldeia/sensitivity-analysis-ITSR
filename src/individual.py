import numpy as np

from it import IT


class Individual:

    def __init__(self, fit_fun, func_set, it = None, label = []):
    
        self.fit_fun = fit_fun        
        self.label = label
        self.func_set = func_set
        self.n_vars = fit_fun.n_vars

        self.it = self._build_random_it() if it is None else it

        self.fitness, self.it.coeffs, self.it.intercept = self.fit_fun.fitness(self.it)

    def _build_random_it(self):

        if len(self.label) == 0:
            self.label = ['x' + str(i) for i in range(self.n_vars)]
        
        n_terms = np.random.randint(1, 4)
        terms = [np.random.randint(0, 4, size=(self.n_vars)).tolist() 
                for _ in range(n_terms)]   
        funcs = [self.func_set[i] 
                for i in np.random.choice(len(self.func_set), size=n_terms)]

        it = IT(terms, funcs, self.label)
      
        return it