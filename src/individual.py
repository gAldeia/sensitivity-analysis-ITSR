import numpy as np

from it import IT


class Individual:

    def __init__(self, fit_fun, func_set, expolim, max_terms, it = None, label = []):
    
        self.fit_fun   = fit_fun        
        self.label     = label
        self.func_set  = func_set
        self.n_vars    = fit_fun.n_vars
        self.expolim   = expolim
        self.max_terms = max_terms

        self.it = self._build_random_it() if it is None else it

        self.fitness, self.it.coeffs, self.it.intercept = self.fit_fun.fitness(self.it)

    def _build_random_it(self):

        if len(self.label) == 0:
            self.label = ['x' + str(i) for i in range(self.n_vars)]
        
        n_terms = np.random.randint(1, self.max_terms + 1)

        terms = [[np.random.randint(-self.expolim, self.expolim + 1) for _ in range(self.n_vars)] 
                for _ in range(n_terms)]   
        funcs = [self.func_set[i] 
                for i in np.random.choice(len(self.func_set), size=n_terms)]

        it = IT(terms, funcs, self.label)
      
        return it