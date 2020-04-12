import numpy as np

from it import IT
from individual import Individual

class Evol_operators:
    def __init__(self, expolim=10, min_nof_terms=1, max_nof_terms=10):
        self.expolim = expolim
        self.min_nof_terms = min_nof_terms
        self.max_nof_terms = max_nof_terms

    
    def _mut_drop(self, it, func_set):
        it.remove_term(np.random.choice(it.len))

        return it

    
    def _mut_add(self, it, func_set):
        
        expolim = self.expolim

        #Faz a criação dentro dos limites de expolim
        term = np.random.randint(-expolim, expolim + 1, size=len(it.terms[0])).tolist()        
        it.add_term(term, func_set[np.random.choice(len(func_set))])
            
        return it

    
    def _mut_term(self, it, func_set):

        expolim = self.expolim
        
        chosen_term = np.random.choice(it.len)

        newt, newf = it.get_term(chosen_term)
        newt[np.random.choice(len(newt))] = np.random.randint(-expolim, expolim + 1)

        it.remove_term(chosen_term)
        it.add_term(newt, newf)
        
        return it

    
    def _mut_func(self, it, func_set):
        chosen_term = np.random.choice(it.len)

        newt, newf = it.get_term(chosen_term)

        it.remove_term(chosen_term)
        it.add_term(newt, func_set[np.random.choice(len(func_set))])
        
        return it

    
    def _mut_interp(self, it, func_set):
        term1_index = np.random.choice(it.len)
        term2_index = np.random.choice(it.len)

        newt, newf = it.get_term(term1_index)
        for i, exp in enumerate(it.terms[term2_index]):
            newt[i] = (newt[i]+exp)%self.expolim * (-1 if newt[i]+exp>=0 else 1)
            
        it.add_term(newt, newf)
            
        return it

    
    def _mut_intern(self, it, func_set):
        term1_index = np.random.choice(it.len)
        term2_index = np.random.choice(it.len)

        newt, newf = it.get_term(term1_index)
        for i, exp in enumerate(it.terms[term2_index]):
            newt[i] = (newt[i]-exp)%self.expolim * (-1 if newt[i]-exp>=0 else 1)
            
        it.add_term(newt, newf)
            
        return it

    
    def mutation(self, args):

        func_set, fit_fun, label, sol = args
        it = sol.it.copy()
        
        mutations = {
            'term' : self._mut_term,
            'func' : self._mut_func
        }

        if it.len>self.min_nof_terms:
            mutations.update({'drop' : self._mut_drop})

        if it.len<self.max_nof_terms:
            mutations.update({
                'add'    : self._mut_add,
                'intern' : self._mut_intern,
                'interp' : self._mut_interp,
            })

        mutation = np.random.choice(list(mutations))

        it = mutations[mutation](it, func_set)
        if it.len == 0:
            return sol.copy()
        else:
            ind = Individual(fit_fun, func_set, self.expolim, self.max_nof_terms, it, label)
            return ind

    
    def crossover(self, func_set, fit_fun, label, it1, it2, rate):

        t1, f1 = it1.it.terms, it1.it.funcs
        t2, f2 = it1.it.terms, it1.it.funcs
        
        picked_t = []
        picked_f = []

        for t, f in zip(t1 + t2, f1+f2):
            if np.random.rand() < rate:
                picked_t.append(t.copy())
                picked_f.append(f)

        it = IT(picked_t, picked_f, label) if len(picked_t)!= 0 else None

        ind = Individual(fit_fun, func_set, self.expolim, self.max_nof_terms, it, label)

        return ind
