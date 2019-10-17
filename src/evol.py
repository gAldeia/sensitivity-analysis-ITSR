import numpy as np

from it import IT
from individual import Individual

#Evol_operators é uma classe estática apenas para agrupar os métodos
class Evol_operators:
    expolim = 10
    min_nof_terms = 1
    max_nof_terms = 10

    @staticmethod
    def _mut_drop(it, func_set):
        it.remove_term(np.random.choice(it.len))

        return it

    @staticmethod
    def _mut_add(it, func_set):
        
        expolim = Evol_operators.expolim

        #Faz a criação dentro dos limites de expolim
        term = np.random.randint(-expolim, expolim + 1, size=len(it.terms[0])).tolist()        
        it.add_term(term, func_set[np.random.choice(len(func_set))])
            
        return it

    @staticmethod
    def _mut_term(it, func_set):

        expolim = Evol_operators.expolim
        
        chosen_term = np.random.choice(it.len)

        newt, newf = it.get_term(chosen_term)
        newt[np.random.choice(len(newt))] = np.random.randint(-expolim, expolim + 1)

        it.remove_term(chosen_term)
        it.add_term(newt, newf)
        
        return it

    @staticmethod
    def _mut_func(it, func_set):
        chosen_term = np.random.choice(it.len)

        newt, newf = it.get_term(chosen_term)

        it.remove_term(chosen_term)
        it.add_term(newt, func_set[np.random.choice(len(func_set))])
        
        return it

    @staticmethod
    def _mut_interp(it, func_set):
        term1_index = np.random.choice(it.len)
        term2_index = np.random.choice(it.len)

        newt, newf = it.get_term(term1_index)
        for i, exp in enumerate(it.terms[term2_index]):
            newt[i] = (newt[i]+exp)%Evol_operators.expolim * (-1 if newt[i]+exp>=0 else 1)
            
        it.add_term(newt, newf)
            
        return it

    @staticmethod
    def _mut_intern(it, func_set):
        term1_index = np.random.choice(it.len)
        term2_index = np.random.choice(it.len)

        newt, newf = it.get_term(term1_index)
        for i, exp in enumerate(it.terms[term2_index]):
            newt[i] = (newt[i]-exp)%Evol_operators.expolim * (-1 if newt[i]-exp>=0 else 1)
            
        it.add_term(newt, newf)
            
        return it

    @staticmethod
    def mutation(args):

        func_set, fit_fun, label, sol = args
        it = sol.it.copy()
        
        mutations = {
            'term' : Evol_operators._mut_term,
            'func' : Evol_operators._mut_func
        }

        if it.len>Evol_operators.min_nof_terms:
            mutations.update({'drop' : Evol_operators._mut_drop})

        if it.len<Evol_operators.max_nof_terms:
            mutations.update({
                'add'    : Evol_operators._mut_add,
                'intern' : Evol_operators._mut_intern,
                'interp' : Evol_operators._mut_interp,
            })

        mutation = np.random.choice(list(mutations))

        it = mutations[mutation](it, func_set)
        if it.len == 0:
            return sol.copy()
        else:
            ind = Individual(fit_fun, func_set, Evol_operators.expolim, Evol_operators.max_nof_terms, it, label)
            return ind

    @staticmethod
    def crossover(func_set, fit_fun, label, it1, it2, rate):

        t1, f1 = it1.it.terms, it1.it.funcs
        t2, f2 = it1.it.terms, it1.it.funcs
        
        picked_t = []
        picked_f = []

        for t, f in zip(t1 + t2, f1+f2):
            if np.random.rand() < rate:
                picked_t.append(t.copy())
                picked_f.append(f)

        # print("---")
        # print(t1, f1)
        # print(t2, f2)
        # print(picked_t, picked_f)

        it = IT(picked_t, picked_f, label) if len(picked_t)!= 0 else None

        ind = Individual(fit_fun, func_set, Evol_operators.expolim, Evol_operators.max_nof_terms, it, label)

        # print("---")
        # print(it1.it.to_str())
        # print(it2.it.to_str())
        # print(ind.it.to_str())

        return ind
