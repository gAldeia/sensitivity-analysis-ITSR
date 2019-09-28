import time
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool
from sklearn import metrics
from sklearn.linear_model import LinearRegression 

#log file
import os.path
import pandas as pd

from it import IT
from fitness import Fitness
from evol import Evol_operators
from individual import Individual


#todo: métodos internos do ITES (com '_') não alteram a self.pop, e sim a pop 
# passada como parâmetro


class ITES:

    def __init__(self, pop_len, gens, funcs, expolim, max_terms, label=[], log=None):

        self.pop_len   = pop_len
        self.gens      = gens
        self.func_set  = funcs
        self.expolim   = expolim
        self.max_terms = max_terms
        self.log       = log
        self.label     = label

        if self.log != None:
            self.results = {c:[] for c in ['gen', 'bestfit', 'pmean']}   


    def run(self, X, Y):

        #Configurando a classe de operadores evolutivos
        Evol_operators.expolim       = self.expolim
        Evol_operators.min_nof_terms = 1
        Evol_operators.max_nof_terms = self.max_terms

        self.fitfun = Fitness(X, Y)

        self.pop = self._create_init_pop()

        for g in range(self.gens):

            ftournament = lambda x, y: x if x.fitness < y.fitness else y

            childs = self.pop

            #crossover
            #childs = self._apply_crossover(childs, ftournament)
            
            #mutação
            childs = self._apply_mutation(childs)           
            
            self.pop = self._apply_tourn_sel(childs + self.pop, ftournament)

            
            self.printInfo(g)

        if self.log != None:
            df = pd.DataFrame(self.results)
            df.to_csv(self.log, index=False)

        return self


    def printInfo(self, g):

        best    = self.get_best()
        bestfit = np.inf if best is None else best.fitness 
        
        pfit  = [ind.fitness for ind in self.pop]
        pmean = 0 if len(pfit) == 0 else np.mean(pfit)
        
        print(g, bestfit, pmean)

        if self.log != None:
            self.results['gen'].append(g)
            self.results['bestfit'].append(bestfit)
            self.results['pmean'].append(pmean)


    def get_best(self):

        # Returns the best IT
        if len(self.pop) == 0:
            return None

        return min(self.pop, key=lambda ind: ind.fitness)


    def _create_init_pop(self):
        
        pop = [Individual(self.fitfun, self.func_set, label=self.label) 
               for _ in range(self.pop_len)]
        
        return pop


    def _apply_tourn_sel(self, pop, tournament, size=None):

        if len(pop) == 0:
            return []
        if size == None:
            size = self.pop_len
        
        new_pop = [tournament(*np.random.choice(pop, 2)) 
                   for _ in range(size)] 
        
        return new_pop


    def _apply_mutation(self, pop):
                
        pool = Pool(nodes=4)
                
        args = [(self.func_set, self.fitfun, self.label, sol) for sol in pop]
        mutated_pop = pool.map(Evol_operators.mutation, args)

        #mutated_pop = [mutation(arg) for arg in args]
        
        return mutated_pop


    def _apply_crossover(self, pop, ftournament):

        cross = []
        rate = 0.30
        
        for _ in range(len(pop)):
            p1, p2 = self._apply_tourn_sel(self.pop, ftournament, 2)

            child = Evol_operators.crossover(self.func_set, self.fitfun, self.label, p1, p2, rate)
            cross.append(child)

        return cross


    def to_str(self):
        # Returns a string containing the whole population

        return '\n'.join(["Score: " + str(sol.fitness) + ", " + sol.it.to_str() for sol in self.pop])