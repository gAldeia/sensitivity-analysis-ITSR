#reminder:  sudo python3.5 -m pip install [package]
#upgrading: sudo python3.5 -m pip install [package] --upgrade

import re  # Regular expressions for string formatting

from copy import deepcopy #create copies to apply mutation 
from math import isnan, isfinite #avoid nan scores being selected

import numpy as np

class IT:
    def __init__(self, terms, funcs, labels=[]):
        # Checks consistent parameters and initializes the class.

        assert all(len(term) == len(terms[0]) for term in terms), \
               'all terms must have the same number of exponents'
        assert len(terms) == len(funcs), \
               'every term must have an associated function'

        self.terms  = []
        self.funcs  = []
        self.coeffs = np.array([])
        self.intercept = 0.0
        self.len    = 0 

        for term, func in zip(terms, funcs):
            self.add_term(term, func)

        if len(labels) == 0:
            self.labels = ['x' + str(i) for i in range(len(terms[0]))]
        else:
            self.labels = labels

    def _eval(self, X):
        # Evaluates the ITs for a given point (Internal use)

        Z = np.zeros( (X.shape[0], self.len) )

        for i, (ni, fi) in enumerate( zip(self.terms, self.funcs) ):
            Z[:, i] = np.prod(X**ni, axis=1)
            Z[:, i] = fi[1](Z[:, i])

        return Z


    def predict(self, X):

        return np.dot(self._eval(X),self.coeffs) + self.intercept


    def to_str(self):
        # Returns a string representing the expression.

        #BOTAR UM LIST COMPREHENSION AQUI, FICA MAIS LEGÍVEL 
        coeffs = list(map(lambda x:str(round(x,6)), self.coeffs))
        funcs  = list(map(lambda x:x[0], self.funcs))

        f_its = []

        for (c, f, t) in zip(coeffs, funcs, self.terms):
            if c!= '0.0' or c!='-0.0':
                term = [x+'^'+str(e) for x,e in zip(self.labels, t) if e != 0]
                f_its.append(c + '*' + f + '(' + ' * '.join(term) + ')')
        
        return re.sub(r'\^1', '', ' + '.join(f_its)) + ' + ' + str(self.intercept)
    

    def add_term(self, term, func):
        # Checks consistence and adds a new term and func into the expression.

        for i in range(self.len):
            if (np.all(self.terms[i] == term)) and (self.funcs[i] == func):
                return self

        if all(ti==0 for ti in term):
            if len(self.terms) == 0:
                #adicionar um novo aleatório até conseguir um que não seja nulo, com expoentes entre 0 e 1
                self.add_term(np.random.randint(0, 2, size=len(term)).tolist(), func)
                
            return self

        self.terms.append(term.copy())
        self.funcs.append(func)
        self.coeffs = np.append(self.coeffs, 1.0)

        self.len = self.len+1

        return self


    def remove_term(self, index):
        # Removes one element from the expression if possible.

        if self.len>1 :
            self.terms.pop(index)
            self.funcs.pop(index)
            self.coeffs = np.delete(self.coeffs, index)

            self.len = self.len-1

        return self


    def get_term(self, index):
        #returns the selected term

        return self.terms[index].copy(), self.funcs[index]


    def simplify(self, threshold):
        #removes all ITs with coeff less than threshold

        size = range(self.len)

        for i in reversed(size):
            if abs(self.coeffs[i]) < threshold:
                self.remove_term(i)

        return self

    def copy(self):

        clone = IT(self.terms, self.funcs, self.labels)
        clone.coeffs = self.coeffs.copy()
        clone.intercept = self.intercept
        
        return clone

    def get_key(self):

        # Transforma sua expressão em uma chave de dicionário

        funcs = list(map(lambda x:x[0], self.funcs))

        sorted_terms = [f'{f}-{t}' for _, f, t in sorted(zip(self.coeffs, funcs, self.terms), key=lambda tup: tup[0])]

        return '_'.join(sorted_terms)