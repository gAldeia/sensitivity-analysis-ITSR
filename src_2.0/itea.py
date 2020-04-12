import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression 

from typing   import Dict, Callable, List, Tuple, Union
from nptyping import Array 

# To run type checking (still in development): mypy --ignore-missing-imports itea.py 
# (this requires typing, nptyping and mypy)
# The development of those tools are work in progress. At this moment (fev/2020), 
# mypy doesn't support type checking for numpy arrays. This way, the types specified in
# this code are just for better understandment of how the code works.


FuncsList = Dict[str, Callable]
MutatList = Dict[str, Callable]
Funcs     = List[str]
Terms     = List[List[int]]
Coeffs    = List[float]
IT        = Tuple[Terms, Funcs]


# Hiding warning messages that can occur during the evolution, since expressions
# for complex datasets can present NaN in specific cases.   
np.seterr(all='ignore')
    

class ITExpr:
    # Class to represent an Interaction-Transformation (IT) function.
    # This class stores an IT, informations regarding the stored IT
    # (score, length, bias, coefficients), and provide fit() and predict() methods.

    # Memorization of scores, coefficients and bias
    _memory = dict()


    def __init__(self, ITs: IT, funcList: FuncsList, labels: List[str] = []) -> None:
        
        assert len(ITs[0])>0 and len(ITs[1])>0, 'Terms or Funcs has len = 0'

        # Below are variables that any method changes after creating a instance
        self.terms: Terms
        self.funcs: Funcs

        self.terms, self.funcs = ITs

        self.labels  : List[str] = labels
        self.funcList: FuncsList = funcList
        self.len     : int       = len(self.terms)

        # Variables that are changed by the fit()
        self.bias    : float     = 0.0
        self.coeffs  : Coeffs    = np.ones(self.len)
        self.fitness : float     = np.inf


    def _eval(self, X: List[List[float]]) -> List[List[float]]:

        Z = np.zeros( (len(X), self.len) )

        for i, (ni, fi) in enumerate( zip(self.terms, self.funcs) ):
            #Z[:, i] = [self.funcList[fi](z) for z in Z[:, i]]
            Z[:, i] = self.funcList[fi](np.prod(X**ni, axis=1))

        return Z


    def __str__(self) -> str:
        # Overriding the str representation to print with a pretty formatation

        terms_str = [] 

        for c, f, t in zip(self.coeffs, self.funcs, self.terms):
            c_str = "" if round(c, 3) == 1.0 else f'{round(c, 3)}*'
            f_str = "" if f == "id" else f
            t_str = ' * '.join([
                "x" + str(i) + ("^"+str(ti) if ti!=1 else "")
                for i, ti in enumerate(t) if ti!=0
            ])

            terms_str.append(f'{c_str}{f_str}({t_str})')

        expr_str = ' + '.join(terms_str)

        for i, l in enumerate(self.labels):
            expr_str = expr_str.replace(f'x{i}', l)
        
        return expr_str  + ("" if self.bias == 0.0 else f' + {round(self.bias, 3)}')


    def fit(self, model, X: List[List[float]], y: List[float]) -> Union[float, None]:

        # Uses a linear model from scikit learn to adjust the coefficients and bias

        key = b''.join([t.tostring() + str.encode(f) for t, f in zip(self.terms, self.funcs)])

        key_t = b''.join([t.tostring() for t in self.terms])
        key_f = b''.join([f.encode()   for f in self.funcs])

        key = (key_t, key_f)

        if key not in ITExpr._memory:
            Z = self._eval(X)

            #assert not (np.isinf(Z).any() or np.isnan(Z).any() or np.any(Z > 1e+300) or np.any(Z < -1e+300)), \
            #    f'One or more Terms presented NaN/inf during the fit.'
            
            if (np.isinf(Z).any() or np.isnan(Z).any() or np.any(Z > 1e+300) or np.any(Z < -1e+300)):
                ITExpr._memory[key] = (
                    np.ones(self.len),
                    0.0,
                    1e+100
                )
            else:
                model.fit(Z, y)

                ITExpr._memory[key] = (
                    model.coef_.tolist(),
                    model.intercept_,
                    np.sqrt(np.square(model.predict(Z) - y).mean())
                )

        self.coeffs, self.bias, self.fitness = ITExpr._memory[key] 

        return self.fitness


    def predict(self, X: List[List[float]]) -> float:

        # Takes a list of x values to predict, and return a list
        # of predicted values (for a single value, you should
        # encapsulate it inside a list)

        return np.dot(self._eval(X), self.coeffs) + self.bias



class MutationIT:
    # Class to hold all mutation methods together (internal usage). 
    # It is guaranteed that, when calling a random mutation, only those that
    # returns a valid IT will be selected.
    # Also, the IT used as argument will not be modified.

    def __init__(self, minterms: int, maxterms: int, nvars: int, expolim: int, funcsList: FuncsList) -> None:
        self.minterms = minterms
        self.maxterms = maxterms
        self.nvars    = nvars
        self.expolim  = expolim
        self.funs     = funcsList

        self.singleITGenerator = _randITBuilder(1, 1, nvars, expolim, funcsList)
    

    def _mut_drop(self, ITs: IT) -> IT:
        terms, funcs = ITs

        index = np.random.randint(0, len(terms))
        mask = [True if i is not index else False for i in range(len(terms))]

        return (terms[mask], funcs[mask])


    def _mut_add(self, ITs: IT) -> IT:
        terms, funcs = ITs
        newt,  newf  = next(self.singleITGenerator)
        
        return ( np.concatenate((terms, newt)), np.concatenate((funcs, newf)) )


    def _mut_term(self, ITs: IT) -> IT:
        terms, funcs = ITs

        index = np.random.randint(0, len(terms))

        newt, _ = next(self.singleITGenerator)
        newf = [funcs[index]]

        mask = [True if i is not index else False for i in range(len(terms))]

        return ( np.concatenate((terms[mask], newt)), np.concatenate((funcs[mask], newf)) )


    def _mut_func(self, ITs: IT) -> IT:
        terms, funcs = ITs

        index = np.random.randint(0, len(terms))

        _, newf = next(self.singleITGenerator)
        newt = [terms[index]]

        mask = [True if i is not index else False for i in range(len(terms))]

        return ( np.concatenate((terms[mask], newt)), np.concatenate((funcs[mask], newf)) )


    def _mut_interp(self, ITs: IT) -> IT:
        terms, funcs = ITs

        term1_index = np.random.choice(len(terms))
        term2_index = np.random.choice(len(terms))

        newt = terms[term1_index] + terms[term2_index]

        newt[newt < self.expolim[0]] = self.expolim[0]
        newt[newt > self.expolim[1]]  = self.expolim[1]
        
        return ( np.concatenate((terms, [newt])), np.concatenate((funcs, [funcs[term1_index]])) )


    def _mut_intern(self, ITs: IT) -> IT:
        terms, funcs = ITs

        term1_index = np.random.choice(len(terms))
        term2_index = np.random.choice(len(terms))

        newt = terms[term1_index] - terms[term2_index]

        newt[newt < self.expolim[0]] = self.expolim[0]
        newt[newt > self.expolim[1]]  = self.expolim[1]
        
        return ( np.concatenate((terms, [newt])), np.concatenate((funcs, [funcs[term1_index]])) )


    def _mut_interaction(self, combf: Callable) -> Callable:
        # Use custom interaction functions to create a new mutation
        # (in development, performance issues are known).

        def _partially(ITs: IT) -> IT:
            terms, funcs = ITs

            term1_index = np.random.choice(len(terms))
            term2_index = np.random.choice(len(terms))

            newt = np.array([combf(terms[term1_index][i], terms[term2_index][i]) for i in range(self.nvars)])

            newt[newt < self.expolim[0]] = self.expolim[0]
            newt[newt > self.expolim[1]] = self.expolim[1]
            
            return ( np.concatenate((terms, [newt])), np.concatenate((funcs, [funcs[term1_index]])) )

        return _partially


    def mutate(self, ITs: IT) -> IT:
        # Function to apply a mutation over a IT.

        mutations = {
            'term' : self._mut_term,
            'func' : self._mut_func
        }

        if len(ITs[0]) > self.minterms:
            mutations['drop'] = self._mut_drop
        
        if len(ITs[0]) < self.maxterms:
            mutations['add']    = self._mut_add

            mutations['interpos'] = self._mut_interp    #positive
            mutations['interneg'] = self._mut_intern    #negtive

            # Another mutation ideas
            #mutations['intertim'] = self._mut_interaction(lambda x, y: x * y)     #times
            #mutations['interdiv'] = self._mut_interaction(lambda x, y: x // y)    #division
            #mutations['intermax'] = self._mut_interaction(lambda x, y: max(x, y)) #maximum
            #mutations['intermin'] = self._mut_interaction(lambda x, y: min(x, y)) #minimum

        return mutations[np.random.choice(list(mutations.keys()))](ITs)


def _randITBuilder(minterms: int, maxterms: int, nvars: int, expolim: Tuple[int], funs: FuncsList) -> IT:
    # Infinite list to create terms. Internal usage.

    while True:
        nterms = np.random.randint(minterms, maxterms + 1)

        terms: Terms = np.random.randint(expolim[0], expolim[1] + 1, size=(nterms, nvars)) 
        funcs: Funcs = np.random.choice(list(funs.keys()), size=nterms)

        yield (terms, funcs)


class ITEA:
    # Symbolic Regression method.

    def __init__(self, funs, minterms, maxterms, model, expolim, popsize, gens, check_fit=False):
        self.funs      = funs
        self.minterms  = minterms
        self.maxterms  = maxterms
        self.model     = model
        self.expolim   = expolim
        self.popsize   = popsize
        self.gens      = gens
        self.check_fit = check_fit
                
        # Memorization used when checking if terms fits or results in NaN.
        self._memory = dict()


    def _sanitizeIT(self, ITs: IT, funcsList, X: List[List[float]]) -> Union[IT, None]:
        # Cleans an IT expression. The ITExpr class assumes that every IT expression
        # passed when creating a new instance was cleaned (the creation of
        # ITExprs manually is disencouraged).
        # This basicaly removes repeated terms and terms with all exponents==0.
        # When self.check_fit=True, the cleaning
        # process will remove every Term that evaluates to NaN/inf. This is useful for
        # convergence graphics, but preliminary experiments shows that this can impact
        # negatively on performance (needs further investigations).

        terms, funcs = ITs[0], ITs[1]

        mask = np.full( len(ITs[0]), False )

        _, unique_ids = np.unique(np.column_stack((terms, funcs)), return_index=True, axis=0)

        for unique_id in unique_ids:
            t, f = terms[unique_id], funcs[unique_id]

            assert f in funcsList.keys(), f'{f} não é uma função válida'

            if np.any(t!=0):
                if self.check_fit:
                    key = (t.tobytes(), f)
                    if key not in self._memory:   
                        Z = funcsList[f](np.prod(X**t, axis=1))

                        self._memory[key] = (np.isinf(Z).any() or np.isnan(Z).any() or np.any(Z > 1e+300) or np.any(Z < -1e+300))
                    
                    mask[unique_id] = not self._memory[key]
                else:
                    mask[unique_id] = True

        return (terms[mask], funcs[mask]) if np.any(mask) else None


    def _generate_random_pop(self) -> List[IT]:
        randITGenerator = _randITBuilder(self.minterms, self.maxterms, self.nvars, self.expolim, self.funs)
        
        pop = []

        while len(pop) < self.popsize:
            itxClean = self._sanitizeIT(next(randITGenerator), self.funs, self.Xtrain)

            if itxClean:
                itexpr = ITExpr(itxClean, self.funs)
                
                itexpr.fit(self.model, self.Xtrain, self.ytrain)
                pop.append(itexpr)
            
        return pop


    def _mutate(self, ind) -> List[IT]:

        itxClean = self._sanitizeIT(self.mutate.mutate((ind.terms, ind.funcs)), self.funs, self.Xtrain)
        
        if itxClean:
            itexpr = ITExpr(itxClean, self.funs)
        
            itexpr.fit(self.model, self.Xtrain, self.ytrain)
            return itexpr
        
        return None 


    def run(self, Xtrain, ytrain, log=None, verbose=False):
        
        # Memorization should be cleaned before every evolution.
        ITExpr._memory = dict()
        self._memory   = dict()

        self.Xtrain  = Xtrain
        self.ytrain  = ytrain
        self.nvars   = len(Xtrain[0])
        self.mutate  = MutationIT(self.minterms, self.maxterms, self.nvars, self.expolim, self.funs)

        if log != None:
            results = {c:[] for c in ['gen', 'bestfit', 'pmean', 'plen']}  
        
        ftournament = lambda x, y: x if x.fitness < y.fitness else y

        pop = self._generate_random_pop()

        if verbose:
            print('gen\tbest fitness\tmean fitness\tmean length')

        for g in range(self.gens):
            pop = pop + list(filter(None, [self._mutate(ind) for ind in pop]))

            pop = [ftournament(*np.random.choice(pop, 2)) for _ in range(self.popsize)] 
            
            best  = min(pop, key= lambda itexpr: itexpr.fitness)
            pmean, plen = np.mean([(itexpr.fitness, itexpr.len) for itexpr in pop], axis=0)

            if verbose:
                print(f'{g}/{self.gens}\t{best.fitness}\t{pmean}\t{plen}')
                
            if log:
                results['gen'].append(g)
                results['bestfit'].append(best.fitness)
                results['pmean'].append(pmean)
                results['plen'].append(plen)
                    
        self.best = min(pop, key= lambda itexpr: itexpr.fitness)
        
        if log != None:
            df = pd.DataFrame(results)
            df.to_csv(log, index=False)

        return self.best


# Example --------------------------------------------------------------------------
if __name__ == '__main__':

    nvars           = 5 # This value is refered to the used dataset (airfoil)
    minterms        = 2
    maxterms        = 10
    model           = LinearRegression(n_jobs=-1)
    expolim         = (-1, 3)
    popsize         = 100
    gens            = 100
    funs: FuncsList = { # should be unary functions, f:R -> R
        'sin'      : np.sin,
        'cos'      : np.cos,
        'tan'      : np.tan,
        'abs'      : np.abs,
        'id'       : lambda x: x,
        'sqrt.abs' : lambda x: np.sqrt(np.absolute(x)),
        'log'      : np.log, 
        'exp'      : np.exp,
    }

    dataset = np.loadtxt(f'../datasets/airfoil-train-0.dat', delimiter=',')
    Xtrain, ytrain = dataset[:, :-1], dataset[:, -1]

    symbreg = ITEA(funs, minterms, maxterms, model, expolim, popsize, gens, False)
    best    = symbreg.run(Xtrain, ytrain, log='./res.csv', verbose=True)

    dataset = np.loadtxt(f'../datasets/airfoil-test-0.dat', delimiter=',')
    Xtest, ytest = dataset[:, :-1], dataset[:, -1]
