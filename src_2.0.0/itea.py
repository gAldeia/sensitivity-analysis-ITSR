import numpy as np

from sklearn.linear_model import LinearRegression 


# Para executar a checagem de tipos: mypy --ignore-missing-imports itea.py 
# (requer os módulos typing, nptyping e mypy, facilmente instalados via pip).
# O desenvolvimento dessas ferramentas ainda está em progresso, e temos várias discussões e implementações
# sendo feitas no momento. Atualmente (fev/2020), o mypy não suporta checagem para o numpy, então as anotações
# nesse código servem apenas para organização (e hinting em algumas IDEs).

from typing   import Dict, Callable, List, Tuple, Union
from nptyping import Array #https://pypi.org/project/nptyping/, typing dedicado a arrays numpy

#TODO: UNIFICAR: FUNS, FUNCSLIST, FUNCLIST, QUAL AFINAL?

# Definição de tipos
FuncsList = Dict[str, Callable]
MutatList = Dict[str, Callable]
Funcs     = List[str]
Terms     = List[List[int]]
Coeffs    = List[float]
IT        = Tuple[Terms, Funcs]
    
np.seterr(all='ignore')
    
# Uma it é uma tupla com termos e funções.
# uma ITExpr é uma classe para guardar uma it e informações associadas à ela (score, tamanho, bias, coeficientes).
# alem disso, a ITExpr procura facilitar o uso de algumas funcionalidades comuns aos modelos de regressão do scikit:
# métodos fit (que ajustam os parâmetros internos) e predict. Como adicional, a classe ITExpr permite imprimir a expressão
# resultante de forma mais legível.
class ITExpr:
    def __init__(self, ITs: IT, funcList: FuncsList, labels: List[str] = []) -> None:
        
        # Variáveis que não são modificadas após criação de uma instância
        self.terms: Terms
        self.funcs: Funcs

        self.terms, self.funcs = ITs

        self.labels  : List[str] = labels
        self.funcList: FuncsList = funcList
        self.len     : int       = len(self.terms)

        # Variáveis que são modificadas com fit
        self.bias    : float     = 0.0
        self.coeffs  : Coeffs    = np.ones(self.len)
        self.err     : float     = np.inf
        #TODO: trocarr err por fitness?

    def __str__(self) -> str:
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


    def _eval(self, X: List[List[float]]) -> List[List[float]]:
        Z = np.zeros( (len(X), self.len) )

        for i, (ni, fi) in enumerate( zip(self.terms, self.funcs) ):
            Z[:, i] = np.prod(X**ni, axis=1)
            Z[:, i] = [self.funcList[fi](z) for z in Z[:, i]]

        return Z


    def fit(self, model, X: List[List[float]], y: List[float]) -> Union[float, None]:

        # Deve receber um modelo com função fit e predict, e que tenha
        # como atributos os coeficientes e intercepto (ou seja, modelo linear)
        # Retorna uma expressão fitada
        Z = self._eval(X)

        if np.isinf(Z).any() or np.isnan(Z).any() or np.any(Z > 1e+300) or np.any(Z < -1e+300):
            return None
        
        model.fit(Z, y)

        self.coeffs = model.coef_.tolist()
        self.bias   = model.intercept_
        self.err    = np.sqrt(np.square(model.predict(Z) - y).mean())

        return self.err


    def predict(self, X: List[List[float]]) -> float:

        return np.dot(self._eval(X), self.coeffs) + self.bias


class MutationIT:

    def __init__(self, minterms: int, maxterms: int, nvars: int, expolim: int, funcsList: FuncsList) -> None:
        self.minterms = minterms
        self.maxterms = maxterms
        self.nvars    = nvars
        self.expolim  = expolim
        self.funs     = funcsList

        self.singleITGenerator = _randITBuilder(1, 1, nvars, expolim, funcsList)

    # Será garantido que as mutações internas só serão chamadas se elas puderem retornar uma IT.
    # Além disso, nenhuma mutação modifica os argumentos passados
    def _mut_drop(self, ITs: IT) -> IT:
        terms, funcs = ITs

        index = np.random.randint(0, len(terms))
        mask = [True if i is not index else False for i in range(len(terms))]

        # TODO: checar se isso retorna cópia
        return (terms[mask], funcs[mask])

    def _mut_add(self, ITs: IT) -> IT:
        terms, funcs = ITs
        newt,  newf  = next(self.singleITGenerator)
        
        return ( np.concatenate((terms, newt)), np.concatenate((funcs, newf)) )

    def _mut_term(self, ITs: IT) -> IT:
        terms, funcs = ITs

        newt, _ = next(self.singleITGenerator)
        terms = np.copy(terms)
        terms[np.random.randint(0, len(terms))] = newt[0]

        return (terms, funcs)

    def _mut_func(self, ITs: IT) -> IT:
        terms, funcs = ITs

        _, newf = next(self.singleITGenerator)
        funcs = np.copy(funcs)
        funcs[np.random.randint(0, len(funcs))] = newf[0]

        return (terms, funcs)

    def _mut_interp(self, ITs: IT) -> IT:
        terms, funcs = ITs

        term1_index = np.random.choice(len(terms))
        term2_index = np.random.choice(len(terms))

        newt = terms[term1_index] + terms[term2_index]

        newt[newt < -self.expolim] = -self.expolim
        newt[newt > self.expolim]  = self.expolim
        
        return ( np.concatenate((terms, [newt])), np.concatenate((funcs, [funcs[term1_index]])) )

    def _mut_intern(self, ITs: IT) -> IT:
        terms, funcs = ITs

        term1_index = np.random.choice(len(terms))
        term2_index = np.random.choice(len(terms))

        newt = terms[term1_index] - terms[term2_index]

        newt[newt < -self.expolim] = -self.expolim
        newt[newt > self.expolim]  = self.expolim
        
        return ( np.concatenate((terms, [newt])), np.concatenate((funcs, [funcs[term1_index]])) )


    def _mut_interaction(self, combf: Callable) -> Callable:
        # Recebe também uma funçao para realizar a combinação, e retorna uma funçaõ
        # que corresponde a uma interação aplicando essa função

        def _partially(ITs: IT) -> IT:
            terms, funcs = ITs

            term1_index = np.random.choice(len(terms))
            term2_index = np.random.choice(len(terms))

            newt = np.array([combf(terms[term1_index][i], terms[term2_index][i]) for i in range(self.nvars)])

            newt[newt < -self.expolim] = -self.expolim
            newt[newt > self.expolim]  = self.expolim
            
            return ( np.concatenate((terms, [newt])), np.concatenate((funcs, [funcs[term1_index]])) )

        return _partially

    def mutate(self, ITs: IT) -> IT:
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

            # Outras ideias de mutação usando interação entre os modelos
            #mutations['intertim'] = self._mut_interaction(lambda x, y: x * y)     #times
            #mutations['interdiv'] = self._mut_interaction(lambda x, y: x // y)    #division
            #mutations['intermax'] = self._mut_interaction(lambda x, y: max(x, y)) #maximum
            #mutations['intermin'] = self._mut_interaction(lambda x, y: min(x, y)) #minimum

        return mutations[np.random.choice(list(mutations.keys()))](ITs)



# Lista infinita com termos aleatórios
def _randITBuilder(minterms: int, maxterms: int, nvars: int, expolim: int, funcsList: FuncsList) -> IT:
    while True:
        nterms = np.random.randint(minterms, maxterms + 1)

        terms: Terms = np.random.randint(-expolim, expolim + 1, size=(nterms, nvars)) 
        funcs: Funcs = np.random.choice(list(funs.keys()), size=nterms)

        yield (terms, funcs)


def sanitizeIT(ITs: IT, funcsList, X: List[List[float]]) -> Union[IT, None]:
    terms, funcs = ITs

    def isInvalid(t, f, X):
        Z = np.prod(X**t, axis=1)
        Z = np.array([funcsList[f](z) for z in Z])

        return np.isinf(Z).any() or np.isnan(Z).any() or np.any(Z > 1e+300) or np.any(Z < -1e+300)

    # Se em algum momento não houver mais termos, o zip(* ... ) vai lançar um ValueError
    try:
        # Pega somente termos não repetidos
        #terms, funcs = zip(* list(set(zip(terms, funcs))) )

        # Removendo aqueles com todos os termos iguais a zero
        terms, funcs = zip(* [(t, f) for t, f in zip(terms, funcs) if np.all(t!=0)] )

        # Removendo os que não fitam
        terms, funcs = zip(* [(t, f) for t, f in zip(terms, funcs) if not isInvalid(t, f, X)] )
    except:
        return None

    return (np.array(terms), np.array(funcs))

def createRandPop(popsize, minterms, maxterms, nvars, expolim, funs, Xtrain, ytrain, model) -> List[IT]:
    randITGenerator = _randITBuilder(minterms, maxterms, nvars, expolim, funs)
    
    pop = []

    # Garantir que a população terá o mesmo tamanho que foi passado
    while len(pop) < popsize:
        itxClean = sanitizeIT(next(randITGenerator), funs, Xtrain)

        if itxClean:
            itexpr = ITExpr(itxClean, funs)
            score  = itexpr.fit(model, Xtrain, ytrain)
        
            if score:
                pop.append(itexpr)
    
    return pop

def mutatePop() -> List[IT]:
    pass

# ---------------------------------------------------------------------------------
if __name__ == '__main__':

    funs: FuncsList = { # Funções devem ser unárias, f:R -> R
        'sin'  : np.sin,
        'cos'  : np.cos,
        'tan'  : np.tan,
        'abs'  : np.abs,
        'id'   : lambda x: x,
        'sqrt' : lambda x: 0 if x<0 else np.sqrt(x),
        'exp'  : lambda x: np.exp(300) if x>=300 else np.exp(x),
        'log'  : lambda x: 0 if x<=0 else np.log(x)
    }

    # Parâmetros para criar uma ITExpr
    nvars    = 5 #Isso vem do dataset que será utilizado
    minterms = 2
    maxterms = 10
    model    = LinearRegression(n_jobs=-1)
    expolim  = 3
    popsize  = 100
    gens     = 100


    dataset = np.loadtxt(f'../datasets/airfoil-train-0.dat', delimiter=',')
    Xtrain, ytrain = dataset[:, :-1], dataset[:, -1]

    dataset = np.loadtxt(f'../datasets/airfoil-test-0.dat', delimiter=',')
    Xtest, ytest = dataset[:, :-1], dataset[:, -1]


    pop = createRandPop(popsize, minterms, maxterms, nvars, expolim, funs, Xtrain, ytrain, model)

    mutate = MutationIT(2, 10, nvars, expolim, funs)

    ftournament = lambda x, y: x if x.err < y.err else y

    def mutate_sanitized(ind, funs, Xtrain):
        itxClean = sanitizeIT(mutate.mutate((ind.terms, ind.funcs)), funs, Xtrain)
        
        if itxClean:
            itexpr = ITExpr(itxClean, funs)
            score  = itexpr.fit(model, Xtrain, ytrain)

            return itexpr if score else None

        return None 
        
    for g in range(gens):
        childs = list(filter(None, [mutate_sanitized(ind, funs, Xtrain) for ind in pop]))

        pop = pop + childs

        pop = [ftournament(*np.random.choice(pop, 2)) for _ in range(popsize)] 
  
        meanErr, meanLen = np.mean([(itexpr.err, itexpr.len) for itexpr in pop], axis=0)
        best = min(pop, key= lambda itexpr: itexpr.err)

        print(g, best.err, meanErr, meanLen)