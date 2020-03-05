import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression 

# Rodando o cProfile:
# python3 -m cProfile -s tottime itea.py > output.txt 

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

    # Variável de classe para compartilharem um dicionário
    # com o score para evitar muitos evals
    _memory = dict()

    # os its são devem receber uma lista de termos para criar a classe. A ideia é não criar expressões com termos inválidos
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
        self.fitness : float     = np.inf


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
            #Z[:, i] = [self.funcList[fi](z) for z in Z[:, i]]
            Z[:, i] = self.funcList[fi](np.prod(X**ni, axis=1))

        return Z


    def fit(self, model, X: List[List[float]], y: List[float]) -> Union[float, None]:

        key = b''.join([t.tostring() + str.encode(f) for t, f in zip(self.terms, self.funcs)])
        
        if key not in ITExpr._memory:
            # Deve receber um modelo com função fit e predict, e que tenha
            # como atributos os coeficientes e intercepto (ou seja, modelo linear)
            # Retorna uma expressão fitada
            Z = self._eval(X)

            assert not (np.isinf(Z).any() or np.isnan(Z).any() or np.any(Z > 1e+300) or np.any(Z < -1e+300)), \
                'Um ou mais termos apresentaram NaN/inf durante o fit. Verifique se as ITs passadas foram "limpadas" antes.'
            
            model.fit(Z, y)

            ITExpr._memory[key] = (
                model.coef_.tolist(),
                model.intercept_,
                np.sqrt(np.square(model.predict(Z) - y).mean())
            )

        self.coeffs, self.bias, self.fitness = ITExpr._memory[key] 

        return self.fitness


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

        newt[newt < -self.expolim[0]] = -self.expolim[0]
        newt[newt > self.expolim[1]]  = self.expolim[1]
        
        return ( np.concatenate((terms, [newt])), np.concatenate((funcs, [funcs[term1_index]])) )


    def _mut_intern(self, ITs: IT) -> IT:
        terms, funcs = ITs

        term1_index = np.random.choice(len(terms))
        term2_index = np.random.choice(len(terms))

        newt = terms[term1_index] - terms[term2_index]

        newt[newt < -self.expolim[0]] = -self.expolim[0]
        newt[newt > self.expolim[1]]  = self.expolim[1]
        
        return ( np.concatenate((terms, [newt])), np.concatenate((funcs, [funcs[term1_index]])) )


    def _mut_interaction(self, combf: Callable) -> Callable:
        # Recebe também uma funçao para realizar a combinação, e retorna uma funçaõ
        # que corresponde a uma interação aplicando essa função

        def _partially(ITs: IT) -> IT:
            terms, funcs = ITs

            term1_index = np.random.choice(len(terms))
            term2_index = np.random.choice(len(terms))

            newt = np.array([combf(terms[term1_index][i], terms[term2_index][i]) for i in range(self.nvars)])

            newt[newt < -self.expolim[0]] = -self.expolim[0]
            newt[newt > self.expolim[1]]  = self.expolim[1]
            
            return ( np.concatenate((terms, [newt])), np.concatenate((funcs, [funcs[term1_index]])) )

        return _partially


    def mutate(self, ITs: IT) -> IT:
        mutations = { #(probabilidade, função de mutação)
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
def _randITBuilder(minterms: int, maxterms: int, nvars: int, expolim: Tuple[int], funs: FuncsList) -> IT:
    while True:
        nterms = np.random.randint(minterms, maxterms + 1)

        terms: Terms = np.random.randint(expolim[0], expolim[1] + 1, size=(nterms, nvars)) 
        funcs: Funcs = np.random.choice(list(funs.keys()), size=nterms)

        yield (terms, funcs)


# Não é feito dentro da IT pois queremos que elas sejam criadas válidas, 
# ao invés de checar no construtor. A ideia é não criar uma IT sem limpar os termos antes,
# para garantir que nada será criado sem nenhum termo
def sanitizeIT(ITs: IT, funcsList, X: List[List[float]]) -> Union[IT, None]:

    def isInvalid(t, f, X):
        key = t.tostring() + str.encode(f)
        
        if key not in sanitizeIT._memory:   
            Z = funcsList[f](np.prod(X**t, axis=1))

            sanitizeIT._memory[key] = np.isinf(Z).any() or np.isnan(Z).any() or np.any(Z > 1e+300) or np.any(Z < -1e+300)
        
        return sanitizeIT._memory[key]
    
    # Se em algum momento não houver mais termos, o zip(* ... ) vai lançar um ValueError
    try:
        terms_funcs = []
        for t, f in zip(ITs[0], ITs[1]):
            if np.any(t!=0):
                # Pelos testes preliminares, tirar termos repetidos faz com que
                # o tempo aumente consideravelmente, mas a convergência é muito
                # melhor
                for t2, f2 in terms_funcs:
                    if (f == f2) and np.all(t == t2):
                        continue

                terms_funcs.append((t, f))

        # Removendo os que não fitam
        terms, funcs = zip(*[(t, f) for t, f in terms_funcs if not isInvalid(t, f, X)])

        return (np.array(terms), np.array(funcs))

    except:
        return None

# Dicionário para memorizar termos inválidos
sanitizeIT._memory = dict()


class ITEA:
    def __init__(self, funs, minterms, maxterms, model, expolim, popsize, gens):
        self.funs     = funs
        self.minterms = minterms
        self.maxterms = maxterms
        self.model    = model
        self.expolim  = expolim
        self.popsize  = popsize
        self.gens     = gens
        

    def _generate_random_pop(self) -> List[IT]:
        randITGenerator = _randITBuilder(self.minterms, self.maxterms, self.nvars, self.expolim, self.funs)
        
        pop = []

        # Garantir que a população terá o mesmo tamanho que foi passado
        while len(pop) < self.popsize:
            itxClean = sanitizeIT(next(randITGenerator), self.funs, self.Xtrain)

            if itxClean:
                itexpr = ITExpr(itxClean, self.funs)
                itexpr.fit(self.model, self.Xtrain, self.ytrain)
            
                pop.append(itexpr)
        
        return pop


    def _mutate(self, ind) -> List[IT]:

        itxClean = sanitizeIT(self.mutate.mutate((ind.terms, ind.funcs)), self.funs, self.Xtrain)
        
        if itxClean:
            itexpr = ITExpr(itxClean, self.funs)
            itexpr.fit(self.model, self.Xtrain, self.ytrain)

            return itexpr

        return None 


    def run(self, Xtrain, ytrain, log=None, verbose=False):
        self.Xtrain = Xtrain
        self.ytrain = ytrain
        self.nvars  = len(Xtrain[0])
        self.mutate = MutationIT(self.minterms, self.maxterms, self.nvars, self.expolim, self.funs)

        if log != None:
            results = {c:[] for c in ['gen', 'bestfit', 'pmean', 'plen']}  
        
        ftournament = lambda x, y: x if x.fitness < y.fitness else y

        pop = self._generate_random_pop()

        if verbose:
            print('gen\tbest fitness\tmean fitness\tmean length')

        for g in range(self.gens):
            pop = pop + list(filter(None, [self._mutate(ind) for ind in pop]))

            pop = [ftournament(*np.random.choice(pop, 2)) for _ in range(self.popsize)] 
            
            if verbose:
                best  = min(pop, key= lambda itexpr: itexpr.fitness)
                pmean, plen = np.mean([(itexpr.fitness, itexpr.len) for itexpr in pop], axis=0)

                if log != None:
                    results['gen'].append(g)
                    results['bestfit'].append(best.fitness)
                    results['pmean'].append(pmean)
                    results['plen'].append(plen)
                    
                print(f'{g}/{self.gens}\t{best.fitness}\t{pmean}\t{plen}')

        self.best = min(pop, key= lambda itexpr: itexpr.fitness)
        
        if log != None:
            df = pd.DataFrame(results)
            df.to_csv(log, index=False)

        return self.best


# ---------------------------------------------------------------------------------
if __name__ == '__main__':

    # Exemplo de uso da regressão simbólica
    funs: FuncsList = { # Funções devem ser unárias, f:R -> R
        'sin'      : np.sin,
        'cos'      : np.cos,
        'tan'      : np.tan,
        'abs'      : np.abs,
        'id'       : lambda x: x,
        'sqrt.abs' : lambda x: np.sqrt(np.absolute(x)),
        'exp'      : lambda x: np.exp(300) if x>=300 else np.exp(x),
        'log'      : lambda x: 0 if x<=0 else np.log(x)
    }

    # Parâmetros para criar uma ITExpr
    nvars    = 5 #Isso vem do dataset que será utilizado
    minterms = 2
    maxterms = 10
    model    = LinearRegression(n_jobs=-1)
    expolim  = (-1, 3)
    popsize  = 100
    gens     = 100

    dataset = np.loadtxt(f'../datasets/airfoil-train-0.dat', delimiter=',')
    Xtrain, ytrain = dataset[:, :-1], dataset[:, -1]

    symbreg = ITEA(funs, minterms, maxterms, model, expolim, popsize, gens)
    best    = symbreg.run(Xtrain, ytrain, log='./res.csv', verbose=True)

    dataset = np.loadtxt(f'../datasets/airfoil-test-0.dat', delimiter=',')
    Xtest, ytest = dataset[:, :-1], dataset[:, -1]
