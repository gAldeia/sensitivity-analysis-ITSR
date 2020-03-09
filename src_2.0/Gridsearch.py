import numpy  as np
import pandas as pd
import itea   as sr

import os.path
import glob
import re
import time
import collections  #Dicionários que se lembram da ordem de inserção

from sklearn   import linear_model, metrics
from itertools import product


#Definindo as métricas que vamos utilizar
def MAE(it, X, y):
    yhat = it.predict(X)
    return np.abs(yhat - y).mean()

def RMSE(it, X, y):
    yhat = it.predict(X)
    return np.sqrt(np.square(yhat - y).mean())

def NMSE(it, X, y):
    yhat = it.predict(X)
    return (np.square(yhat - y)/(yhat.mean()*y.mean())).mean()


#OBS: não interromper o script bem no finalzinho de uma evolução, é o momento
#onde ele começa a salvar vários .csv com as informações, e pode quebrar

# ONDE AS INFORMAÇÕES SÃO SALVAS: ----------------------------------------------
#evolution log:
#   - para plotar gráficos de convergência, salva os nomes no formato
#   {dataset}-{fold}-{rep}-{atributos utilizados}.csv
#grid log:
#   - para um dado dataset, fold e rep, guarda informação de todas as combinaçõe
#     executadas para aquele dataset
#resultsregression:
#   - guarda o resultado do melhor indivíduo do gridsearch para cada base-fold-rep


#Nosso método precisa responder as seguintes perguntas: ------------------------
#   - existe um conjunto de parâmetros fixos que sempre (ou quase sempre) domina todos os outros?
#   - todos os parâmetros devem ser avaliados ao testar um novo dataset (ou podemos deixar alguns fixos)?
#   - qual a variação do erro  quando alteramos os parâmetros? 

#Ideia para fazer a análise de sensibilidade: ----------------------------------
#   - primeiro pegamos as 30 execuções e, para cada possível configuração de hiper-
#     -parâmetros, pegamos a mediana (mais robusta a outliers, que ocorrem na com-
#     -putação evolutiva). Então vamos ter uma matriz n-dimensional, onde n é 
#     o número de hiper-parâmetros variados.
#   - para avaliar quais podem ser fixados ou quais dominam sobre os outros, fazemos
#     uma análise na variância (ou outra métrica de dispersão) do score da matriz
#     n-dimensional obtida anteriormente. Ao pegar todos os valores de um array
#     1-dimensional dessa matriz, é como se o parâmetro correspondente ao array
#     estivesse fixo, enquanto os outros estão variando. (PENSAR MELHOR NISSO)
#   - O passo anterior deve ser feito para cada dataset, para vermos o comportamento
#    individual.
#   - Depois disso vem a "análise global", onde pegamos a média da variância, e
#     com isso podemos ver os parâmetros com média baixa (implicando que não precisam
#     ser avaliados para cada dataset), ou o contrário.
#   - para ver a validade do item anterior, fazemos uma análise de p-valor entre
#     as distribuições obtidas para cada base de dados
#   - uma sugestão seria utilizar o coeficiente de variação (buscar por referências
#     sobre ele depois)



class Gridsearch_ITES:

    def __init__(self, name):
        
        #Parâmetros fixados, serão usados em todos
        self._default_params = {
            'popsize'  : 100, #500,
            'gens'     : 100, #1000,
            'minterms' : 1,
            'model'    : linear_model.LinearRegression(n_jobs=-1),
            'funs'     : {
                "id"      : lambda x: x,
                "sin"     : np.sin, 
                "cos"     : np.cos,        
                "tanh"    : np.tanh,
                "sqrt.abs": lambda x: np.sqrt(np.absolute(x)),
                "log"     : np.log, 
                "exp"     : np.exp,
            }
        }

        self._search_params  = {
            'expolim'  : [(-1, 1), (-2, 2), (-3, 3), (-4, 4), (-5, 5)],  #obs: se expolim = x, range é [-x, x]
            'maxterms' : [2, 4, 6, 8, 10],

        }
        
        #Nome do arquivo para salvar o log do gridsearch.
        self.name = name

        #Note que search_params UNIÃO default_params deve CONTER todos os parâmetros obrigatórios do modelo

        #guardará os scores dos parametros para teste, onde a key desse
        #dicionário é um conjunto de tuplas (key, value) dos valores tes-
        #tados para cada hyperparâmetro.
        self._tested_params = { } #Salva o resultado do teste
        self._tested_return = { } #Salva informações para retornar

        #aqui é um dicionário com os parâmetros utilizados para o melhor resultado
        self.best_params         = { }
        self.results_best_params = None
        self.best_score          = np.inf


    def _create_regressor(self, **kwargs):

        #cria um modelo com os parâmetros padrões (default), mas modificando
        #aqueles passados como kwargs (para poder variar isso), então faz o 
        #fit com a base passada para treino e faz a validação com a base
        #passada para validação, e entõ retorna a métrica utilizada.

        #Aqui ele só cria, não faz a regressão

        _params = {**self._default_params}

        for key, value in kwargs.items():
            _params[key] = value

        sufix = "-".join("=".join([k, str(v)]) for k, v in sorted(kwargs.items()))

        logfile = f'evolution_log/{self.name}-{sufix}.csv'

        itea = sr.ITEA(**_params)

        return itea, logfile


    def _eval(self, ites, X_train, y_train, X_test, y_test, logfile):
        
        bestsol = ites.run(X_train, y_train, log=logfile, verbose=True)
        #ITES só salva o log no final, pois não faz sentido retomar uma evolução no meio
        #Ainda, ele sobrescreve um arquivo se já existir com esse nome, para atu-
        #alizar a execução.

        #Fica a cargo do gridsearch fazer o controle da execução (ou não) do ites

        return bestsol


    def cross_val(self, X_train, y_train, X_test, y_test):

        #o asterisco faz o zip reverso. Aqui criamos todas as configurações
        #de hyperparâmetros que queremos variar no nosso gridsearch
        keys, values = zip(*self._search_params.items())
        hyperconfigs = [dict(zip(keys, v)) for v in product(*values)]

        for hp in hyperconfigs:

            #Caso já tenha feito, não repete, mas recupera os resultados
            if os.path.isfile('./grid_log/' + self.name + '.csv'):
                searchDF = pd.read_csv('./grid_log/' + self.name + '.csv')

                for key, value in hp.items():
                    searchDF = searchDF[searchDF[key]==value]  
                    
                if len(searchDF)==1:
                    rmse_test = searchDF['rmse_test'].values
                    mae_test  = searchDF['mae_test'].values
                    nmse_test = searchDF['nmse_test'].values
                    expr      = searchDF['expr'].values
                
                    self._tested_params[tuple(hp.items())] = rmse_test
                    self._tested_return[tuple(hp.items())] = (rmse_test, mae_test, nmse_test, expr)
                
                    print(f'(Gridsearch) already evaluated configuration {hp}')
                    
                    continue
            
            print(f'(Gridsearch) evaluating configuration {hp}')

            ites, logfile = self._create_regressor(**hp)
            bestsol       = self._eval(ites, X_train, y_train, X_test, y_test, logfile)

            rmse_test = RMSE(bestsol, X_test, y_test)
            mae_test  = MAE(bestsol, X_test, y_test)
            nmse_test = NMSE(bestsol, X_test, y_test)

            expr = str(bestsol)

            #Salva para poder retomar depois
            self._save_csv(hp, rmse_test, mae_test, nmse_test, expr)

            #transforma os hyperparâmetros testados em tuplas para poderem
            #ser utilizadas como key no dicionário. Posteriormente fica mais
            #fácil utilizar assim para converter a melhor combinação de parâ-
            #metros testada para dicionário usando dict(key)

            #As expressões são comparadas pelo valor armazenado em _tested_params.
            #Se for usar outra métrica (ou uma heurística diferente), mudar só ele
            self._tested_params[tuple(hp.items())] = rmse_test
            self._tested_return[tuple(hp.items())] = (rmse_test, mae_test, nmse_test, expr)

        #Atualiza o melhor score e parâmetro
        self.get_best()

        return self.best_score, self.best_params, self.results_best_params


    def get_best(self):

        assert len(self._tested_params.items()) != 0, \
            'get best utilizado sem fazer a busca!'

        best_params = min(self._tested_params, key=lambda key: self._tested_params[key])

        self.best_score          = self._tested_params[best_params]
        self.results_best_params = self._tested_return[best_params]
        self.best_params         = dict(best_params)
        
        return self.best_score, self.best_params, self.results_best_params


    def print_info(self):

        print("--- Gridsearch ranges ---")
        for key, value in self._search_params.items():
            print(key + " range: " + str(value))

        print('-- best final result --')
        print("parameters: " + str(self.best_params))
        print("score: " + str(self.best_score))
        print("-------------------------")


    def _save_csv(self, tested, rmse_test, mae_test, nmse_test, expr):

        #salvar os resultados num arquivo.
        #irá salvar os parâmetros (apenas os que foram variados) e adicionará
        #uma última coluna com o valor da métrica

        columns = [key for key in self._search_params.keys()] +\
                  ['rmse_test', 'mae_test', 'nmse_test', 'expr']

        results = {c:[] for c in columns}

        fname = './grid_log/' + self.name + '.csv'

        if os.path.isfile(fname):
            resultsDF = pd.read_csv(fname)
            results   = resultsDF.to_dict('list')

        for key, value in tested.items():
            results[key].append(value)

        results['rmse_test'].append(rmse_test)
        results['mae_test'].append(mae_test)
        results['nmse_test'].append(nmse_test)
        results['expr'].append(expr)

        df = pd.DataFrame(results)
        df.to_csv(fname, index=False)


if __name__ == '__main__':

    #select which datasets to use
    datasets = ['airfoil', 'concrete', 'energyCooling', 'energyHeating',
        'towerData', 'wineRed', 'wineWhite', 'yacht']    

    columns = ['dataset','best_params','RMSE_test','NMSE_test','MAE_test','Time','Expression','Fold','Rep']
    
    # arquivo com resultados
    fname = 'resultsregression.csv'
    results = {c:[] for c in columns}

    if os.path.isfile(fname):
        resultsDF = pd.read_csv(fname)
        results   = resultsDF.to_dict('list')

    # pastas para os logs:
    if not os.path.exists('grid_log'):
        os.makedirs('grid_log')
    if not os.path.exists('evolution_log'):
        os.makedirs('evolution_log')

    for ds in datasets:
        for fold in range(5):
            dataset = np.loadtxt(f'../datasets/{ds}-train-{fold}.dat', delimiter=',')
            Xtrain, ytrain = dataset[:, :-1], dataset[:, -1]
            
            dataset = np.loadtxt(f'../datasets/{ds}-test-{fold}.dat', delimiter=',')
            Xtest, ytest = dataset[:, :-1], dataset[:, -1]
            
            for rep in range(6): #Número de execuções por fold

                if os.path.isfile(fname):
                    #Abre o arquivo e carrega caso ele exista
                    resultsDF = pd.read_csv(fname)
                    results   = resultsDF.to_dict('list')

                    #Evitando refazer repetidos e possibilita retomar o teste
                    if len(resultsDF[
                            (resultsDF['dataset']==ds) &
                            (resultsDF['Fold']==fold)  &
                            (resultsDF['Rep']==rep)])==1:
                        print(f'already done gridsearch for {ds}-{fold}-{rep}')
                        
                        continue

                print(f'performing gridsearch for {ds}-{fold}-{rep}')

                grid = Gridsearch_ITES(f'{ds}-{fold}-{rep}')

                start = time.time()
                best_score, best_params, results_best_params = grid.cross_val(Xtrain, ytrain, Xtest, ytest)
                tot_time = time.time() - start
                
                grid.print_info()   
                
                rmse_test, nmse_test, mae_test, expr = results_best_params

                results['dataset'].append(ds)
                results['best_params'].append(best_params)
                results['NMSE_test'].append(nmse_test)
                results['RMSE_test'].append(rmse_test)
                results['MAE_test'].append(mae_test)
                results['Time'].append(tot_time)  #Tempo do gridsearch, não de cada uma!
                results['Expression'].append(expr)
                results['Fold'].append(fold)
                results['Rep'].append(rep)

                df = pd.DataFrame(results)
                df.to_csv(fname, index=False)

    print('done')
