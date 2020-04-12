import numpy  as np
import pandas as pd
import itea   as sr

import os.path
import glob
import re
import time
import collections

from sklearn   import linear_model, metrics
from itertools import product


np.seterr(all='ignore')


def MAE(it, X, y):
    yhat = it.predict(X)
    return np.abs(yhat - y).mean()

def RMSE(it, X, y):
    yhat = it.predict(X)
    return np.sqrt(np.square(yhat - y).mean())

def NMSE(it, X, y):
    yhat = it.predict(X)
    return (np.square(yhat - y)/(yhat.mean()*y.mean())).mean()


# GRIDSEARCH DATA STRUCTURE -----------------------------------------------------------
#evolution log:
#   - used to plot convergence of configurations
#   {dataset}-{fold}-{rep}-{hiperconfig}.csv
#grid log:
#   - performance of each configuration for a given {dataset}-{fold}-{rep}
#resultsregression.csv:
#   - stores the best configuration for a given {dataset}-{fold}-{rep}




class Gridsearch_ITES:
    def __init__(self, name):
        self._default_params = {
            'pop_len' : 100, #500,
            'gens'    : 300, #1000,
            'funcs'   : [
                ("id"      , lambda x: x),
                ("sin"     , np.sin), 
                ("cos"     , np.cos),        
                ("tanh"    , np.tanh),
                ("sqrt.abs", lambda x: np.sqrt(np.absolute(x))),
                ("log"     , np.log), 
                ("exp"     , np.exp),
            ]
        }

        self._search_params  = {
            'degree_range' : [1, 2, 3, 4, 5],  #obs: degree_range = x -> degree will vary in [-x, x]
            'max_terms'    : [2, 4, 6, 8, 10]
        }
        
        self.name = name

        self._tested_params = { 
        self._tested_return = { }

        self.best_params         = { }
        self.results_best_params = None
        self.best_score          = np.inf


    def _create_regressor(self, **kwargs):
        _params = {**self._default_params}

        for key, value in kwargs.items():
            _params[key] = value

        sufix = "-".join("=".join([k, str(v)]) for k, v in sorted(kwargs.items()))

        logfile = f'evolution_log/{self.name}-{sufix}.csv'

        ites = sr.ITEA(log=logfile, **_params)

        return ites


    def _eval(self, ites, X_train, y_train):
        
        ites.run(X_train, y_train)

        bestsol = ites.get_best()

        return bestsol


    def cross_val(self, X_train, y_train, X_test, y_test):
        keys, values = zip(*self._search_params.items())
        hyperconfigs = [dict(zip(keys, v)) for v in product(*values)]

        for hp in hyperconfigs:
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

            ites    = self._create_regressor(**hp)
            bestsol = self._eval(ites, X_train, y_train)

            rmse_test = RMSE(bestsol.it, X_test, y_test)
            mae_test  = MAE(bestsol.it, X_test, y_test)
            nmse_test = NMSE(bestsol.it, X_test, y_test)

            expr = bestsol.it.to_str()

            self._save_csv(hp, rmse_test, mae_test, nmse_test, expr)

            self._tested_params[tuple(hp.items())] = rmse_test
            self._tested_return[tuple(hp.items())] = (rmse_test, mae_test, nmse_test, expr)

        self.get_best()

        return self.best_score, self.best_params, self.results_best_params


    def get_best(self):

        assert len(self._tested_params.items()) != 0, \
            'get_best() being called without executing cross_val!'

        best_params      = min(self._tested_params, key=lambda key: self._tested_params[key])

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

    columns = ['dataset','best_params','RMSE_test','NMSE_test','MAE_test','Expression','Fold','Rep']
    
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

                best_score, best_params, results_best_params = grid.cross_val(Xtrain, ytrain, Xtest, ytest)
                
                grid.print_info()   
                
                rmse_test, nmse_test, mae_test, expr = results_best_params

                results['dataset'].append(ds)
                results['best_params'].append(best_params)
                results['NMSE_test'].append(nmse_test)
                results['RMSE_test'].append(rmse_test)
                results['MAE_test'].append(mae_test)
                results['Expression'].append(expr)
                results['Fold'].append(fold)
                results['Rep'].append(rep)

                df = pd.DataFrame(results)
                df.to_csv(fname, index=False)

    print('done')