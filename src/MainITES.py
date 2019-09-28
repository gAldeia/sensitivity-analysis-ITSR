import pandas as pd
import glob # Search for files in directory
import re # Regular expressions
import time
import numpy as np
import os.path

from sklearn import linear_model, metrics

import ites as sr


def MAE(it, X, y):
    yhat = it.predict(X)
    return np.abs(yhat - y).mean()

def RMSE(it, X, y):
    yhat = it.predict(X)
    return np.sqrt(np.square(yhat - y).mean())

def NMSE(it, X, y):
    yhat = it.predict(X)
    return (np.square(yhat - y)/(yhat.mean()*y.mean())).mean()


if __name__ =='__main__':

    columns = [
        'dataset',
        'RMSE_train',
        'RMSE_test',
        'NMSE_train',
        'NMSE_test',
        'MAE_train',
        'MAE_test',
        'Time',
        'Expression',
        'Fold'
    ]
    
    #select which datasets to use
    datasets = ['airfoil', 'concrete', 'energyCooling', 'energyHeating', 'towerData', 'wineRed', 'wineWhite', 'yacht']    
    
    # arquivo com resultados
    fname = 'resultsregression.csv'
    results = {c:[] for c in columns}

    if os.path.isfile(fname):
        resultsDF = pd.read_csv(fname)
        results   = resultsDF.to_dict('list')


    funcs = [
        #linear
        ("id" , lambda x: x),
        #("abs", np.abs),
        #("max", lambda x: np.maximum(0.0, x)),
        #("min", lambda x: np.minimum(0.0, x)),

        #trigonometricas
        ("sin"  , np.sin), 
        ("cos"  , np.cos),        
        #("cosh" , np.cosh), #muito sensível
        #("sinh" , np.sinh), #muito sensível
        ("tanh" , np.tanh), #não contínua

        #não-lineares
        ("sqrt" , np.sqrt), #não contínua <0
        ("log"  , np.log),  #não contínua <0
        ("log1p", np.log1p),
        ("exp"  , np.exp),
    ]


    for train in datasets:
        print(train)
        for fold in range(5):
            dataset = np.loadtxt(f'../datasets/{train}-train-{fold}.dat', delimiter=',')
            Xtrain, ytrain = dataset[:, :-1], dataset[:, -1]
            
            dataset = np.loadtxt(f'../datasets/{train}-test-{fold}.dat', delimiter=',')
            Xtest, ytest = dataset[:, :-1], dataset[:, -1]

            maxrep = 6
            
            for rep in range(maxrep):

                logfile = f'evolution_log/{train}-{fold}-{rep}-100-log.txt'

                # Começar de onde parou de acordo com os logs
                if os.path.isfile(logfile):
                    continue

                ites = sr.ITES(pop_len=100, gens=100, funcs=funcs, expolim=3, max_terms=5, log=logfile)

                start = time.time()

                ites.run(Xtrain, ytrain)
                
                tot_time = time.time() - start

                bestsol = ites.get_best()
                expr = bestsol.it.to_str()

                print(bestsol.fitness)
                
                rmse_train, rmse_test = RMSE(bestsol.it, Xtrain, ytrain), RMSE(bestsol.it, Xtest, ytest)
                nmse_train, nmse_test = NMSE(bestsol.it, Xtrain, ytrain), NMSE(bestsol.it, Xtest, ytest)
                mae_train,  mae_test  = MAE(bestsol.it, Xtrain, ytrain),  MAE(bestsol.it, Xtest, ytest)

                print(f'RMSE (Train): {rmse_train}, RMSE (Test): {rmse_test}, NMSE (Train): {nmse_train}, NMSE (Test): {nmse_test}, EXPR: {expr}')

                results['dataset'].append(train)
                results['RMSE_train'].append(rmse_train)
                results['RMSE_test'].append(rmse_test)
                results['NMSE_train'].append(nmse_train)
                results['NMSE_test'].append(nmse_test)
                results['MAE_train'].append(mae_train)
                results['MAE_test'].append(mae_test)
                results['Time'].append(tot_time)
                results['Expression'].append(expr)
                results['Fold'].append(fold)

                df = pd.DataFrame(results)
                df.to_csv(fname, index=False)