import numpy as np
import pandas as pd

import os.path
import sys
import time


from _gridsearch     import Gridsearch

from sklearn.model_selection import train_test_split
from itertools               import product



def regression_task(regressor_pack, X, y, name=None):

    # separo em treino-teste. a parte de treino será utilizada no Kfold
    # para otimização de hyperparâmetros (nisso ela será dividida em treino
    # e validação), e a parte de teste será utilizada para obter o resultado final
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    regressor, normalization = regressor_pack
    default_params, search_params, builder, fitter, varimp = regressor

    # crio o modelo. Cada modelo faz seu próprio ajuste de hiper parâmetros
    # e cross validation, pois cada um tem um jeito diferente.
    # Todos tem a mesma estrutura: criar classe, usar o crossval, pegar o resultado
    # Dentro da própria classe será feito a divisão em treino-validação
    grid = Gridsearch(builder, fitter, default_params, search_params, varimp)

    grid.cross_val(X_train, y_train, X_test, y_test, metric=r2_score, greater_is_better=True, normalize_mode=normalization)

    #Salvar um log com todos os dados
    #grid.save_csv(name)
    grid.print_info()
    
    (score_best, varimp_best, shap), (score_worst, varimp_worst), score_all = grid.test()
    
    print("test score: " + str(score_best))

    return score_best, score_worst, score_all, grid.best_params, varimp_best, shap


def run_test_regression():

    sampless        = [600]
    relevancy_rates = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] #Decidindo a quantidade de atributos relevantes
    modes           = ['clean', 'noisy'] #, 'random']
    
    regressor_pack = [
        (Knn_regressor, 'z_score'),
    ]

    #isso aqui é padrão. Para testes (não misturar os .csv) usar 10
    matrix_size = 100 #10, 25, 50, 100

    #Criando todas as coordenadas da matriz inferior. Aqui eu já assumo que ele vai ser flattened depois
    #lembrando que se no processing k=-1, k=0, etc. AQUI TEM QUE ESTAR IGUAL
    coords = np.array(list(zip(*np.tril_indices(matrix_size, k=-1))))
    
    #Testando cada modo para diferentes variações
    samples_behavior = list(product(modes, regressor_pack, sampless, relevancy_rates))

    #Nome do arquivo que ele vai ver se tem resultados já calculados e salvar os próximos
    fname = 'resultados_testes_matrixsize'+str(matrix_size)+'_shap_diag.csv'
    columns = ['mode', 'regressor', 'samples', 'relevancy_rate', 'relevancy_number', 'test_score_best', 'test_score_worst', 'time', 'best_params', 'varimp', 'shap', 'test_score_all']

    results = {c:[] for c in columns}

    resultsDF = pd.DataFrame(columns=columns)

    if os.path.isfile(fname):
        #Abre o arquivo e carrega caso ele exista
        resultsDF = pd.read_csv(fname)
        results   = resultsDF.to_dict('list')

    for mode, regressor_pack, samples, relevancy_rate in samples_behavior:

        #Evitando refazer repetidos e possibilitando retomar o teste
        if len(resultsDF[(resultsDF['mode']==mode) &
                         (resultsDF['regressor']==regressor_pack[0][2].__name__) &
                         (resultsDF['samples']==samples) &
                         (resultsDF['relevancy_rate']==relevancy_rate)])>0:
            print("already done")
            continue

        #FALTA LER X E Y AQUI!!!!

        log_name = "_".join([str(p) for p in [mode, regressor_pack[0][2].__name__, samples, relevancy_rate]])
        print(log_name)

        start = time.time()
        score_best, score_worst, score_all, best_params, varimp, shap = regression_task(regressor_pack, X, y, log_name)
        end = time.time()

        results['mode'].append(mode)
        results['regressor'].append(regressor_pack[0][2].__name__)
        results['samples'].append(samples)
        results['relevancy_rate'].append(relevancy_rate)
        results['relevancy_number'].append(int(len(coords)*relevancy_rate))
        results['test_score_best'].append(score_best)
        results['test_score_worst'].append(score_worst)
        results['time'].append(end-start)
        results['best_params'].append(best_params)
        results['varimp'].append(varimp)
        results['shap'].append(shap)
        results['test_score_all'].append(score_all)

        df = pd.DataFrame(results)
        df.to_csv(fname, index=False)


if __name__ == '__main__':

    run_test_regression()