#própria classe de gridsearch criada por mim

import numpy  as np
import pandas as pd

import os.path

from itertools                 import product
import collections #Dicionários que se lembram da ordem de inserção

class Gridsearch:

    def __init__(self):
        
        #Parâmetros fixados, serão usados em todos
        self._default_params = {
            #poplen
            #generations
            #funcs
        }

        self._search_params  = {
            #maxterms
            #limite de expoentes
        }
        
        #Note que search_params UNIÃO default_params deve CONTER todos os parâmetros obrigatórios do modelo

        #guardará os scores dos parametros para teste, onde a key desse
        #dicionário é um conjunto de tuplas (key, value) dos valores tes-
        #tados para cada hyperparâmetro.
        self._tested_params = { } #Salva os dados do treino (validação cruzada)
        self._test_score    = collections.OrderedDict() #Salva os dados do teste (para a partição de teste)

        self.best_score     = np.inf

        #aqui é um dicionário com os parâmetros utilizados para o
        #melhor parâmetro
        self.best_params  = { }
        self.worst_params = { }


    def _create_regressor(self, **kwargs):

        #cria um modelo com os parâmetros padrões (default), mas modificando
        #aqueles passados como kwargs (para poder variar isso), então faz o 
        #fit com a base passada para treino e faz a validação com a base
        #passada para validação, e entõ retorna a métrica utilizada.

        #uso interno, não faço verificação dos argumentos aqui

        _params = {**self._default_params}

        for key, value in kwargs.items():
            _params[key] = value

        regressor = self.evalfunc(**_params)

        return regressor


    def _eval(self, regressor, X_train, y_train, X_val, y_val):
        
        regressor = self.fitter(regressor, X_train, y_train)

        y_pred = regressor.predict(X_val)

        return self.metric(y_val, y_pred)


    def cross_val(self, X_train, y_train, X_test, y_test):
        
        #recebe uma métrica e informação se deve minimizar ou maximizar o
        #valor dela para determinar o melhor resultado. Já possui uma métrica
        #padrão, e nessa definição de crossval nós temos o minimize de acordo 
        #com a métrica padrão.

        #salvando informações de avaliação dos resultados para evitar
        #muitos parâmetros em chamada e possibilitar obter essa informação
        #depois se necessário
        self.metric = metric
        
        #salvando os valores passados caso queiramos pegar eles depois 
        #e para diferenciar dos folds aqui
        self.X_train = X_train
        self.y_train = y_train

        #Valores para teste, que NÃO devem ser misturados com os de treino
        self.X_test = X_test
        self.y_test = y_test

        #é um gridsearch com crossvalidação

        #o asterisco faz o zip reverso. Aqui criamos todas as configurações
        #de hyperparâmetros que queremos variar no nosso gridsearch
        keys, values = zip(*self._search_params.items())
        hyperconfigs = [dict(zip(keys, v)) for v in product(*values)]
        
        for hp in hyperconfigs:

            #TODO: RECONHECER SE JÁ RODOU ESSA CONFIGURAÇÃO ESPECÍFICA E LER OS DADOS CASO SIM
            #no grid log tenho que ter a key pro evolution_log!

            #Crio um regressor para ser utilizado em todos os folds
            #um detalhe: quando fazemos um fit nós perdemos informação do
            #fit anterior, por isso é válido fazer isso.
            regressor = self._create_regressor(**hp)

            #train index terá os índices para o treino, e o val para a validação.
            #faço cópia do X para garantir que não vou modificar o original
            # na hora de gerar o conjunto normalizado
            X_train, y_train = self.X_train[train_index], self.y_train[train_index]
            X_val,   y_val   = self.X_train[val_index],   self.y_train[val_index]

            hp_score = score self._eval(regressor, X_train, y_train, X_val, y_val)

            #transforma os hyperparâmetros testados em tuplas para poderem
            #ser utilizadas como key no dicionário. Posteriormente fica mais
            #fácil utilizar assim para converter a melhor combinação de parâ-
            #metros testada para dicionário usando dict(key)

            self._tested_params[tuple(hp.items())] = hp_score

            #Rodando o teste ---------------------------------------------------
            score = self._eval(regressor, self.X_train, self.y_train, self.X_test, self.y_test)

            #Salvando o teste desse regressor
            self._test_score[tuple(hp.items())] = score

        #aqui o minimize é utilizado. Ao fazer isso, os default são atualizados
        #para utilizar os melhores!

        _, best_config = self.get_best()

        return best_config


    def get_best(self):

        assert len(self._tested_params.items()) != 0, \
            'get best utilizado sem fazer a busca!'

        best_params      = min(self._tested_params, key=lambda key: self._tested_params[key])

        self.best_score  = self._tested_params[best_params]
        self.best_params = dict(best_params)
        
        return self.best_score, self.best_params  


    def print_info(self):

        print("--- Gridsearch ranges ---")
        for key, value in self._search_params.items():
            print(key + " range: " + str(value))

        print('-- best final result --')
        print("parameters: " + str(self.best_params))
        print("score: " + str(self.best_score))
        print("-------------------------")


    def save_csv(self, name=None):

        #salvar os resultados num arquivo.
        #irá salvar os parâmetros (apenas os que foram variados) e adicionará
        #uma última coluna com o valor médio da métrica (valor médio pois é
        #feito um crossvalidation)

        fname = './grid_log/' + (self.evalfunc.__name__ if name==None else name) + '.csv'

        columns = [key for key in self._search_params.keys()] + ['hp_score']

        results = {c:[] for c in columns}

        if os.path.isfile(fname):
            resultsDF = pd.read_csv(fname)
            results   = resultsDF.to_dict('list')

        for attempt, score in self._tested_params.items():
        
            for key, value in dict(attempt).items():
                results[key].append(value)

            results['hp_score'].append(score)

        df = pd.DataFrame(results)
        df.to_csv(fname, index=False)