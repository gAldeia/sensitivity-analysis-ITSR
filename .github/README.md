Sensitivity Analysis of Interaction-Transformation Evolutionary Algorithm for Symbolic Regression
======

Repository containing the code for performing and saving a sensitivity analysis of the hyper-parameters for the algorithm based on the Interaction-Transformation (IT) representation, as well as a python notebook to plot results helping visualizing the results.

The aim is to answer the following questions:

  * Does exist an unique set of fixed parameters that frequently dominates other combinations?
  * Are goodness-of-fit and size of the model conflicting objectives?
  * How the goodness-of-fit varies when we change some of the parameters?

### Paper abstract

> **The balance between approximation error and model complexity is an important trade-off for Symbolic Regression algorithms. This trade-off is achieved by means of specific operators for bloat control, modified operators,  limits to the size of the generated expressions and multi-objective optimization.**
> **Recently, the representation Interaction-Transformation was introduced with the goal of limiting the search space to simpler expressions, thus avoiding bloating. This representation was used in the context of an Evolutionary Algorithm in order to find concise expressions resulting in small approximation errors competitive with the literature. Particular to this algorithm, two parameters control the complexity of the generated expression. This paper investigates the influence of those parameters w.r.t. the goodness-of-fit. Through some extensive experiments, we find that  _max_terms_ is more important to control goodness-of-fit but also that there is a limit to which increasing its value renders any benefits. Also, since _expolim_ has a smaller influence to the results it can be set to a default value without impacting the final results.**

> **Keywords: parametric analysis, evolutionary algorithms, symbolic regression.**
  
Aldeia, G. S. I. and de França, F. O. (2020). A parametric study of interactrion-transformation evolutionary algorithm for symbolic regression. In _2020 IEEE World Congress on Computational Intelligence (WCCI)_.

Installation and Usage
------

Clone or download this repository. We recomend to clone by using

    > git clone --depth=1 --branch=master https://github.com/gAldeia/sensitivity-analysis-ITSR.git sensitivity-analysis-ITSR    

The original version (used in the paper) is in the __./src/__ folder.

A newer version is in __./src_2.0/__.

Below there's Instructions of how to use the ITEA for both versions.


### Symbolic Regression

To perform a symbolic regression, there are two main steps: _i)_ create the regressor and _ii)_ run it. Below are examples of how to create and perform the symbolic regression, with sugestions of values for each parameter. __The usage of version 2.0 is encouraged__.

For the original version (__./src/__), copy all files that starts with lowercase to the same folder as your script, import the _ites_, set the parameters, create the class and run it with any dataset:

```python
import itea  as sr
import numpy as np

from sklearn import datasets


degree_range = 2
max_terms    = 10
pop_len      = 100
gens         = 300
funcs        = [ #list of tuples, containing (func name, function)
    ("id"      , lambda x: x),
    ("sin"     , np.sin), 
    ("cos"     , np.cos),        
    ("tanh"    , np.tanh),
    ("sqrt.abs", lambda x: np.sqrt(np.absolute(x))),
    ("log"     , np.log), 
    ("exp"     , np.exp),
 ]

# custom functions should be based on numpy, to allow
# the application over an array (if the function works on a single value,
# create a lambda function to map it to an array)

itea = sr.ITEA(
    pop_len      = pop_len,      # size of the population
    gens         = gens,         # number of generations
    funcs        = funcs,        # transformation functions to be used
    degree_range = degree_range, # max and min degree allowed in the IT,
    max_terms    = max_terms,    # maximum number of terms allowed,
    label        = [],           # labels to the features of the problem,
    log          = None          # file name to save the evolution log
)

X, y = datasets.load_diabetes(return_X_y=True)

itea.run(X, y)

# retrieve the best solution
best_function = itea.get_best()

# print the equivalent equation
print(best_function.it.to_str())

# predicts a new value
pred = best_function.it.predict(X[0:2, :])
print(pred, y[0:2])

# prints the fitness
print(best_function.fitness)
```

The version 2.0 allows the configuration of the degree range and the maximum and minimum number of terms in a more flexible way than the original version. In this usage example, we will create an dictionary of the parameters, instead of creating each one individually. 

Also, you can pass the model used to adjust the coefficients (should be a model that returns the coefficients and the bias, as the linear models from sklearn does).

There are some minor differences between the parameters (i.e. the transformation functions list). The version 2.0 uses numpy arrays more intensively to provide a better performance, although it does not use pathos for multiprocessing (the old versiond does, but shows bugs when using the linear model with mutiple jobs, while the newer version doesn't have this limitation). A performance comparision between the two versions is planned.

Although python does not support type check, in version 2.0, the libraries _typing_ and _nptyping_ are used to provide clearer explanation of the code.

```python
import itea  as sr
import numpy as np

from sklearn import linear_model
from sklearn import datasets


params = {
    'popsize'  : 150,
    'gens'     : 100, 
    'minterms' : 1,
    'maxterms' : 4,
    'model'    : linear_model.LinearRegression(n_jobs=-1),
    'funs'     : { #(dictionary of functions)
        "id"      : lambda x: x,
        "sin"     : np.sin, 
        "cos"     : np.cos,        
        "tanh"    : np.tanh,
        "sqrt.abs": lambda x: np.sqrt(np.absolute(x)),
        "log"     : np.log, 
        "exp"     : np.exp,
    },
    'expolim'  : (-2, 2)
}

itea = sr.ITEA(**params)


X, y = datasets.load_diabetes(return_X_y=True)

# the verbose prints informations of convergenge,
# that can be saved in the log file as well
best_function = itea.run(X, y, log=None, verbose=True)

# now print is overloaded
print(best_function)

# predicts a new value
pred = best_function.predict(X[0:2, :])
print(pred, y[0:2])

# prints the fitness
print(best_function.fitness)
```

### Gridsearch

To execute the experimental setup used in the paper, just run the __./src/Gridsearch.py__ with a python at version 3.7 or higher. The code is made to be able to continue tests if an execution is interrupted, based on the saved files:
- (FOLDER) __/evolution_log/__:
    - Save files that are used to plot convergence graphics;
- (FOLDER) __/grid_log/__:
    - Save information about every combination of hyper-parameters being tested for each dataset
- (FILE) __resultsregression.csv__:
    - Saves the best result found for each dataset, on each fold, on each repetition.
    
    

### Repository content

The content of this repository is organized as follow.

| Folder   | Description                                                                                                                                                                                                                  |
|----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Datasets | Folder containing all the datasets used in the experiments, created by splitting the original dataset into 5 folds. Each fold is then saved on a file to allow reproducibility of the experiments under the same conditions. |
| Results  | Folder containing a jupyter notebook used to generate the plots used in the paper, obtain the p-value, and another useful informations. Every plot generated is there.                                                     |
| Src      | Folder containing the source code to the symbolic regression algorithm (ITEA), as well as the Gridsearch applyied.                                                                                                           |
| Src_2.0  | Folder with a compact version of the source code (all within one file), to provide a easy to use version of the studied algorithm.                                                                                           |

### Pre requisites

The following libraries are used in the code:

| Src         | Src_2.0  |
|-------------|----------|
| numpy       | numpy    |
| pandas      | -        |
| os          | -        |
| glob        | -        |
| re          | -        |
| time        | -        |
| collections | -        |
| sklearn     | sklearn  |
| itertools   | -        |
| copy        | -        |
| math        | -        |
| pathos      | -        |
| -           | typing   |
| -           | nptyping |
