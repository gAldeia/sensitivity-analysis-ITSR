Sensitivity Analysis ITSR
======

Repository containing the code for performing and saving a sensitivity analysis of the hyper-parameters for the algorithm based on the Interaction-Transformation (IT) representation, as well as a python notebook to plot results helping visualizing the results.

The aim is to answer questions about the hyper-parameters, and propose a method to quantify the answer of the following:

  * Does exist an unique set of fixed parameters that frequently dominates other combinations?
  * Are goodness-of-fit and size of the model conflicting objectives?
  * How the goodness-of-fit varies when we change some of the parameters?

### Paper abstract

> **The balance between approximation error and model complexity is an important trade-off for Symbolic Regression algorithms. This trade-off is achieved by means of specific operators for bloat control, modified operators,  limits to the size of the generated expressions and multi-objective optimization.**
> **Recently, the representation Interaction-Transformation was introduced with the goal of limiting the search space to simpler expressions, thus avoiding bloating. This representation was used in the context of an Evolutionary Algorithm in order to find concise expressions resulting in small approximation errors competitive with the literature. Particular to this algorithm, two parameters control the complexity of the generated expression. This paper investigates the influence of those parameters w.r.t. the goodness-of-fit. Through some extensive experiments, we find that  _max_terms_ is more important to control goodness-of-fit but also that there is a limit to which increasing its value renders any benefits. Also, since _expolim_ has a smaller influence to the results it can be set to a default value without impacting the final results.**

> **Keywords: parametric analysis, evolutionary algorithms, symbolic regression.**
  
Installation and Usage
------

Clone or download this repository.

just run the /src/Gridsearch.py with a python at version 3.5 or higher. The code is made to be able to continue tests if an execution is interrupted, based on the saved files:
- __(FOLDER) /evolution_log/__:
    - Save files that are used to plot convergence graphics;
- __(FOLDER) /grid_log/__:
    - Save information about every combination of hyper-parameters being tested for each dataset
- __(FILE) resultsregression.csv__:
    - Saves the best result found for each dataset, on each fold, on each repetition.


### Repository content

The content of this repository is organized as follow.

| Folder   | Description                                                                                                                                                                                                                  |
|----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Datasets | Folder containing all the datasets used in the experiments, created by splitting the original dataset into 5 folds. Each fold is then saved on a file to allow reproducibility of the experiments under the same conditions. |
| Results  | Folder containing a jupyter notebook used to generate the plots used in the article, obtain the p-value, and another useful informations. Every plot generated is there.                                                     |
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


License
------
