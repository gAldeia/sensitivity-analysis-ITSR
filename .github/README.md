Sensitivity Analysis ITSR
======

Repository containing the code for performing and saving a sensitivity analysis of the hyper-parameters for the algorithm based on the Interaction-Transformation (IT) representation, as well as a python notebook to plot results helping visualizing the results.

The aim is to answer questions about the hyper-parameters, and propose a method to quantify the answer of the following:
- Exists one set of fixed hyper-parameters that always (or almost always) dominate the others?
- Every hyper-parameter must be availed when testing a new dataset, or there's some that can be fixed?
- How the error varyies when we change each parameter?

  
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



### Pre requisites

The following libraries are used in the code:
* numpy
* pandas
* os
* glob
* re
* time
* collections
* sklearn
* itertools
* copy
* math
* pathos



License
------
