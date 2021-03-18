# Readme
# Code for: "Multilevel selection favors fragmentation modes that maintain cooperative interactions in multispecies communities"

## Author: Gil Henriques, Simon van Vliet, & Michael Doebeli
henriques@zoology.ubc.ca
vanvliet@zoology.ubc.ca
doebeli@zoology.ubc.ca


This repository contains the python code required to run the model (in Folder python_model_code) and R code required to make the figures (in folder R_figure_code).

## The following R code is supplied:
- RDS data files that contain figure data (imported and converted from python code output)
- R scripts to recreate the figures
Files and scripts for each figure are collected in individual folders.

The output of the python model code was imported in R using the reticulate library, using the following code:

```
library(reticulate)
use_condaenv("path_to_conda_env")
pd <- import("pandas")
filePath <- here::here("path_to_datafile", "data_file_name.pkl")
df <- pd$read_pickle(filePath) #dataframe
saveRDS(df, file = here::here("path_to_store", "file_name_RDS"))
```

## The following python Code is supplied:
The code has been tested using the Conda environment specified in mls_env.yml

### Code required to reproduce manuscript figure (in folder "manuscriptFigureCode")
Each script produces the data for one or more figure panels

Data is exported as Pandas data frame and can be plotted with the R scripts that are provided along cite the manuscript.

### Code implementing main model (in folder "mainCode")
#### mainCode/MlsGroupDynamics_main.py
Implementation of main Multilevel selection model, without trait evolution

No direct user access required -> run model with code described below

#### mainCode/MlsGroupDynamics_evolve.py
Implementation of main Multilevel selection model, with trait evolution at individual level

No direct user access required -> run model with code described below

#### mainCode/MlsGroupDynamics_evolve_groups.py
Implementation of main Multilevel selection model, with trait evolution at group level

No direct user access required -> run model with code described below

#### mainCode/MlsGroupDynamics_utilities.py
Collection of utility functions required for model solving

No direct user access required -> run model with code described below

#### mainCode/MlsGroupDynamics_plotUtilities.py
Collection of utility functions required for model plotting

No direct user access required -> run model with code described below

### Code to run model single time (in folder "exploreModelCode")
#### singleRun.py
Runs main model single time and plots results

Code provides explanation of model parameters

#### singleRunEvolution.py
Runs main model with evolution of traits at individual level a single time and plots results

### Code to explore parameter space (in folder "exploreModelCode")
#### MlsGroupDynamics_scanStates.py
Scans 2D parameter space (fractional size of offspring, and fraction of parent assigned to offspring)

Code can scan up to 3 additional parameters

Results are stored in single file on disk

Plot result with: plotScanStates.py
Export results to CSV with exportScanStates.py

Code supports parallel cores

#### MlsGroupDynamics_scanTransects.py
Scans 1D perimeter of 2D parameter space (fractional size of offspring, and fraction of parent assigned to offspring).

Perimeter location 0 corresponds to upper left corner of 2D space (0,1)
Perimeter location 0.5 corresponds to rigth corner of 2D space (0.5,0.5)
Perimeter location 1 corresponds to lower left corner of 2D space (0,0)

Code can scan up to 3 additional parameters

Results are stored in single file on disk

Plot result with: plotParScanTransect.py
Export results to CSV with exportParScanTransect.py

Code supports parallel cores

#### MlsGroupDynamics_scanMutationalMeltdown.py
Scans 2D parameter space (fractional size of offspring, and fraction of parent assigned to offspring)

For each location in parameter space the maximal mutational load is calculated. i.e. the maximum mutation rate at which the population can maintain a non-zero density.

Code can scan up to 3 additional parameters

Results are stored in single file on disk

Plot result with: plotMutationalMeltdown.py
Export results to CSV with exportMutationalMeltdown.py

Code supports parallel cores

#### MlsGroupDynamics_scan2D.py			
Scans 2D parameter space (fractional size of offspring, and fraction of parent assigned to offspring)

Code supports only a single set of other model parameters at a time

Results are stored in single file on disk

Plot result with: plotParScan.py

Code supports parallel cores

### Batch utility scripts (in folder "exploreModelCode")
#### MlsBatch.py		
Depreciated: use MlsGroupDynamics_scanStates instead

Performs multiple runs of MlsGroupDynamics_scan2D, varying model parameters.

Results for each run individually stored on disk.

Plot results with plotAllScan.py

Code supports parallel cores (but inefficiently)

#### MlsEvoBatch.py		
Performs multiple runs of evolution model with traits varying at individual level, varying model parameters and initial state.

Results for each run individually stored on disk.

Plot results with plotEvolutionBatch.py and plotEvolutionBatchMovie.py

Code supports parallel cores

#### MlsEvoBatchGroup.py		
Performs multiple runs of evolution model with traits varying at group level, varying model parameters and initial state.

Results for each run individually stored on disk.

Plot results with plotEvolutionBatch.py and plotEvolutionBatchMovie.py

Code supports parallel cores

### plotting utility scripts (in folder "exploreModelCode")
#### MlsGroupDynamics_plotUtilities.py
Contains main plotting function used by various pieces of code

No direct user access required -> run model with code described above

#### plotSingleRun.py
Plots outcome of single model run

No direct user access required -> run model with code described above

#### plotSingleRun_evolution.py
Plots outcome of single model run, where traits evolve

No direct user access required -> run model with code described above

#### makeEvolutionMovie.py
Makes movie of single model run, where traits evolve

No direct user access required -> run model with code described above
