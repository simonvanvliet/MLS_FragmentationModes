# Readme

## Author: Simon van Vliet & Gil Henriques
vanvliet@zoology.ubc.ca
henriques@zoology.ubc.ca

## The following Code is supplied:

### Code implementing main model
#### MlsGroupDynamics_main.py
Implementation of main Multilevel selection model, without trait evolution

No direct user access required -> run model with code described below

#### MlsGroupDynamics_evolve.py
Implementation of main Multilevel selection model, with trait evolution at individual level

No direct user access required -> run model with code described below

#### MlsGroupDynamics_evolve_groups.py
Implementation of main Multilevel selection model, with trait evolution at group level

No direct user access required -> run model with code described below

#### MlsGroupDynamics_utilities.py
Collection of utility functions required for model solving

No direct user access required -> run model with code described below

### Code to run model single time
#### singleRun.py
Runs main model single time and plots results

Code provides explanation of model parameters

#### singleRunEvolution.py
Runs main model with evolution of traits at individual level a single time and plots results

### Code to explore parameter space
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

### Batch utility scripts
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

### plotting utility scripts
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
