#!/bin/bash

#SBATCH --job-name=mlsEvo             #This is the name of your job
#SBATCH --cpus-per-task=1             #This is the number of cores reserved
#SBATCH --mem-per-cpu=2G              #This is the memory reserved per core.
#Total memory reserved: 2GB

#SBATCH --time=1:00:00               #This is the time that your task will run
#SBATCH --qos=6hours                 #You will run in this queue

# Paths to STDOUT or STDERR files should be absolute or relative to current working directory
#SBATCH --output=arrayJob/mlsEvo.o_%A_%a
#SBATCH --error=arrayJob/mlsEvo.e_%A_%a

#You selected an array of jobs from 0 to 29 with 30 simultaneous jobs
#SBATCH --array=0-29                                %30
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=simon.vanvliet@unibas.ch        #You will be notified via email when your task ends or fails

#This job runs from the current working directory


#Remember:
#The variable $TMPDIR points to the local hard disks in the computing nodes.
#The variable $HOME points to your home directory.
#The variable $SLURM_JOBID stores the ID number of your job.


#load your required modules below
#################################


#export your required environment variables below
#################################################
conda activate mls_env

#add your command lines below
#############################
python mlsFig_evolutionRuns_SLURM.py $SLURM_ARRAY_TASK_ID $HOME