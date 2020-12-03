#!/bin/bash

#SBATCH --job-name=vVlietMutMeltdownNew     #This is the name of your job
#SBATCH --cpus-per-task=20                  #This is the number of cores reserved
#SBATCH --mem-per-cpu=2G                    #This is the memory reserved per core.
#Total memory reserved: 40GB
#SBATCH --nodes=1              # number of compute nodes


#SBATCH --time=168:00:00        #This is the time that your task will run
#SBATCH --qos=1week           #You will run in this queue

# Paths to STDOUT or STDERR files should be absolute or relative to current working directory
#SBATCH --output=vVlietMutMeltErrNew     #This is the joined STDOUT and STDERR file
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=simon.vanvliet@unibas.ch        #You will be notified via email when your task ends or fails

#This job runs from the current working directory


#Remember:
#The variable $TMPDIR points to the local hard disks in the computing nodes.
#The variable $HOME points to your home directory.
#The variable $SLURM_JOBID stores the ID number of your job.


#load your required modules below
#################################
ml Python/3.7.4-GCCcore-8.3.0
source $HOME/mls/bin/activate

#export your required environment variables below
#################################################


#add your command lines below
#############################
python mlsFig_mutationalMeltdown.py
