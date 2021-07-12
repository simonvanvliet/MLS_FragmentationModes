library(reticulate)
library(tidyverse)

#change next line to location of conda python environment (see "conda env list" in terminal)
use_condaenv(condaenv = "mls_R_env",required = TRUE)

#import pandas
pd <- import("pandas")

#load data from python datafile
files <- dir(here::here("python_data_rev", "mutMeltNoDensDep_python"), pattern = ".pkl")

output <- list(); i <- 1
for(file in files){
  filePath <- here::here("python_data_rev", "mutMeltNoDensDep_python", file)
  df <- pd$read_pickle(filePath) #dataframe 
  df$file <- file
  output[[i]] <- df
  i <- i+1
}
output <- bind_rows(output)

saveRDS(output, file = here::here("R_figure_code", "Rev_mutationalMeltdownNoDensDep", "data_meltdown_noDensDep_RDS"))
