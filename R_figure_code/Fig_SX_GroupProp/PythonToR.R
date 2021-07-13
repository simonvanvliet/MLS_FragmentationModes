library(reticulate)
library(tidyverse)

#change next line to location of conda python environment (see "conda env list" in terminal)
use_condaenv(condaenv = "mls_R_env", conda="/Applications/anaconda3/condabin/conda",required = TRUE)

#import pandas
pd <- import("pandas")

#load data from python datafile
files <- dir(here::here("python_data_rev", "MLSGroupProp"), pattern = ".pkl")

output <- list(); i <- 1
for(file in files){
  filePath <- here::here("python_data_rev", "MLSGroupProp", file)
  df <- pd$read_pickle(filePath) #dataframe 
  if(nrow(df)>0){
    df$file <- file
    output[[i]] <- df
    i <- i+1
  }
}
output <- bind_rows(output)

saveRDS(output, file = here::here("R_figure_code", "Fig_SX_GroupProp", "data_groupProp_RDS"))
