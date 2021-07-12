library(tidyverse)

df1 <- readRDS(here::here("R_figure_code", "mutationalMeltdown", "data_meltdown_migr_RDS"))
df2 <- readRDS(here::here("R_figure_code", "mutationalMeltdown", "data_meltdown_migr_more_replicates_RDS"))

df <- bind_rows(df1, df2)

plot_df <- df %>% 
  group_by(indv_NType, indv_migrR, offspr_frac, offspr_size) %>% 
  summarize(max_mu = mean(log(maxMu), na.rm = TRUE),
            pop_size = mean(NTot, na.rm = TRUE),
            nr_groups = mean(NGrp, na.rm = TRUE))

# for the first row (migration = 0) we get the data from the mutational meltdown on slope 
df3 <- readRDS(here::here("R_figure_code", "mutationalMeltdown", "data_meltdown_RDS")) %>% 
  filter(indv_migrR == 0, gr_SFis == 0) 
df4 <- readRDS(here::here("R_figure_code", "mutationalMeltdown", "data_meltdown_more_replicates_RDS")) %>% 
  filter(indv_migrR == 0, gr_SFis == 0) 

df5 <- bind_rows(df3, df4) %>% 
  group_by(indv_NType, indv_migrR, offspr_frac, offspr_size) %>% 
  summarize(max_mu = mean(log(maxMu), na.rm = TRUE),
            pop_size = mean(NTot, na.rm = TRUE),
            nr_groups = mean(NGrp, na.rm = TRUE))
#summarize(max_mu = max(log(maxMu), na.rm = TRUE))

plot_df2 <- bind_rows(plot_df, df5)


delta_size <- (plot_df$offspr_size %>% unique)[2]-(plot_df$offspr_size %>% unique)[1]
delta_frac <- (plot_df$offspr_frac %>% unique)[2]-(plot_df$offspr_frac %>% unique)[1]
avg <- list(); i <- 0; for (migr in c(0, 0.01, 0.1, 1)) {
  for (nr_types in 1:4) {
    i <- i + 1
    
    filtered_data <- plot_df2 %>% 
      filter(indv_NType == nr_types, indv_migrR == migr)
    averaged_data <- filtered_data
    averaged_data$avg_max_mu <- 999
    
    for(j in 1:nrow(filtered_data)){
      neighbours <- filtered_data %>% 
        filter(offspr_size <= filtered_data[j,]$offspr_size + delta_size,
               offspr_size >= filtered_data[j,]$offspr_size - delta_size,
               offspr_frac <= filtered_data[j,]$offspr_frac + delta_frac,
               offspr_frac >= filtered_data[j,]$offspr_frac - delta_frac)
      
      averaged_data[j,]$avg_max_mu <- mean(neighbours$max_mu, na.rm = TRUE)
      
      averaged_data[j,]$avg_max_mu <- ifelse(is.nan(averaged_data[j,]$max_mu), 
                                             NA, 
                                             mean(neighbours$max_mu, na.rm = TRUE))
    }
    
    avg[[i]] <- averaged_data
    
    print(i)
  }
}
avg <- bind_rows(avg)

saveRDS(object = avg, file = here::here("R_figure_code", "mutationalMeltdown", "migr_neighbour_avg_RDS"))
