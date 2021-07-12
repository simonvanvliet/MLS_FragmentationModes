library(tidyverse)

df1 <- readRDS(here::here("R_figure_code", "mutationalMeltdown", "data_meltdown_RDS"))
df2 <- readRDS(here::here("R_figure_code", "mutationalMeltdown", "data_meltdown_more_replicates_RDS"))
df <- bind_rows(df1, df2) # df2 contains extra replicate runs, but the same parameters as df1

plot_df <- df %>% 
  group_by(indv_NType, gr_SFis, offspr_frac, offspr_size) %>% 
  summarize(max_mu = mean(log(maxMu), na.rm = TRUE))

# Nearest neighbour averaging ---------------------------------------------
delta_size <- (plot_df$offspr_size %>% unique)[2]-(plot_df$offspr_size %>% unique)[1]
delta_frac <- (plot_df$offspr_frac %>% unique)[2]-(plot_df$offspr_frac %>% unique)[1]
avg <- list(); i <- 0; for (slope in c(0, 0.1, 2, 4)) {
  for (nr_types in 1:4) {
    i <- i + 1
    
    filtered_data <- plot_df %>% 
      filter(indv_NType == nr_types, gr_SFis == slope)
    averaged_data <- filtered_data
    averaged_data$avg_max_mu <- 999
    
    for(j in 1:nrow(filtered_data)){
      neighbours <- filtered_data %>% 
        filter(offspr_size <= filtered_data[j,]$offspr_size + delta_size,
               offspr_size >= filtered_data[j,]$offspr_size - delta_size,
               offspr_frac <= filtered_data[j,]$offspr_frac + delta_frac,
               offspr_frac >= filtered_data[j,]$offspr_frac - delta_frac)
      
      averaged_data[j,]$avg_max_mu <- ifelse(is.nan(averaged_data[j,]$max_mu), 
                                             NA, 
                                             mean(neighbours$max_mu, na.rm = TRUE))
    }
    
    avg[[i]] <- averaged_data
    
    print(i)
  }
}
avg <- bind_rows(avg)

saveRDS(object = avg, file = here::here("R_figure_code", "mutationalMeltdown", "slope_neighbour_avg_RDS"))
