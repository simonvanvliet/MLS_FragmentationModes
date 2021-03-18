library(tidyverse)

df1 <- readRDS(here::here("R_figure_code", "mutationalMeltdown", "data_meltdown_migr_RDS"))
df2 <- readRDS(here::here("R_figure_code", "mutationalMeltdown", "data_meltdown_migr_more_replicates_RDS"))

df <- bind_rows(df1, df2)

plot_df <- df %>% 
  group_by(indv_NType, indv_migrR, offspr_frac, offspr_size) %>% 
  summarize(max_mu = mean(log(maxMu), na.rm = TRUE),
            pop_size = mean(NTot, na.rm = TRUE),
            nr_groups = mean(NGrp, na.rm = TRUE))
#summarize(max_mu = max(log(maxMu), na.rm = TRUE))

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
    }
    
    avg[[i]] <- averaged_data
    
    print(i)
  }
}
avg <- bind_rows(avg)

saveRDS(object = avg, file = here::here("R_figure_code", "mutationalMeltdown", "migr_neighbour_avg_RDS"))


avg$indv_migrR <- as.character(avg$indv_migrR)
avg[avg$indv_migrR == "0",]$indv_migrR <- "0.0"
avg$indv_migrR <- factor(avg$indv_migrR, levels = c("0.0", "0.01", "0.1","1", "5"))

avg %>% 
  mutate(avg_max_mu = ifelse(avg_max_mu < -10, -10, avg_max_mu)) %>% 
  ggplot(aes(x = offspr_size, y = offspr_frac, fill = avg_max_mu)) +
  geom_tile() +
  facet_grid(indv_migrR ~ indv_NType) +
  labs(x = expression("Frac. offsp. size ("*italic(s)*")"),
       y = expression("Frac. offsp. number ("*italic(n)*")"),
       fill = "Maximum\nmutation\nrate\n(log)") +
  cowplot::theme_cowplot() +
  theme(aspect.ratio = 1,
        strip.background = element_blank())+
  guides(fill = element_blank()) +
  scale_fill_viridis_c() +
  cowplot::panel_border() +
  facet_grid(glue::glue('italic(nu)*" = {indv_migrR}"') ~ glue::glue('italic(m)*" = {indv_NType}"'), 
             labeller = label_parsed) +
  scale_x_continuous(breaks = c(0, 0.25, 0.5), labels = c("0", "0.25", "0.5")) +
  scale_y_continuous(breaks = c(0, 0.5, 1), labels = c("0", "0.5", "1"))

#ggsave(filename = here::here("FigureData", "mutationalMeltdown", "meltdown_migr_neighbour_avg.pdf"),
#       height = 7, width = 8)
