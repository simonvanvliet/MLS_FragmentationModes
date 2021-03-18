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
      
      averaged_data[j,]$avg_max_mu <- mean(neighbours$max_mu, na.rm = TRUE)
    }
    
    avg[[i]] <- averaged_data
    
    print(i)
  }
}
avg <- bind_rows(avg)

saveRDS(object = avg, file = here::here("R_figure_code", "mutationalMeltdown", "slope_neighbour_avg_RDS"))
avg <- readRDS(file = here::here("R_figure_code", "mutationalMeltdown", "slope_neighbour_avg_RDS"))

avg$gr_SFis <- as.character(avg$gr_SFis)
avg[avg$gr_SFis == "0",]$gr_SFis <- "0.0"
avg$gr_SFis <- factor(avg$gr_SFis, levels = c("0.0", "0.1", "2","4"))


avg %>% 
  mutate(avg_max_mu = ifelse(avg_max_mu < -10, -10, avg_max_mu)) %>% 
  ggplot(aes(x = offspr_size, y = offspr_frac, fill = avg_max_mu)) +
  geom_tile() +
  facet_grid(gr_SFis ~ indv_NType) +
  labs(x = expression("Frac. offsp. size ("*italic(s)*")"),
       y = expression("Frac. offsp. number ("*italic(n)*")"),
       fill = "Maximum\nmutation\nrate\n(log)") +
  cowplot::theme_cowplot() +
  theme(aspect.ratio = 1,
        strip.background = element_blank())+
  guides(fill = element_blank()) +
  scale_fill_viridis_c() +
  cowplot::panel_border() +
  facet_grid(glue::glue('italic(S)*" = {gr_SFis}"') ~ glue::glue('italic(m)*" = {indv_NType}"'), 
             labeller = label_parsed) +
  scale_x_continuous(breaks = c(0, 0.25, 0.5), labels = c("0", "0.25", "0.5")) +
  scale_y_continuous(breaks = c(0, 0.5, 1), labels = c("0", "0.5", "1"))

#ggsave(filename = here::here("FigureData", "mutationalMeltdown", "meltdown_slope_neighbour_avg.pdf"),
#       height = 7, width = 8)

