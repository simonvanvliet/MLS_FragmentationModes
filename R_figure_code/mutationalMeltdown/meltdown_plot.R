library(tidyverse)

df1 <- readRDS(here::here("RcodePlots", "mutationalMeltdown", "data_meltdown_RDS"))
df2 <- readRDS(here::here("RcodePlots", "mutationalMeltdown", "data_meltdown_more_replicates_RDS"))
df <- bind_rows(df1, df2) # df2 contains extra replicate runs, but the same parameters as df1

plot_df <- df %>% 
  group_by(indv_NType, gr_SFis, offspr_frac, offspr_size) %>% 
  summarize(max_mu = mean((maxMu), na.rm = TRUE))

plot <- plot_df %>% 
  filter(indv_NType == 1,
         gr_SFis == 0) %>% 
  ggplot(aes(x = offspr_size, y = offspr_frac)) +
  geom_tile(aes(fill = max_mu)) +
  scale_fill_viridis_c() +
  labs(x = expression("Frac. offsp. size ("*italic(s)*")"),
     y = expression("Frac. offsp. number ("*italic(n)*")"),
     fill = "Max. mutation rate") +
  cowplot::theme_cowplot() +
  theme(aspect.ratio = 1,
        strip.background = element_blank(),
        legend.direction = "horizontal", 
        legend.position = c(0.55, 0.15),
        legend.key.height = unit(0.12, "in"),
        legend.title = element_text(size = 10),
        legend.text = element_text(size = 9)) +
  guides(fill = guide_colourbar(title.position = "top"))

saveRDS(plot, file = here::here("RcodePlots", "mutationalMeltdown", "meltdown_plot_RDS"))
