library(tidyverse)

df1 <- readRDS(here::here("R_figure_code", "mutationalMeltdown", "data_meltdown_RDS"))
df2 <- readRDS(here::here("R_figure_code", "mutationalMeltdown", "data_meltdown_more_replicates_RDS"))
df <- bind_rows(df1, df2) # df2 contains extra replicate runs, but the same parameters as df1

plot_df <- df %>% 
  group_by(indv_NType, gr_SFis, offspr_frac, offspr_size) %>% 
  summarize(max_mu = mean(log(maxMu), na.rm = TRUE))
  #summarize(max_mu = max(log(maxMu), na.rm = TRUE))

# This version makes a hull around the top 5 values of max_mu ---------

max_df <- plot_df %>% 
  ungroup() %>% 
  mutate(max_mu = ifelse(max_mu == -Inf, NA, max_mu)) %>% 
  mutate(max_mu = if_else(max_mu < -10, -10, max_mu)) %>% 
  mutate(m = indv_NType, S = gr_SFis) %>% 
  group_by(m, S) %>% 
  filter(!is.na(max_mu)) %>% 
  slice_max(order_by = max_mu, n = 5) %>% 
  slice(chull(offspr_size, offspr_frac))




# Plot --------------------------------------------------------------------
plot_df$gr_SFis <- as.character(plot_df$gr_SFis)
plot_df[plot_df$gr_SFis == "0",]$gr_SFis <- "0.0"
plot_df$gr_SFis <- factor(plot_df$gr_SFis, levels = c("0.0", "0.1", "2","4"))

max_df$S <- as.character(max_df$S)
max_df[max_df$S == "0",]$S <- "0.0"
max_df$S <- factor(max_df$S, levels = c("0.0", "0.1", "2","4"))

(plot <- plot_df %>% 
  mutate(max_mu = ifelse(max_mu == -Inf, NA, max_mu)) %>% 
  mutate(max_mu = if_else(max_mu < -10, -10, max_mu)) %>% 
  mutate(m = indv_NType, S = gr_SFis) %>% 
  ggplot(aes(x = offspr_size, y = offspr_frac)) +
  geom_tile(aes(fill = max_mu)) +
  labs(x = expression("Fractional offspring size ("*italic(s)*")"),
       y = expression("Fractional offspring number ("*italic(n)*")"),
       fill = "Maximum\nmutation\nrate\n(log)") +
  cowplot::theme_cowplot() +
  theme(aspect.ratio = 1,
        strip.background = element_blank())+
  guides(fill = element_blank()) +
  scale_fill_viridis_c() +
  cowplot::panel_border() +
  facet_grid(glue::glue('italic(sigma)*" = {S}"') ~ glue::glue('italic(m)*" = {m}"'), 
             labeller = label_parsed) +
   geom_polygon(data = max_df, color = "white", alpha = 0, size = 0.8) +
   scale_x_continuous(breaks = c(0, 0.25, 0.5), labels = c("0", "0.25", "0.5")) +
   scale_y_continuous(breaks = c(0, 0.5, 1), labels = c("0", "0.5", "1"))

  )

ggsave(filename = here::here("R_figure_code", "mutationalMeltdown", "meltdown_slope_mean_of_log_single_scale_with_max_polygon.pdf"),
       plot = plot, height = 7, width = 8)
