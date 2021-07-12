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



# This version makes a hull around the top 5 values of max_mu ---------
 
max_df <- plot_df2 %>% 
  ungroup() %>% 
  mutate(max_mu = ifelse(max_mu == -Inf, NA, max_mu)) %>% 
  mutate(max_mu = if_else(max_mu < -10, -10, max_mu)) %>% 
  mutate(m = indv_NType, v = indv_migrR) %>% 
  filter( v < 5) %>% 
  group_by(m, v) %>% 
  filter(!is.na(max_mu)) %>% 
  slice_max(order_by = max_mu, n = 5) %>% 
  slice(chull(offspr_size, offspr_frac))





# Plot --------------------------------------------------------------------
plot_df2$indv_migrR <- as.character(plot_df2$indv_migrR)
plot_df2[plot_df2$indv_migrR == "0",]$indv_migrR <- "0.0"
plot_df2$indv_migrR <- factor(plot_df2$indv_migrR, levels = c("0.0", "0.01", "0.1","1", "5"))

max_df$v <- as.character(max_df$v)
max_df[max_df$v == "0",]$v <- "0.0"
max_df$v <- factor(max_df$v, levels = c("0.0", "0.01", "0.1","1"))


(plot <- plot_df2 %>% 
  mutate(max_mu = ifelse(max_mu == -Inf, NA, max_mu)) %>% 
  mutate(max_mu = if_else(max_mu < -10, -10, max_mu)) %>% 
  mutate(m = indv_NType, v = indv_migrR) %>% 
    filter( v != "5") %>% 
  ggplot(aes(x = offspr_size, y = offspr_frac)) +
  geom_tile(aes(fill = max_mu)) +
  labs(x = expression("Fractional offspring size ("*italic(s)*")"),
       y = expression("Fractional offspring number ("*italic(n)*")"),
       fill = "Maximum\nmutation\nrate\n(log)") +
  cowplot::theme_cowplot() +
  theme(aspect.ratio = 1,
        strip.background = element_blank()) +
  guides(fill = element_blank()) +
  scale_fill_viridis_c() +
  cowplot::panel_border() +
   facet_grid(glue::glue('nu*" = {v}"') ~ glue::glue('italic(m)*" = {m}"'), 
              labeller = label_parsed) +
   geom_polygon(data = max_df, color = "white", alpha = 0, size = 0.8) +
   scale_x_continuous(breaks = c(0, 0.25, 0.5), labels = c("0", "0.25", "0.5")) +
   scale_y_continuous(breaks = c(0, 0.5, 1), labels = c("0", "0.5", "1"))
 
)

ggsave(filename = here::here("R_figure_code", "mutationalMeltdown", "meltdown_migr_mean_of_log_single_scale_with_max_polygon.pdf"),
       plot = plot, height = 7, width = 8)
