library(tidyverse)

df <- readRDS(here::here("R_figure_code", "Fig_SX_GroupProp", "data_groupProp_RDS"))

plotA <- df %>% 
  mutate(frag_mode = case_when(
    frag_mode_idx == 0 ~ "Complete fragmentation",
    frag_mode_idx == 1 ~ "Single-cell reproduction",
    frag_mode_idx == 2 ~ "Binary fission"
  )) %>% 
  filter(indv_NType == 2,
         indv_mutR %in% c(0.001, 0.01)) %>% 
  ggplot(aes(x = group_size, color = factor(indv_mutR))) +
  geom_freqpoly(bins = 15) + 
  facet_grid(~ frag_mode) +
  coord_cartesian(ylim = c(0, 1600)) +
  cowplot::theme_cowplot() +
  cowplot::panel_border() +
  labs(color = "Mutation rate",
       y = "No. of groups",
       x = expression("Group size ("*italic("N")[i]*")" )) +
  theme(legend.position = "top",
        strip.background = element_rect(fill = "white")) +
  scale_y_continuous(expand = expansion(mult = c(0, 0))) +
  scale_color_manual(values = c("black", "darkgrey"))


plotB <- df %>% 
  mutate(frag_mode = case_when(
    frag_mode_idx == 0 ~ "Complete fragmentation",
    frag_mode_idx == 1 ~ "Single-cell reproduction",
    frag_mode_idx == 2 ~ "Binary fission"
  )) %>% 
  filter(indv_NType == 2,
         indv_mutR %in% c(0.001, 0.01)) %>% 
  ggplot(aes(x = coop_freq, fill = factor(indv_mutR), color = factor(indv_mutR))) +
  geom_freqpoly(bins = 15) + 
  facet_grid(~ frag_mode) +
  coord_cartesian(ylim = c(0, 1600)) +
  cowplot::theme_cowplot() +
  cowplot::panel_border() +
  labs(color = "Mutation rate",
       y = "No. of groups",
       x = "Fraction WT cells") +
  theme(legend.position = "none",
        strip.background = element_rect(fill = "white")) +
  scale_y_continuous(expand = expansion(mult = c(0, 0))) +
  scale_color_manual(values = c("black", "darkgrey"))
  
plotC <- df %>% 
  mutate(frag_mode = case_when(
    frag_mode_idx == 0 ~ "Complete fragmentation",
    frag_mode_idx == 1 ~ "Single-cell reproduction",
    frag_mode_idx == 2 ~ "Binary fission"
  )) %>% 
  filter(indv_NType == 2,
         indv_mutR %in% c(0.001, 0.01)) %>% 
  ggplot(aes(x = group_size, y = coop_freq, color = factor(indv_mutR))) +
  geom_smooth(alpha = 0.2) +
  facet_grid(~ frag_mode) +
  coord_cartesian(ylim = c(0, 1)) +
  cowplot::theme_cowplot() +
  cowplot::panel_border() +
  labs(color = "Mutation rate",
       y = "Fraction WT cells",
       x = expression("Group size ("*italic("N")[i]*")" )) +
  theme(legend.position = "none",
        strip.background = element_rect(fill = "white")) +
  scale_y_continuous(expand = expansion(mult = c(0, 0))) +
  scale_color_manual(values = c("black", "darkgrey"))

full_plot <- egg::ggarrange(plotA, plotB, plotC, ncol = 1)

ggsave(filename = here::here("R_figure_code", "Fig_SX_GroupProp", "groupProp.pdf"),
      plot = full_plot,
      width = 7.25,
      height = 7.65)
