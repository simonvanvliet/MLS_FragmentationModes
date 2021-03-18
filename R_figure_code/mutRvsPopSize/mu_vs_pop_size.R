library(tidyverse)

df <- readRDS(here::here("R_figure_code", "mutRvsPopSize", "data_mutation_vs_popsize_RDS")) %>% as_tibble()

colors <- c("#474648", "#2F76AC", "#D9665A", "#E0C537") # colorblind friendly four-color palette


# Population size as function of mu
(mu_vs_pop_size_plot <- df %>% 
  mutate(frag_mode = paste(offspr_size, offspr_frac, sep = "_"),
         unique_run = paste(run_idx, replicate_idx, sep = "_")) %>% 
  select_if(~n_distinct(.) > 1) %>% 
  #filter(NTot_mav >= 1) %>% 
  ggplot(aes(x = indv_mutR, y = NTot_mav, color = frag_mode)) +
  annotation_logticks(sides = "b") +
  stat_summary(fun = mean, geom = "line", size = 1) +
  geom_jitter(width = resolution(log10(df$indv_mutR), FALSE) * 0.3, alpha = 0.8) +
  scale_x_log10() +
  scale_y_continuous(labels = scales::label_comma()) +
  cowplot::theme_cowplot() +
  theme(legend.position = c(0.45, 0.9),
        legend.justification = "left",
        legend.text = element_text(size = 11),
        legend.title = element_blank()) +
  scale_color_manual(labels = c("Single-cell repr.", "Complete frag.", "Binary fission"),
    values = colors[-1]) +
  labs(x = expression("Mutation rate ("*mu*")"), y = expression("Productivity ("*italic(N)[tot]*")"))
)

saveRDS(mu_vs_pop_size_plot, file = here::here("R_figure_code", "mutRvsPopSize", "plot_mutation_vs_popsize_RDS"))



# Fraction cooperators as function of mu
(mu_vs_frac_coop <- df %>% 
    mutate(frag_mode = paste(offspr_size, offspr_frac, sep = "_"),
           unique_run = paste(run_idx, replicate_idx, sep = "_")) %>% 
    select_if(~n_distinct(.) > 1) %>% 
    group_by(frag_mode) %>% 
    mutate(max_frac = max(fCoop_mav),
           fCoop_rel = fCoop_mav / max_frac) %>% 
    ggplot(aes(x = indv_mutR, y = fCoop_rel, color = frag_mode)) +
    annotation_logticks(sides = "b") +
    stat_summary(fun = mean, geom = "line", size = 1) +
    geom_jitter(width = resolution(log10(df$indv_mutR), FALSE) * 0.3, alpha = 0.8) +
    scale_x_log10() +
    scale_y_continuous(labels = scales::label_comma()) +
    cowplot::theme_cowplot() +
    theme(legend.position = c(0.45, 0.9),
          legend.justification = "left",
          legend.text = element_text(size = 11),
          legend.title = element_blank()) +
    scale_color_manual(labels = c("Single-cell repr.", "Complete frag.", "Binary fission"),
                       values = colors[-1]) +
    labs(x = expression("Mutation rate ("*mu*")"), y = expression("Relative fraction of WT cells"))
)

saveRDS(mu_vs_frac_coop, file = here::here("R_figure_code", "mutRvsPopSize", "plot_mutation_vs_frac_coop_RDS"))



