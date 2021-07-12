library(tidyverse)

df_without <- readRDS(file = here::here("R_figure_code", "dynamicsWOGrpEvents", "data_without_group_events_RDS"))
df_without <- df_without %>% tibble()

df_with <- readRDS(file = here::here("R_figure_code", "dynamicsWGrpEvents", "data_with_group_events_RDS"))
df_with <- df_with %>% tibble() %>% filter(gr_CFis == 0.01)

df <- bind_rows(df_without, df_with) %>% 
  filter(indv_mutR == 0.001 & gr_CFis %in% c(0.01, 0)) %>% 
  mutate(fission_strategy = paste0("s = ", offspr_size, ", n = ", offspr_frac),
         NGrp_mav = ifelse(NGrp_mav == 0, NA, NGrp_mav),
         unique_run = paste0(run_idx, replicate_idx)) %>%
  # Pull ten replicates from the "no group events" case
  mutate(fission_strategy = if_else(gr_CFis == 0, "No group events", fission_strategy))

time_units_per_bin <- 75
grand_means <- df %>% 
  mutate(time_bins = cut(time, 
                         breaks = seq(from = 0, to = max(df$time), by = time_units_per_bin),
                         labels = FALSE),
         time_bins = seq(from = 0, to = max(df$time), by = time_units_per_bin)[time_bins+1]) %>%
  group_by(time_bins, fission_strategy) %>% 
  summarize(NGrp_mean = mean(NGrp)) %>% 
  bind_rows(tribble(~time_bins, ~fission_strategy, ~NGrp_mean,
                    0, "No group events", 100,
                    0, "s = 0.01, n = 0.01", 100,
                    0, "s = 0.01, n = 0.99", 100,
                    0, "s = 0.5, n = 0.5", 100))

colors <- c("#474648", "#2F76AC", "#D9665A", "#E0C537") # colorblind friendly four-color palette

plotG <- df %>% 
  mutate(unique_run = paste0(unique_run, fission_strategy)) %>% 
  filter(NGrp_mav >= 1) %>% 
  ggplot(aes(x = time, y = NGrp_mav, color = fission_strategy)) +
  annotation_logticks(sides = "l") +
  geom_line(alpha = 0.3, aes(group = unique_run), show.legend = FALSE) +
  geom_line(data = filter(grand_means, NGrp_mean > 1), aes(x = time_bins, y = NGrp_mean),
            size = 1) +
  cowplot::theme_cowplot() +
  theme(legend.position = c(0.3,0.75),
        legend.justification = "left",
        legend.text = element_text(size = 11),
        legend.title = element_blank()) +
  scale_y_log10(labels = scales::label_comma(accuracy = 1)) +
  scale_color_manual(labels = c("No group events", "Single-cell reproduction", "Complete fragmentation", "Binary fission"),
                     values = colors) +
  labs(x = "Time", y = expression("Number of groups ("*italic(G)*")"))


grand_means_N <- df %>% 
  mutate(time_bins = cut(time, 
                         breaks = seq(from = 0, to = max(df$time), by = time_units_per_bin),
                         labels = FALSE),
         time_bins = seq(from = 0, to = max(df$time), by = time_units_per_bin)[time_bins+1]) %>%
  group_by(time_bins, fission_strategy) %>% 
  summarize(N_mean = mean(NTot_mav)) %>% 
  bind_rows(tribble(~time_bins, ~fission_strategy, ~N_mean,
                    0, "No group events", 10000,
                    0, "s = 0.01, n = 0.01", 10000,
                    0, "s = 0.01, n = 0.99", 10000,
                    0, "s = 0.5, n = 0.5", 10000))

plotN <- df %>% 
  mutate(unique_run = paste0(unique_run, fission_strategy)) %>% 
  filter(NTot_mav >= 1) %>% 
  ggplot(aes(x = time, y = NTot_mav, color = fission_strategy)) +
  annotation_logticks(sides = "l") +
  geom_line(alpha = 0.3, aes(group = unique_run), show.legend = FALSE) +
  geom_line(data = filter(grand_means_N, N_mean > 1), aes(x = time_bins, y = N_mean),
            size = 1) +
  cowplot::theme_cowplot() +
  theme(legend.position = "none") +
  scale_y_log10(labels = scales::label_comma(accuracy = 1)) +
  scale_color_manual(labels = c("No group events", "Single-cell reproduction", "Complete fragmentation", "Binary fission"),
                     values = colors) +
  labs(x = "Time", y = expression("Productivity ("*italic(N)[total]*")"))


plotMut <- df %>% 
  filter(indv_mutR == "0.001",
         gr_CFis == "0") %>% 
  filter(unique_run == "11") %>% 
  select(time, N0_mav, N0mut_mav) %>% 
  pivot_longer(cols = -time) %>% 
  ggplot() +
  geom_area(aes(x = time, y = value, fill = name)) +
  cowplot::theme_cowplot() +
  theme(legend.position = c(0.2,0.75),
        legend.justification = "left",
        legend.text = element_text(size = 11),
        legend.title = element_blank()) +
  scale_x_continuous(expand = expansion(0, 0)) +
  scale_y_continuous(expand = expansion(0, 0), labels = scales::label_comma(accuracy = 1)) +
  labs(x = "Time", y = "Number of cells") +
  scale_fill_manual(labels = c("Wild-type cells", "Mutant cells"),
                    values = c("darkgrey", "black"))


top_plots <- cowplot::plot_grid(plotMut, plotG, plotN, 
                                nrow = 1, labels = c("A", "B", "C"),
                                rel_widths = c(0.9, 1.5, 1.5))  

mu_n_plot <- readRDS(here::here("R_figure_code", "mutRvsPopSize", "plot_mutation_vs_popsize_RDS")) +
  ylab(expression("Productivity ("*italic(N)[total]*")"))

meltdown_plot <- readRDS(here::here("R_figure_code", "mutationalMeltdown", "meltdown_plot_RDS")) +
  labs(x = expression("Fractional offspring size ("*italic(s)*")"),
       y = expression("Fractional offspring number ("*italic(n)*")   "))
       
mu_fcoop <-readRDS(here::here("R_figure_code", "mutRvsPopSize", "plot_mutation_vs_frac_coop_RDS")) +
  labs(y = "Normalized fraction of WT cells   ")

bottom_plots <- cowplot::plot_grid(mu_n_plot, 
                                   meltdown_plot, 
                                   mu_fcoop,
                                   nrow = 1, labels = c("D", "E", "F"))

all_plots <- cowplot::plot_grid(top_plots, bottom_plots, nrow = 2)



ggsave(here::here("R_figure_code", "dynamicsWGrpEvents", "dynamics_with_and_without_group_events.pdf"), 
       plot = all_plots, width = 11, height = 7)


