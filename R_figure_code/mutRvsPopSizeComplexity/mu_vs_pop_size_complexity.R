library(tidyverse)

df <- readRDS(here::here("R_figure_code", "mutRvsPopSizeComplexity", "data_mutation_vs_popsize_complexity_RDS")) %>% as_tibble()

# df doesn't have column names, so I will create column names from a different data frame
colnames <- colnames(readRDS(file = here::here("R_figure_code", "mutRvsPopSize", "data_mutation_vs_popsize_RDS")))

one_type <- df %>% filter(is.nan(`32`))
colnames(one_type) <- colnames
one_type <- one_type[1:32]

two_types <- df %>% filter(!is.nan(`35`) & is.nan(`36`))
colnames(two_types) <- c(colnames[1:23], 
                         c("N0", "N1", "N0mut", "N1mut"), 
                         colnames[26:30], 
                         c("N0_mav", "N1_mav", "N0mut_mav", "N1mut_mav"))
two_types <- two_types[1:36]

three_types <- df %>% filter(!is.nan(`39`) & is.nan(`40`))
colnames(three_types) <- c(colnames[1:23],
                           c("N0", "N1", "N2", "N0mut", "N1mut", "N2mut"), 
                           colnames[26:30],
                           c("N0_mav", "N1_mav", "N2_mav", "N0mut_mav", "N1mut_mav", "N2mut_mav"))
three_types <- three_types[1:40]

four_types <- df %>% filter(!is.nan(`40`))
colnames(four_types) <- c(colnames[1:23],
                          c("N0", "N1", "N2", "N3", "N0mut", "N1mut", "N2mut", "N3mut"), 
                          colnames[26:30],
                          c("N0_mav", "N1_mav", "N2_mav", "N3_mav", "N0mut_mav", "N1mut_mav", "N2mut_mav", "N3mut_mav"))

(mu_vs_pop_size_plot <- rbind(
  one_type %>% select(indv_mutR, NTot_mav) %>% mutate(Nr_Types = "1"),
  two_types %>% select(indv_mutR, NTot_mav) %>% mutate(Nr_Types = "2"),
  three_types %>% select(indv_mutR, NTot_mav) %>% mutate(Nr_Types = "3"),
  four_types %>% select(indv_mutR, NTot_mav) %>% mutate(Nr_Types = "4")) %>% 
    group_by(Nr_Types) %>% 
  mutate(NTot_rel = NTot_mav/max(NTot_mav)) %>% 
  ggplot(aes(x = indv_mutR, y = NTot_rel, color = (Nr_Types))) +
  annotation_logticks(sides = "b") +
  #stat_summary(fun = mean, geom = "line", size = 1) +
  geom_jitter(width = resolution(log10(one_type$indv_mutR), FALSE) * 0.3, alpha = 0.8) +
  scale_x_log10() +
  scale_y_continuous(labels = scales::label_comma()) +
  cowplot::theme_cowplot() +
  theme(legend.position = c(0.55, 0.8),
        legend.justification = "left",
        legend.text = element_text(size = 9),
        legend.title = element_text(size = 9)) +
  labs(x = expression("Mutation rate ("*mu*")"), y = expression("Relative productivity"),
       color = "No. species") +
  scale_color_grey()
)

saveRDS(mu_vs_pop_size_plot, file = here::here("R_figure_code", "mutRvsPopSizeComplexity", "plot_mutation_vs_n_complexity_RDS"))

