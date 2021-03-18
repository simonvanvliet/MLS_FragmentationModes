library(tidyverse)

df <- readRDS(file = here::here("R_figure_code", "transects", "data_transects_RDS"))

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

(transect_mu_plot <- three_types %>% 
  tibble() %>% 
  unite("strategy", c(offspr_size, offspr_frac), sep = "\n") %>%
  filter(indv_mutR <= 0.025) %>% 
  ggplot() +
  geom_point(aes(x = strategy, y = NTot_mav, color = (as.factor(indv_mutR)))) +
  cowplot::theme_cowplot() +
  theme(legend.position = c(0.55, 0.8),
        legend.text = element_text(size = 9),
        legend.title = element_text(size = 9)) +
  labs(x = "Reproduction strategy", y = expression("Productivity ("*italic(N)[tot]*")"), color = "Mutation\nrate") +
  scale_x_discrete(breaks = c("0.01\n0.99", "0.26\n0.74", "0.5\n0.5"), 
                   labels = c(expression(atop(italic("s")*"= 0.01", italic(n)*"= 0.99")), 
                              expression(atop(italic("s")*"= 0.26", italic(n)*"= 0.74")),
                              expression(atop(italic("s")*"= 0.5", italic(n)*"= 0.5"))
                   ),
                   expand = expansion(mult = c(0.05, 0.1))) +
  scale_y_continuous(labels = scales::label_comma(accuracy = 1)) +
  scale_color_grey())

saveRDS(transect_mu_plot, file = here::here("R_figure_code", "transects", "plot_transect_mu_RDS"))
                   

(transect_nr_species_plot <- rbind(
  one_type %>% select(indv_mutR, NTot_mav, offspr_size, offspr_frac) %>% mutate(Nr_Types = "1"),
  two_types %>% select(indv_mutR, NTot_mav, offspr_size, offspr_frac) %>% mutate(Nr_Types = "2"),
  three_types %>% select(indv_mutR, NTot_mav, offspr_size, offspr_frac) %>% mutate(Nr_Types = "3"),
  four_types %>% select(indv_mutR, NTot_mav, offspr_size, offspr_frac) %>% mutate(Nr_Types = "4")
  ) %>% 
  tibble() %>% 
  filter(indv_mutR == 0.001) %>% 
  unite("strategy", c(offspr_size, offspr_frac), sep = "\n") %>%
  ggplot() +
  geom_point(aes(x = strategy, y = NTot_mav, color = (Nr_Types))) +
  cowplot::theme_cowplot() +
  theme(legend.position = c(0.55, 0.8),
        legend.text = element_text(size = 9),
        legend.title = element_text(size = 9)) +
  labs(x = "Reproduction strategy", y = expression("Productivity ("*italic(N)[tot]*")"), color = expression("No. species")) +
  scale_x_discrete(breaks = c("0.01\n0.99",  "0.26\n0.74",  "0.5\n0.5"), 
                   labels = c(expression(atop(italic("s")*"= 0.01", italic(n)*"= 0.99")), 
                              expression(atop(italic("s")*"= 0.26", italic(n)*"= 0.74")),
                              expression(atop(italic("s")*"= 0.5", italic(n)*"= 0.5"))
                   ),
                   expand = expansion(mult = c(0.05, 0.05))) +
  scale_y_continuous(labels = scales::label_comma(accuracy = 1)) +
  scale_color_grey()
)

saveRDS(transect_nr_species_plot, file = here::here("R_figure_code", "transects", "plot_transect_nr_species_RDS"))
