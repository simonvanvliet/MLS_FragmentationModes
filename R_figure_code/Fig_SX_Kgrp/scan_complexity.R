library(tidyverse)

df <- readRDS(here::here("R_figure_code", "Fig_SX_Kgrp", "data_scan_complexity_Kgrp_RDS"))

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


plot_df <- rbind(
  one_type %>% 
    group_by(indv_mutR, offspr_frac, offspr_size) %>% 
    summarize(nr_cells = mean(NTot_mav, na.rm = TRUE),
              nr_groups = mean(NGrp_mav, na.rm = TRUE)) %>% 
    ungroup() %>% 
    mutate(nr_spp = "1"),
  
  two_types %>% 
    group_by(indv_mutR, offspr_frac, offspr_size) %>% 
    summarize(nr_cells = mean(NTot_mav, na.rm = TRUE),
              nr_groups = mean(NGrp_mav, na.rm = TRUE)) %>% 
    ungroup() %>% 
    mutate(nr_spp = "2"),
  
  three_types %>% 
    group_by(indv_mutR, offspr_frac, offspr_size) %>% 
    summarize(nr_cells = mean(NTot_mav, na.rm = TRUE),
              nr_groups = mean(NGrp_mav, na.rm = TRUE)) %>% 
    ungroup() %>% 
    mutate(nr_spp = "3"),
  
  four_types %>% 
    group_by(indv_mutR, offspr_frac, offspr_size) %>% 
    summarize(nr_cells = mean(NTot_mav, na.rm = TRUE),
              nr_groups = mean(NGrp_mav, na.rm = TRUE)) %>% 
    ungroup() %>% 
    mutate(nr_spp = "4")
)

plot_list <- vector(mode = "list", length = 3)
for (i in 1:3){
  species_number <- i; mu <- 0.001
  
  p <- plot_df %>% 
    filter(indv_mutR == mu) %>%
    filter(nr_spp == species_number) %>% 
    ggplot() +
    geom_tile(aes(x = offspr_size, y = offspr_frac, fill = nr_cells)) +
    cowplot::theme_cowplot() +
    cowplot::panel_border() +
    labs(subtitle = substitute(expr = paste(italic("m"), " = ", i), env = list(i = i)),
         x = expression(italic(s))) +
    theme(aspect.ratio = 1,
          axis.title.y = element_blank(),
          axis.text.x = element_text(),
          axis.title.x = element_text(),
          legend.text = element_text(size = 8),
          legend.direction = "horizontal",
          legend.key.width = unit(0.45, "cm"),
          legend.key.height = unit(0.225, "cm"),
          legend.title = element_blank(),
          legend.position = c(0.4, 0.1),
          plot.subtitle = element_text(size = 14)) +
    scale_fill_viridis_c(breaks = scales::pretty_breaks(n = 2)) +
    scale_x_continuous(breaks = c(0, 0.25, 0.5)) 
  if(i == 1) {
    p <- p + theme(axis.text.y = element_text(),
                   axis.title.y = element_text(angle = 90)) +
      labs(y = expression(italic(n))) 
  } else { p <- p + theme(axis.text.y = element_blank()) }
  
  plot_list[[i]] <- p
}

triangles <- egg::ggarrange(plots = plot_list, nrow = 1)

plot_mu <- rbind(
  one_type %>% select(offspr_frac, offspr_size, NTot_mav, indv_mutR) %>% mutate(nr_spp = "1"),
  two_types %>% select(offspr_frac, offspr_size, NTot_mav, indv_mutR) %>% mutate(nr_spp = "2"),
  three_types %>% select(offspr_frac, offspr_size, NTot_mav, indv_mutR) %>% mutate(nr_spp = "3"),
  four_types %>% select(offspr_frac, offspr_size, NTot_mav, indv_mutR) %>% mutate(nr_spp = "4")
) %>% 
  group_by(offspr_size) %>% 
  filter(offspr_frac == max(offspr_frac)) %>% 
  filter(nr_spp == 2) %>% 
  ggplot(aes(x = offspr_size, y = NTot_mav, color = as.factor(indv_mutR))) +
  geom_point() +
  #stat_summary(fun = "mean", geom = "line", size = 1) +
  geom_smooth(se = FALSE) +
  cowplot::theme_cowplot() +
  theme(legend.text = element_text(size = 9),
        legend.title = element_text(size = 9)) +
  labs(x = "Reproduction strategy\n(upper transect)", y = expression("Productivity ("*italic(N)[tot]*")"), color = "Mutation\nrate") +
  scale_x_continuous(breaks = c(0.01, 0.248, 0.486), 
                   labels = c(expression(atop(italic("s")*"= 0.01", italic(n)*"= 0.92")), 
                              expression(atop(italic("s")*"= 0.248", italic(n)*"= 0.71")),
                              expression(atop(italic("s")*"= 0.486", italic(n)*"= 0.5"))
                   ),
                   expand = expansion(mult = c(0.05, 0.1))) +
  scale_y_continuous(labels = scales::label_comma(accuracy = 1)) +
  scale_color_grey()

plot_K <- cowplot::plot_grid(triangles, plot_mu, labels = c("A", "B"),
                   nrow = 1, rel_widths = c(1, 0.6))

ggsave(plot = plot_K, 
       filename = here::here("R_figure_code", "Fig_SX_Kgrp", "plot_Kgrp.pdf"),
       width = 11, height = 3
       )                  
