library(tidyverse)

df <- readRDS(here::here("R_figure_code", "Fig_SX_death_rate_WIP", "data_scan_complexity_RDS"))

plot_df <- df %>% 
  group_by(indv_NType, indv_mutR, offspr_frac, offspr_size) %>% 
  summarize(nr_cells = mean(NTot_mav, na.rm = TRUE),
            nr_groups = mean(NGrp_mav, na.rm = TRUE)) %>% 
  ungroup() %>% 
  rename(nr_spp = indv_NType)

plot_list <- vector(mode = "list", length = 3)
for (i in 1:3){
  species_number <- i; 
  mu <- 0.01
  
  p <- plot_df %>% 
    filter(indv_mutR == mu) %>%
    filter(nr_spp == species_number) %>% 
    ggplot() +
    geom_tile(aes(x = offspr_size, y = offspr_frac, fill = nr_cells)) +
    cowplot::theme_cowplot() +
    cowplot::panel_border() +
    labs(subtitle = substitute(expr = paste(italic("m"), " = ", i), env = list(i = i))) +
    theme(aspect.ratio = 1,
          axis.title.y = element_blank(),
          axis.text.y = element_blank(),
          axis.title.x = element_blank(),
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
      labs(y = expression(atop("Fractional offspring", "number ("*italic(n)*")")))
  } else if (i == 2) {
    p <- p + theme(axis.text.x = element_text(),
                   axis.title.x = element_text()) +
      labs(x = expression("Fractional offspring size ("*italic(s)*")"))
  } else { p <- p + theme(axis.text.y = element_blank()) }

  plot_list[[i]] <- p
}

triangles <- egg::ggarrange(plots = plot_list, nrow = 1)


ggsave(plot = triangles, 
       filename = here::here("R_figure_code", "Fig_SX_death_rate_WIP", "trangles.pdf"),
       width = 9, height = 3
)                 


### TEMPORARY PLOT TO SHOW SIMON

plot_mu_temporary <- df %>%
  select(offspr_frac, offspr_size, NTot_mav, indv_mutR, indv_NType) %>% 
  rename(nr_spp = indv_NType) %>% 
  group_by(offspr_size) %>% 
  filter(offspr_frac == max(offspr_frac)) %>% 
  ggplot(aes(x = offspr_size, y = NTot_mav, color = as.factor(indv_mutR))) +
  geom_smooth(se = FALSE) +
  #stat_summary(fun = "mean", geom = "line", size = 1) +
  geom_point() +
  cowplot::theme_cowplot() +
  theme(legend.text = element_text(size = 9),
        legend.title = element_text(size = 9)) +
  labs(x = "Reproduction strategy\n(upper transect)", y = expression("Productivity ("*italic(N)[total]*")"), color = "Mutation\nrate") +
  scale_x_continuous(breaks = c(0.01, 0.248, 0.486), 
                     labels = c(expression(atop(italic("s")*"= 0.01", italic(n)*"= 0.92")), 
                                expression(atop(italic("s")*"= 0.248", italic(n)*"= 0.71")),
                                expression(atop(italic("s")*"= 0.486", italic(n)*"= 0.5"))
                     ),
                     expand = expansion(mult = c(0.05, 0.1))) +
  scale_y_continuous(labels = scales::label_comma(accuracy = 1)) +
  scale_color_grey() +
  facet_wrap(~ nr_spp)

ggsave(plot = plot_mu_temporary, 
       filename = here::here("R_figure_code", "Fig_SX_death_rate_WIP", "mu.pdf"),
       width = 11, height = 4
)           


### END OF TEMPORARY PLOT



plot_mu <- df %>%
  select(offspr_frac, offspr_size, NTot_mav, indv_mutR, indv_NType) %>% 
  rename(nr_spp = indv_NType) %>% 
  group_by(offspr_size) %>% 
  filter(offspr_frac == max(offspr_frac)) %>% 
  filter(nr_spp == 1) %>% 
  ggplot(aes(x = offspr_size, y = NTot_mav, color = as.factor(indv_mutR))) +
  geom_smooth(se = FALSE) +
  #stat_summary(fun = "mean", geom = "line", size = 1) +
  geom_point() +
  cowplot::theme_cowplot() +
  theme(legend.text = element_text(size = 9),
        legend.title = element_text(size = 9)) +
  labs(x = "Reproduction strategy\n(upper transect)", y = expression("Productivity ("*italic(N)[total]*")"), color = "Mutation\nrate") +
  scale_x_continuous(breaks = c(0.01, 0.248, 0.486), 
                     labels = c(expression(atop(italic("s")*"= 0.01", italic(n)*"= 0.92")), 
                                expression(atop(italic("s")*"= 0.248", italic(n)*"= 0.71")),
                                expression(atop(italic("s")*"= 0.486", italic(n)*"= 0.5"))
                     ),
                     expand = expansion(mult = c(0.05, 0.1))) +
  scale_y_continuous(labels = scales::label_comma(accuracy = 1)) +
  scale_color_grey()


plot_Ni <- cowplot::plot_grid(triangles, plot_mu, labels = c("A", "B"),
                              nrow = 1, rel_widths = c(1, 0.6))



ggsave(plot = plot_Ni, 
       filename = here::here("R_figure_code", "Fig_SX_Ni", "plot_Ni.pdf"),
       width = 11, height = 3
       )                  
