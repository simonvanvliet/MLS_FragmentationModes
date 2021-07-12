library(tidyverse)

df <- readRDS(here::here("R_figure_code", "Rev_scanComplexityNoDensDep", "data_scan_complexity_NoDensDep_RDS"))

one_type <- df %>% filter(indv_NType==1)

two_types <- df %>% filter(indv_NType==2)

three_types <- df %>% filter(indv_NType==3)


plot_df <- rbind(
  one_type %>% 
    group_by(indv_asymmetry, offspr_frac, offspr_size) %>% 
    summarize(nr_cells = mean(NTot_mav, na.rm = TRUE),
              nr_groups = mean(NGrp_mav, na.rm = TRUE)) %>% 
    ungroup() %>% 
    mutate(nr_spp = "1"),
  
  two_types %>% 
    group_by(indv_asymmetry, offspr_frac, offspr_size) %>% 
    summarize(nr_cells = mean(NTot_mav, na.rm = TRUE),
              nr_groups = mean(NGrp_mav, na.rm = TRUE)) %>% 
    ungroup() %>% 
    mutate(nr_spp = "2"),
  
  three_types %>% 
    group_by(indv_asymmetry, offspr_frac, offspr_size) %>% 
    summarize(nr_cells = mean(NTot_mav, na.rm = TRUE),
              nr_groups = mean(NGrp_mav, na.rm = TRUE)) %>% 
    ungroup() %>% 
    mutate(nr_spp = "3")
)

plot_list <- vector(mode = "list", length = 3)
for (i in 1:3){
  species_number <- i; asymmetry <- 1
  
  p <- plot_df %>% 
    filter(indv_asymmetry == asymmetry) %>%
    filter(nr_spp == species_number) %>% 
    ggplot() +
    geom_tile(aes(x = offspr_size, y = offspr_frac, fill = nr_cells)) +
    cowplot::theme_cowplot() +
    cowplot::panel_border() +
    labs(subtitle = substitute(expr = paste(italic("m"), " = ", i), env = list(i = i)),
         fill = expression(italic(N)[total])) +
    theme(aspect.ratio = 1,
          axis.title.y = element_blank(),
          axis.text.x = element_text(),
          axis.title.x = element_blank(),
          legend.text = element_text(size = 8),
          legend.direction = "horizontal",
          legend.key.width = unit(0.4, "cm"),
          legend.key.height = unit(0.225, "cm"),
          legend.title = element_text(size = 9),
          legend.position = c(0.525, 0.175),
          plot.subtitle = element_text(size = 14)) +
    guides(fill = guide_colourbar(title.position = "top", title.hjust = 1)) +
    scale_fill_viridis_c(breaks = scales::pretty_breaks(n = 2)#, limits = c(0,49000)
    ) +
    scale_x_continuous(breaks = c(0, 0.25, 0.5)) 
  if(i == 1) {
    p <- p + theme(axis.text.y = element_text(),
                   axis.title.y = element_text(angle = 90)) +
      labs(y = expression(atop("Fractional offspring", "number ("*italic(n)*")"))) 
  } else { p <- p + theme(axis.text.y = element_blank()) }
  
  plot_list[[i]] <- p
}



plot_list[[1]] <- plot_list[[1]] +
  annotate("segment", x = 0, y = 1, xend = 0.5, yend = 0.5,
           color = "salmon", size = 1.5) +
  annotate("text", x = 0.325, y = 0.85, label = "Upper\ntransect", 
           color = "salmon", hjust = 0) +
  annotate("curve", x = 0.32, y = 0.85, xend = 0.25, yend = 0.78, 
           arrow = arrow(length = unit(0.03, "npc")), color = "salmon")

plot_list[[1]] <- plot_list[[1]] + coord_cartesian(ylim = c(0.0, 0.98))
plot_list[[2]] <- plot_list[[2]] + coord_cartesian(ylim = c(0.0, 0.98))
plot_list[[3]] <- plot_list[[3]] + coord_cartesian(ylim = c(0.0, 0.98))

triangles <- egg::ggarrange(plots = plot_list, nrow = 1)
triangles <- ggpubr::annotate_figure(triangles, bottom = ggpubr::text_grob(expression("Fractional offspring size ("*italic(s)*")"), 
                                                                           size = 14))

saveRDS(triangles, file = here::here("R_figure_code", "Rev_scanComplexityNoDensDep", "plot_scan_complexity_NoDensDep_RDS"))

scan_comp_plot <- readRDS(here::here("R_figure_code", "Rev_scanComplexityNoDensDep", "plot_scan_complexity_NoDensDep_RDS"))

plotA <- plotA + 
  theme(axis.line.x = element_line(color = "salmon", size = 1.5),
        axis.title.x = element_text(color = "salmon")) +
  labs(x = "Reproduction strategy\n(upper transect)",
       y = expression("Productivity ("*italic(N)[total]*")"))

plotB <- plotB + 
  labs(y = "Normalized productivity          ")

plotC <- plotC + 
  theme(axis.line.x = element_line(color = "salmon", size = 1.5),
        axis.title.x = element_text(color = "salmon")) +
  labs(x = "Reproduction strategy\n(upper transect)",
       y = expression("Productivity ("*italic(N)[total]*")"))

final_plot <- cowplot::plot_grid(
  cowplot::plot_grid(scan_comp_plot, labels = c("A")),
  cowplot::plot_grid((plotA + theme(legend.position = c(0.65, 0.75)) + scale_color_manual(values = viridisLite::viridis(4))), 
                     (plotB + theme(legend.position = c(0.65, 0.75)) + scale_color_manual(values = viridisLite::viridis(4))), 
                     (plotC + theme(legend.position = c(0.65, 0.8)) + scale_color_grey()), 
                     nrow = 1, labels = c("B", "C", "D"), align = "h", axis = "b"),
  nrow = 2)

ggsave(here::here("R_figure_code", "Rev_scanComplexityNoDensDep", "complexity_plot_NoDensDep.pdf"),
       plot = final_plot,
       width = 9.5, height = 6)



