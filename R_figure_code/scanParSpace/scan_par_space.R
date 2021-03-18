library(tidyverse)

df <- readRDS(here::here("R_figure_code", "scanParSpace", "data_scan_par_space_RDS"))

plot_list <- vector(mode = "list", length = 2)
for(i in 1:2){
  plot_df <- df %>% 
    group_by(offspr_frac, offspr_size) %>% 
    summarize(nr_cells = mean(NTot_mav, na.rm = TRUE),
              nr_groups = mean(NGrp_mav, na.rm = TRUE)) 
  
  if(i == 2) { plot_df$fill <- plot_df$nr_cells; label <- expression("Productivity ("*italic(N)[tot]*")")
  } else { plot_df$fill <- plot_df$nr_groups; label <- expression("Number of groups ("*italic(G)*")") }
  
  p <- plot_df %>% 
    ggplot(aes(x = offspr_size, y = offspr_frac, fill = fill)) +
    geom_tile() +
    cowplot::theme_cowplot() +
    cowplot::panel_border() +
    labs(x = expression("Fractional offspring size ("*italic(s)*")"),
         subtitle = label) +
    theme(aspect.ratio = 1,
          axis.title.y = element_blank(),
          legend.text = element_text(size = 9),
          legend.direction = "horizontal",
          legend.key.width = unit(0.7, "cm"),
          legend.key.height = unit(0.3, "cm"),
          legend.title = element_blank(),
          legend.position = c(0.4, 0.1),
          plot.subtitle = element_text(size = 14)) +
    scale_fill_viridis_c(breaks = scales::pretty_breaks(n = 3)) 
  
  if(i == 1) {
    p <- p + theme(axis.text.y = element_text(),
                   axis.title.y = element_text(angle = 90)) +
      labs(y = expression("Fractional offspring number ("*italic(n)*")")) 
  } else { p <- p + theme(axis.text.y = element_blank()) }
  
  plot_list[[i]] <- p
}

(plot <- egg::ggarrange(plots = plot_list, nrow = 1))

ggsave(filename = here::here("R_figure_code", "scanParSpace", "scan_par_space.pdf"),
       plot = plot,
       width = 7, height = 4)

saveRDS(plot, 
        file = here::here("R_figure_code", "scanParSpace", "plot_scan_par_space_RDS"))

