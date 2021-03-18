library(tidyverse)
library(cowplot)

df <- readRDS(here::here("RcodePlots", "Fig_SX_Pichugin", "data_pichugin_RDS"))
df <- tibble(df)

df_summary <- df %>% 
  group_by(alpha_b, offspr_size, offspr_frac) %>% 
  summarize(r_groups = mean(r_groups),
            r_tot = mean(r_tot)) %>% 
  ungroup()

plot_list <- vector(mode = "list", length = 7)
i <- 1; for (alpha in (df$alpha_b %>% unique %>% rev)){
  p <- df_summary %>% 
    filter(alpha_b == alpha) %>% 
    ggplot(aes(x = offspr_size, y = offspr_frac, fill = r_groups)) +
    geom_tile() +
    cowplot::theme_cowplot() +
    cowplot::panel_border() +
    labs(subtitle = bquote(kappa == .(alpha)),
         x = expression(italic(s))) +
    theme(aspect.ratio = 1,
          axis.title.y = element_blank(),
          axis.title.x = element_text(size = 12),
          axis.text.x = element_text(size = 9),
          legend.text = element_text(size = 8),
          legend.direction = "horizontal",
          legend.key.width = unit(0.45, "cm"),
          legend.key.height = unit(0.1, "cm"),
          legend.title = element_blank(),
          legend.position = c(0.3, 0.1)) +
    scale_fill_viridis_c(breaks = scales::pretty_breaks(n = 3)) 
  
  if(i == 1) {
    p <- p + theme(axis.text.y = element_text(size = 9),
                   axis.title.y = element_text(size = 12)) +
      labs(y = expression(italic(n))) 
  } else { p <- p + theme(axis.text.y = element_blank()) }
  
  plot_list[[i]] <- p
  
  i <- i + 1
}

plot_r_groups <- egg::ggarrange(plots = plot_list, nrow = 1)


plot_list <- vector(mode = "list", length = 7)
i <- 1; for (alpha in (df$alpha_b %>% unique %>% rev)){
  p <- df_summary %>% 
    filter(alpha_b == alpha) %>% 
    ggplot(aes(x = offspr_size, y = offspr_frac, fill = r_tot)) +
    geom_tile() +
    cowplot::theme_cowplot() +
    cowplot::panel_border() +
    labs(subtitle = bquote(kappa == .(alpha)),
         x = expression(italic(s))) +
    theme(aspect.ratio = 1,
          axis.title.y = element_blank(),
          axis.title.x = element_text(size = 12),
          axis.text.x = element_text(size = 9),
          legend.text = element_text(size = 8),
          legend.direction = "horizontal",
          legend.key.width = unit(0.45, "cm"),
          legend.key.height = unit(0.1, "cm"),
          legend.title = element_blank(),
          legend.position = c(0.3, 0.1)) +
    scale_fill_viridis_c(breaks = scales::pretty_breaks(n = 3)) 
  
  if(i == 1) {
    p <- p + theme(axis.text.y = element_text(size = 9),
                   axis.title.y = element_text(size = 12)) +
      labs(y = expression(italic(n))) 
  } else { p <- p + theme(axis.text.y = element_blank()) }
  
  plot_list[[i]] <- p
  
  i <- i + 1
}

plot_r_tot <- egg::ggarrange(plots = plot_list, nrow = 1)

plot_grid(plot_r_groups,
          plot_r_tot,
          nrow = 2,
          labels = c("A", "B"))


ggsave(
  here::here("RcodePlots", "Fig_SX_Pichugin", "pichugin_figure.pdf"),
  width = 13, height = 5)
