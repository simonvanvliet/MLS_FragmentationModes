library(tidyverse)

slope <- readRDS(file = here::here("R_figure_code", "Rev_mutationalMeltdownNoDensDep", "neighbour_avg_NoDensDep_RDS"))


slope <- slope %>% 
  mutate(avg_max_mu = ifelse(avg_max_mu < -10, -10, avg_max_mu))

ggplot(slope, aes(x = offspr_size, y = offspr_frac, fill = avg_max_mu)) +
geom_tile() +
facet_grid(~glue::glue('italic(m)*" = {indv_NType}"'), labeller = label_parsed) +
labs(x = expression("Fractional offspring size ("*italic(s)*")"),
     y = expression("Fractional offspring number ("*italic(n)*")"),
     fill = "Maximum\nmutation\nrate\n(log)") +
cowplot::theme_cowplot() +
theme(aspect.ratio = 1,
      strip.background = element_blank())+
guides(fill = element_blank()) +
scale_fill_viridis_c() +
cowplot::panel_border() +
scale_x_continuous(breaks = c(0, 0.25, 0.5), labels = c("0", "0.25", "0.5")) +
scale_y_continuous(breaks = c(0, 0.5, 1), labels = c("0", "0.5", "1"))

ggsave(filename = here::here("R_figure_code", "Rev_mutationalMeltdownNoDensDep", "meltdown_neighbour_avg_NoDensDep.pdf"),
       height = 5.75, 
       width = 8)
