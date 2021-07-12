library(tidyverse)

slope <- readRDS(file = here::here("R_figure_code", "mutationalMeltdown", "slope_neighbour_avg_RDS"))
migr <-readRDS(file = here::here("R_figure_code", "mutationalMeltdown", "migr_neighbour_avg_RDS"))

slope$gr_SFis <- as.character(slope$gr_SFis)
slope[slope$gr_SFis == "0",]$gr_SFis <- "0.0"
slope$gr_SFis <- factor(slope$gr_SFis, levels = c("0.0", "0.1", "2","4"))

migr$indv_migrR <- as.character(migr$indv_migrR)
migr[migr$indv_migrR == "0",]$indv_migrR <- "0.0"
migr$indv_migrR <- factor(migr$indv_migrR, levels = c("0.0", "0.01", "0.1","1", "5"))

slope <- slope %>% 
  mutate(avg_max_mu = ifelse(avg_max_mu < -10, -10, avg_max_mu)) %>% 
  filter(gr_SFis %in% c("0.0", "2"))

migr <- migr %>% 
  mutate(avg_max_mu = ifelse(avg_max_mu < -10, -10, avg_max_mu)) %>% 
  filter(indv_migrR == "1")

bind_rows(slope, migr) %>% 
  mutate(facet_var = ifelse(!is.na(indv_migrR), "list(italic(sigma) == 0, italic(nu) == 1)",
                            ifelse(gr_SFis == "0.0", "list(italic(sigma) == 0, italic(nu) == 0)",
                                   "list(italic(sigma) == 2, italic(nu) == 0)"))) %>%
  ggplot(aes(x = offspr_size, y = offspr_frac, fill = avg_max_mu)) +
  geom_tile() +
  facet_grid(facet_var ~ glue::glue('italic(m)*" = {indv_NType}"'), labeller = label_parsed) +
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

ggsave(filename = here::here("R_figure_code", "mutationalMeltdown", "meltdown_neighbour_avg.pdf"),
       height = 5.75, 
       width = 8)
