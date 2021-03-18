library(tidyverse)
library(patchwork)

landscape <- readRDS(here::here("RcodePlots", "evolution", "landscape_new_RDS"))

landscape <- landscape %>% 
  group_by(indv_NType, gr_SFis, offspr_size, offspr_frac) %>% 
  summarize(ntot = mean(NTot_mav, na.rm = T),
            ngrp = mean(NGrp_mav, na.rm = T),
            size = mean(groupSizeAv_mav, na.rm = T))

long_landscape <- landscape %>% 
  pivot_longer(c(ntot, ngrp, size), names_to = "variable")

# Read in evolution files
evolution <- readRDS(here::here("RcodePlots", "evolution", "evolution_rds"))

evol_df <- evolution %>% 
  separate(file, c("file", "indv_NType"), sep = "nTyp") %>% 
  separate(indv_NType, c("indv_NType", "gr_SFis"), sep = "_fisS") %>% 
  separate(gr_SFis, c("gr_SFis", "file3"), sep = "_si") #%>% 
  #filter(!id %in% c("18", "17", "6"))# %>% 
  #ggplot(aes(x = offspr_size, y = offspr_frac, group = id)) +
  #geom_path(aes(color = time)) +
  #facet_wrap(indv_NType ~ gr_SFis)
  

plots <- list()
j <- 0
for(i in c("ntot", "ngrp", "size")) {
  j <- j+1
  
  plots[[j]] <- long_landscape %>% 
    filter(variable == i) %>% 
    #filter(indv_K == 50) %>% 
    ggplot(aes(x = offspr_size, y = offspr_frac)) +
    geom_tile(aes(fill = value)) +
    cowplot::theme_cowplot() +
    labs(x = expression("Fractional offspring size ("*italic(s)*")"),
         y = expression("Fractional offspring number ("*italic(n)*")")) +
    cowplot::theme_cowplot() +
    theme(aspect.ratio = 1,
          legend.text = element_text(size = 10),
          strip.background = element_blank(),
          panel.spacing.x = unit(4, "mm"),
          legend.key.height = unit(3, "mm")
          ) +
    scale_x_continuous(breaks = c(0, 0.25, 0.5),
                       labels = c("0", "0.25", "0.5")) +
    scale_y_continuous(breaks = c(0, 0.5, 1),
                       labels = c("0", "0.5", "1")) +
    scale_fill_viridis_c(breaks = scales::pretty_breaks(n = 3),
                         labels = scales::label_number_si()) +
    cowplot::panel_border() +
    facet_wrap(
      glue::glue('italic(m)*" = {indv_NType}"') ~ glue::glue('italic(S)*" = {gr_SFis}"'), 
      labeller = label_parsed, nrow = 1) +
    geom_path(data = evol_df, aes(group = id, color = time), show.legend = FALSE) +
    scale_color_gradient(low = "red3", high = "mistyrose")
  
  if(j != 3) plots[[j]] <- plots[[j]] + theme(axis.text.x = element_blank(),
                                              axis.title.x = element_blank(),
                                              axis.ticks.x = element_blank(),
                                              axis.line.x = element_blank())
  if(j != 2) plots[[j]] <- plots[[j]] + theme(axis.title.y = element_blank())
  
  if(j != 1) plots[[j]] <- plots[[j]] + theme(strip.background = element_blank(),
                                              strip.text.x = element_blank())
 
  if(j == 1) plots[[j]] <- plots[[j]] + labs(fill = bquote(italic(N)["tot"]))
  if(j == 2) plots[[j]] <- plots[[j]] + labs(fill = bquote(italic(G)))
  if(j == 3) plots[[j]] <- plots[[j]] + labs(fill = bquote(bar(italic(N)["i"])))
}


plots[[1]] / plots[[2]] / plots[[3]]


ggsave(file = here::here("RcodePlots", "evolution", "evolution_plot.pdf"),
       width = 7, height = 5)
