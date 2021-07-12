library(tidyverse)

Kcells <- 100
Ktotal <- 10^5
Kgroups <- 10^3

Ni <- seq(1, 100, by = 1)


# Death rate --------------------------------------------------------------
d <- Ni/Kcells
d_SI <- 1/Kcells

p3 <- tibble(Ni = Ni, d = d, d_SI = d_SI) %>% 
  ggplot(aes(x = Ni, y = d)) + 
  geom_line(size = 1) +
  geom_line(aes(y = d_SI), lty = "dashed", size = 1) +
  cowplot::theme_cowplot(font_size = 12) +
  labs(y = expression("Cell death rate ("*italic("d"[i])*")"),
       x = expression("Number of cells in group ("*italic("N")[i]*")"),
       tag = "C") +
  annotate(geom = "text", label = "SI", x = 97.5, y = 0.08, size = 3.5)



# Extinction rate ---------------------------------------------------------
Ntotal <- seq(1, 10^5, by = 1000)

Di <- tibble(Ntotal = Ntotal, Di = Ntotal/Ktotal) %>% 
  ggplot(aes(x = Ntotal/1000, y = Di)) +
  geom_line(size = 1) +
  cowplot::theme_cowplot(font_size = 12) +
  labs(y = expression("Group extinction rate ("*italic("D"[i])*")"),
       x = expression("Total number of cells ("*italic("N")[total]*"), thousands"),
       tag = "D")

G <- seq(1, 1000, by = 10)

SI_inset <- tibble(G = G, Di_SI = G/Kgroups) %>% 
  ggplot(aes(x = G, y = Di_SI)) +
  geom_line(lty = "dashed", size = 1) +
  cowplot::theme_cowplot(font_size = 12) +
  labs(y = expression(italic("D"[i])),
       x = expression("Nr. groups ("*italic("G")*")")) +
  annotate(geom = "text", label = "SI", x = 600, y = 0.9, size = 3.5) +
  scale_y_continuous(breaks = c(0, 0.5, 1)) +
  scale_x_continuous(breaks = c(0, 500, 1000))

p4 <- Di + annotation_custom(ggplotGrob(SI_inset), 
                       xmin = 50, xmax = 100, 
                       ymin = -0.05, ymax = 0.55)


# Fission rate ------------------------------------------------------------

B0 <- 0.05

p2 <- tibble(Ni = Ni, Bi_flat = B0 + 0 * Ni, Bi_slope = B0 + 2 * Ni) %>% 
  ggplot(aes(x = Ni, y = Bi_slope)) +
  geom_line(size = 1) +
  geom_line(aes(y = Bi_flat), size = 1) +
  cowplot::theme_cowplot(font_size = 12) +
  labs(y = expression("Group fission rate ("*italic("B"[i])*")"),
       x = expression("Number of cells in group ("*italic("N")[i]*")"),
       tag = "B") +
  annotate(geom = "text", label = expression(sigma*" = 0 (default)"), x = 86, y = 15, size = 3.5) +
  annotate(geom = "text", label = expression(sigma*" = 2"), x = 90, y = 160, size = 3.5) 


# Birth rate --------------------------------------------------------------

fwt <- seq(0, 1, length = 100)

m_line <- function(fwt, m){(m^(m-1)) * (fwt^(m-1))}

p1 <- tibble(fwt = fwt, m1 = fwt, 
       m2 = m_line(fwt, 2), m3 = m_line(fwt, 3), m4 = m_line(fwt, 4)) %>% 
  ggplot(aes(x = fwt, y = m4)) +
  geom_line(size = 1) +
  geom_line(aes(y = m3), size = 1) +
  geom_line(aes(y = m2), size = 1) +
  geom_line(aes(y = m1),size = 1) +
  cowplot::theme_cowplot(font_size = 12) +
  coord_cartesian(ylim = c(0, 10)) +
  labs(y = expression("Mutant cell birth rate ("*italic("b")[italic("i,j")]^"mut"*")"),
       x = expression("Fraction of WT cells per group ("*italic("n")[italic("i,k")]^"wt"/italic("N"[i])*")"),
       tag = "A") +
  annotate(geom = "text", label = expression(italic("m")*"= 1"), x = 0.95, y = 0.4, size = 3.5) +
  annotate(geom = "text", label = expression(italic("m")*"= 2"), x = 0.95, y = 2.6, size = 3.5) +
  annotate(geom = "text", label = expression(italic("m")*"= 3"), x = 0.98, y = 7, size = 3.5) +
  annotate(geom = "text", label = expression(italic("m")*"= 4"), x = 0.6, y = 9, size = 3.5) 


# Compose -----------------------------------------------------------------

library(patchwork)

(p1 + p2) / (p3 + p4)
ggsave("R_figure_code/Fig_SX_rates/rates_figure.pdf", width = 8, height = 6)
