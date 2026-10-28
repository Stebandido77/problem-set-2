# ==============================================================================
# 13_figuras_presentacion.R
# Figuras nuevas para los decks best y compare (document/slides/). Todas las
# cifras salen de las tablas de views/tablas/, de la base de hogares de train y
# de predicciones ya guardadas; no se entrena nada. Colores de la plantilla Beamer
# de Uniandes. Cada figura se guarda al tamaño exacto en que se incrusta.
# ==============================================================================

source(here::here("scripts", "00_configuracion.R"))

RUTA_FIGURAS_SLIDES <- here("document", "slides", "figuras")
dir.create(RUTA_FIGURAS_SLIDES, showWarnings = FALSE, recursive = TRUE)
RUTA_NIVEL7 <- file.path(RUTA_TABLAS, "nivel7")

# Paleta de la plantilla (beamerthemeUniandes.sty)
COLOR_AMARILLO <- "#FFD100"
COLOR_ACENTO <- "#C79A00"
COLOR_NEGRO <- "#1A1A1A"
COLOR_GRIS <- "#4D4D4D"
COLOR_GRIS_MEDIO <- "#B3B3B3"
COLOR_ROJO <- "#B3261E"

# Ancho de texto de la plantilla Beamer 16:9 a 11 pt (\textwidth = 398,34 pt), en
# pulgadas. Las figuras se guardan a la fracción de ese ancho que ocupan en la
# slide, así la letra que se ve proyectada es la que se define aquí.
ANCHO_TEXTO_SLIDE <- 398.34 / 72.27
TAMANO_BASE_SLIDES <- 16
# Ejes, leyendas y etiquetas van al 80 % de la base (proporción de theme_minimal):
# 12,8 pt, más que el texto de la slide (9-11 pt). En mm para geom_text.
TAMANO_ETIQUETA <- 0.8 * TAMANO_BASE_SLIDES / .pt
# Calibri es más angosta que Arial y deja más espacio a los datos; si no está
# instalada se usa la sans del sistema.
FUENTE_SLIDES <- if ("Calibri" %in% systemfonts::system_fonts()$family) "Calibri" else "sans"
update_geom_defaults("text", list(family = FUENTE_SLIDES, size = TAMANO_ETIQUETA))

# Tema de las slides: el del proyecto con base de 16 pt, fuente compacta y sin
# recortar los títulos de panel.
# Argumentos: tamano_base = tamaño de letra base en puntos.
# Devuelve: tema de ggplot2.
tema_slides <- function(tamano_base = TAMANO_BASE_SLIDES) {
  tema_proyecto(tamano_base) +
    theme(
      text = element_text(family = FUENTE_SLIDES),
      plot.title = element_text(face = "bold", size = rel(0.85)),
      axis.title = element_text(colour = COLOR_GRIS),
      axis.text = element_text(colour = COLOR_NEGRO),
      strip.text = element_text(face = "bold", hjust = 0, size = rel(0.8)),
      strip.clip = "off",
      panel.grid.major.y = element_blank(),
      plot.margin = margin(3, 8, 3, 3)
    )
}

# Formatos con coma decimal, como en el texto de las slides.
formato_f1 <- function(x) formatC(x, format = "f", digits = 3, decimal.mark = ",")
formato_pct <- scales::label_percent(accuracy = 1, decimal.mark = ",", suffix = " %")
formato_num <- scales::label_number(decimal.mark = ",")

# Guarda una figura en document/slides/figuras en PDF (para LaTeX) y PNG, al tamaño
# exacto en que se incrusta (sin escalar).
# Argumentos: grafico, nombre (sin extensión), fraccion = fracción del ancho de
#   texto que ocupa en la slide (la misma del \column), alto = alto en pulgadas.
# Devuelve: el gráfico, invisible.
guardar_figura_slides <- function(grafico, nombre, fraccion = 1, alto = 2) {
  ancho <- fraccion * ANCHO_TEXTO_SLIDE
  ggsave(file.path(RUTA_FIGURAS_SLIDES, paste0(nombre, ".pdf")), grafico,
         width = ancho, height = alto, device = cairo_pdf)
  ggsave(file.path(RUTA_FIGURAS_SLIDES, paste0(nombre, ".png")), grafico,
         width = ancho, height = alto, dpi = 300, bg = "white")
  invisible(grafico)
}

# Lee el F1 de holdout de un candidato en una tabla de views/tablas.
# Argumentos: archivo (ruta relativa a views/tablas), candidato,
#   columna_candidato y columna_f1 (nombres de columna en esa tabla).
# Devuelve: número.
leer_f1 <- function(archivo, candidato, columna_candidato = "candidato",
                    columna_f1 = "f1_holdout") {
  tabla <- read_csv(file.path(RUTA_TABLAS, archivo), show_col_types = FALSE)
  valor <- tabla[[columna_f1]][tabla[[columna_candidato]] == candidato]
  stopifnot(length(valor) == 1)
  valor
}

# ---- 1. Correlatos de la pobreza en train --------------------------------------
train <- readRDS(file.path(RUTA_PROCESADOS, "train_hogar.rds"))

# Tasa de pobreza por categoría de una variable.
# Argumentos: datos, variable (columna sin comillas), panel (título del panel).
# Devuelve: tibble con panel, categoria, tasa y n.
tasa_por <- function(datos, variable, panel) {
  datos |>
    filter(!is.na({{ variable }})) |>
    group_by(categoria = {{ variable }}) |>
    summarise(tasa = mean(pobre == "pobre"), n = n(), .groups = "drop") |>
    mutate(panel = panel)
}

# Solo los dos paneles del título de la slide (trabajo y dependencia); la
# educación del jefe y la vivienda se citan en el texto.
correlatos <- bind_rows(
  train |>
    mutate(tramo = cut(prop_ocupados, c(-Inf, 0, 0.34, 0.5, 0.67, Inf),
                       labels = c("0", "≤ 1/3", "1/3–1/2", "1/2–2/3", "> 2/3"))) |>
    tasa_por(tramo, "Ocupados\npor persona"),
  train |>
    mutate(tramo = cut(tasa_dependencia, c(-Inf, 0, 0.5, 1, 2, Inf),
                       labels = c("0", "≤ 0,5", "0,5–1", "1–2", "> 2"))) |>
    tasa_por(tramo, "Dependientes\npor adulto")
) |>
  mutate(panel = factor(panel, levels = unique(panel)),
         categoria = fct_rev(fct_inorder(as.character(categoria))))

# El promedio (20 %) va en el pie de la slide: una línea aquí cruzaría las etiquetas.
grafico_correlatos <- ggplot(correlatos, aes(tasa, categoria)) +
  geom_col(fill = COLOR_ACENTO, width = 0.7) +
  geom_text(aes(label = formato_pct(tasa)), hjust = -0.1, colour = COLOR_NEGRO) +
  facet_wrap(~panel, scales = "free_y", nrow = 1) +
  scale_x_continuous(expand = expansion(mult = c(0, 0.35))) +
  coord_cartesian(clip = "off") +
  labs(x = NULL, y = NULL) +
  tema_slides() +
  theme(axis.text.x = element_blank(), panel.grid = element_blank(),
        panel.spacing.x = unit(16, "pt"), plot.margin = margin(3, 22, 3, 3))
guardar_figura_slides(grafico_correlatos, "correlatos_pobreza", fraccion = 0.62, alto = 2.05)

# ---- 2. Escalera: cambios de variables vs. de algoritmo ---------------------------
# Siete peldaños: el ingreso estimado (laboral, total y por fuente) va en un solo
# paso y el XGBoost se omite; las sumas por tipo de cambio no se alteran.
escalera <- tribble(
  ~paso, ~tipo, ~f1,
  "Logit, 5 variables de la teoría", "Piso", leer_f1("leaderboard_interno.csv", "n2_logit", "Candidato", "F1 holdout"),
  "Logit, 48 variables de personas", "Variables", leer_f1("leaderboard_interno.csv", "n3_logit", "Candidato", "F1 holdout"),
  "Boosting afinado, mismas 48", "Algoritmo", leer_f1("leaderboard_interno.csv", "n6_lightgbm", "Candidato", "F1 holdout"),
  "+ trabajo y composición del hogar", "Variables", leer_f1("nivel7/ablacion_etapa1.csv", "n7_a1_laborales"),
  "+ ingreso estimado por persona", "Variables", leer_f1("nivel7/etapa6_ablacion.csv", "n7_v3_fuentes_bag5"),
  "+ resúmenes por rol en el hogar", "Variables", leer_f1("nivel7/etapa7_a.csv", "n7_a_agregados_bag5"),
  "Mezcla final", "Algoritmo", leer_f1("nivel7/etapa18_resumen.csv", "n18_mezcla_n13_limpio")
) |>
  mutate(
    delta = f1 - lag(f1),
    paso = factor(paso, levels = rev(paso)),
    tipo = factor(tipo, levels = c("Piso", "Variables", "Algoritmo"))
  )

aporte <- escalera |>
  filter(tipo != "Piso") |>
  group_by(tipo) |>
  summarise(aporte = sum(delta), .groups = "drop")
print(aporte)
write_csv(aporte, file.path(RUTA_FIGURAS_SLIDES, "aporte_variables_algoritmo.csv"))

grafico_escalera <- ggplot(escalera, aes(f1, paso, colour = tipo)) +
  geom_segment(aes(x = 0.4, xend = f1, yend = paso), linewidth = 0.8, alpha = 0.35) +
  geom_point(size = 3.6) +
  geom_text(aes(label = formato_f1(f1)), hjust = -0.35, colour = COLOR_NEGRO) +
  scale_colour_manual(
    values = c("Piso" = COLOR_GRIS_MEDIO, "Variables" = COLOR_ACENTO, "Algoritmo" = COLOR_NEGRO),
    labels = c("Partida", "Cambian las variables", "Cambia el algoritmo"), name = NULL
  ) +
  scale_x_continuous(limits = c(0.4, 0.83)) +
  coord_cartesian(clip = "off") +
  labs(x = NULL, y = NULL) +
  tema_slides() +
  theme(axis.text.x = element_blank(), panel.grid = element_blank(),
        legend.position = "top", legend.location = "plot", legend.justification = "left",
        legend.margin = margin(0, 0, 0, 0), legend.box.spacing = unit(2, "pt"))
guardar_figura_slides(grafico_escalera, "escalera_completa", fraccion = 1, alto = 2.1)

# ---- 3. Por qué hay techo -----------------------------------------------------------
oraculo <- read_csv(file.path(RUTA_NIVEL7, "techo_oraculo.csv"), show_col_types = FALSE) |>
  mutate(escenario = recode(
    escenario,
    "auxiliares predichos (modelo final)" = "Estimado",
    "laboral observado" = "Laboral",
    "total observado" = "Total",
    "laboral y total observados" = "Ambos"
  ), escenario = fct_inorder(escenario))

grafico_oraculo <- ggplot(oraculo, aes(escenario, f1_holdout,
                                       fill = escenario == first(levels(escenario)))) +
  geom_col(width = 0.7) +
  geom_text(aes(label = formato_f1(f1_holdout)), vjust = -0.3) +
  scale_fill_manual(values = c(`TRUE` = COLOR_ACENTO, `FALSE` = COLOR_GRIS_MEDIO), guide = "none") +
  coord_cartesian(ylim = c(0.6, 0.96)) +
  labs(title = "A. Si el modelo viera el ingreso real", x = NULL, y = NULL) +
  tema_slides() +
  theme(axis.text.y = element_blank(), panel.grid = element_blank())

bandas <- read_csv(file.path(RUTA_NIVEL7, "techo_bandas.csv"), show_col_types = FALSE) |>
  filter(banda %in% c("dentro [0.8; 1.2]", "fuera [0.8; 1.2]")) |>
  mutate(banda = recode(banda, "dentro [0.8; 1.2]" = "± 20 % de\nla línea",
                        "fuera [0.8; 1.2]" = "Resto"))

grafico_bandas <- ggplot(bandas, aes(banda, f1, fill = banda)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = formato_f1(f1)), vjust = -0.3) +
  scale_fill_manual(values = c(COLOR_ROJO, COLOR_GRIS_MEDIO), guide = "none") +
  coord_cartesian(ylim = c(0.5, 0.93)) +
  labs(title = "B. Hogares cerca de la línea", x = NULL, y = NULL) +
  tema_slides() +
  theme(axis.text.y = element_blank(), panel.grid = element_blank())

library(patchwork)
grafico_techo <- grafico_oraculo + plot_spacer() + grafico_bandas +
  plot_layout(widths = c(1.4, 0.12, 1))
guardar_figura_slides(grafico_techo, "techo_dos_paneles", fraccion = 1, alto = 1.85)

# Panel C (más datos), al apéndice del deck best
curva <- read_csv(file.path(RUTA_NIVEL7, "techo_curva_aprendizaje.csv"), show_col_types = FALSE)
grafico_curva <- ggplot(curva, aes(hogares_entrenamiento / 1000, f1_holdout)) +
  geom_line(colour = COLOR_ACENTO, linewidth = 1.2) +
  geom_point(colour = COLOR_ACENTO, size = 3.2) +
  geom_text(aes(label = formato_f1(f1_holdout)), vjust = -0.9) +
  scale_y_continuous(labels = formato_num, breaks = c(0.75, 0.77, 0.79)) +
  coord_cartesian(ylim = c(0.75, 0.79), xlim = c(25, 140)) +
  scale_x_continuous(breaks = round(curva$hogares_entrenamiento / 1000)) +
  labs(x = "Miles de hogares", y = "F1") +
  tema_slides() +
  theme(panel.grid.major.y = element_line(colour = "grey92"))
guardar_figura_slides(grafico_curva, "techo_curva_datos", fraccion = 0.6, alto = 2.3)

# ---- 4. Peso de cada grupo de variables en el modelo base (n7_a, 76 % de la mezcla) --
importancia <- read_csv(file.path(RUTA_NIVEL7, "etapa7_a_importancia.csv"), show_col_types = FALSE) |>
  mutate(familia = case_when(
    str_detect(variable, "ingreso_pred|laboral_pc|pensiones_pc|_pc_lp_v3|_pc_v3") ~ "Ingreso estimado",
    str_detect(variable, "arriendo|tenencia|vivienda|cuartos|dormitorios") ~ "Vivienda y arriendo",
    str_detect(variable, "^ag_") ~ "Resúmenes por rol",
    TRUE ~ "Resto"
  ))

participacion_familias <- importancia |>
  group_by(familia) |>
  summarise(share_gain = sum(share_gain), n_variables = n(), .groups = "drop") |>
  arrange(desc(share_gain))
print(participacion_familias)
write_csv(participacion_familias, file.path(RUTA_FIGURAS_SLIDES, "importancia_familias.csv"))
print(slice_max(importancia, share_gain, n = 5))

grafico_importancia <- participacion_familias |>
  mutate(familia = fct_reorder(familia, share_gain),
         destacada = familia == "Ingreso estimado") |>
  ggplot(aes(share_gain, familia, fill = destacada)) +
  geom_col(width = 0.7) +
  geom_text(aes(label = formato_pct(share_gain)), hjust = -0.15) +
  scale_fill_manual(values = c(`TRUE` = COLOR_ACENTO, `FALSE` = COLOR_GRIS_MEDIO), guide = "none") +
  scale_x_continuous(expand = expansion(mult = c(0, 0.2))) +
  coord_cartesian(clip = "off") +
  labs(x = NULL, y = NULL) +
  tema_slides() +
  theme(axis.text.x = element_blank(), panel.grid = element_blank())
guardar_figura_slides(grafico_importancia, "importancia_n7_a", fraccion = 0.56, alto = 1.8)

# ---- 5. Búsqueda del techo: todos los enfoques contra la base ----------------------
resumen16 <- read_csv(file.path(RUTA_NIVEL7, "etapa16_fase1_resumen.csv"), show_col_types = FALSE)
base_holdout <- resumen16$f1_holdout[resumen16$candidato == "n7_a_agregados_bag5"]
emd_holdout <- resumen16$mde_holdout[resumen16$candidato == "n13_mezcla_final"]

enfoques <- tribble(
  ~enfoque, ~grupo, ~f1,
  "Rediseño desde cero", "Otras variables o datos", leer_f1("nivel7/etapa18_resumen.csv", "n18_limpio"),
  "Rangos de ingreso", "Otras variables o datos", leer_f1("nivel7/etapa12_resumen.csv", "n12_cuantiles_directo"),
  "Red por personas", "Otros algoritmos", leer_f1("nivel7/etapa12_resumen.csv", "n12_conjuntos_bag3"),
  "Algoritmo genético", "Otras variables o datos", leer_f1("nivel7/etapa9_candidatos.csv", "n9_ga_bagging"),
  "AutoML, base cruda", "Otras variables o datos", leer_f1("nivel7/etapa12_resumen.csv", "n12_autogluon_a"),
  "Red TabM", "Otros algoritmos", leer_f1("nivel7/etapa10_candidatos.csv", "n10_tabm_bag3"),
  "Árboles aleatorios", "Otros algoritmos", leer_f1("nivel7/etapa7_b.csv", "n7_b2_extratrees_abc_agr"),
  "Efecto por ciudad", "Otros algoritmos", leer_f1("nivel7/etapa12_resumen.csv", "n12_gpboost_dominio_1s"),
  "Red RealMLP", "Otros algoritmos", leer_f1("nivel7/etapa10_candidatos.csv", "n10_realmlp_bag5"),
  "Boosting EBM", "Otros algoritmos", leer_f1("nivel7/etapa15_resumen.csv", "n15_ebm_top60_s0"),
  "AutoML, nuestras", "Otros algoritmos", leer_f1("nivel7/etapa12_resumen.csv", "n12_autogluon_b"),
  "Ingreso reafinado", "Otras variables o datos", leer_f1("nivel7/etapa8_candidatos.csv", "n8_aux_tuneados_bag5"),
  "Variables automáticas", "Otras variables o datos", leer_f1("nivel7/etapa14_resumen.csv", "n14_a_gp_lgb_s0"),
  "CatBoost", "Otros algoritmos", leer_f1("nivel7/etapa7_b.csv", "n7_b3_catboost_agr"),
  "Vecinos cercanos", "Otros algoritmos", leer_f1("nivel7/etapa16_fase1_resumen.csv", "A + KNN (bag5)"),
  "Aprender del test", "Otras variables o datos", leer_f1("nivel7/etapa15_resumen.csv", "n15_pseudo_s0")
) |>
  mutate(enfoque = fct_reorder(enfoque, f1),
         grupo = factor(grupo, levels = c("Otros algoritmos", "Otras variables o datos"),
                        labels = c("Otro algoritmo", "Otras variables")))

# Banda amarilla: LightGBM base ± efecto mínimo detectable en holdout (se explica en la slide)
grafico_enfoques <- ggplot(enfoques, aes(f1, enfoque)) +
  annotate("rect", xmin = base_holdout - emd_holdout, xmax = base_holdout + emd_holdout,
           ymin = -Inf, ymax = Inf, fill = COLOR_AMARILLO, alpha = 0.35) +
  geom_vline(xintercept = base_holdout, colour = COLOR_GRIS, linetype = "dashed") +
  geom_point(size = 3.2, colour = COLOR_NEGRO) +
  facet_wrap(~grupo, scales = "free_y") +
  scale_x_continuous(limits = c(0.745, 0.786), breaks = c(0.75, 0.77), labels = formato_num) +
  labs(x = NULL, y = NULL) +
  tema_slides() +
  theme(panel.grid.major.x = element_line(colour = "grey92"), panel.spacing.x = unit(14, "pt"),
        strip.text = element_text(hjust = 1))
guardar_figura_slides(grafico_enfoques, "busqueda_techo", fraccion = 1, alto = 2.3)

# ---- 6. Política: a quién excluye cada modelo ------------------------------------
bienestar <- read_csv(file.path(RUTA_NIVEL7, "etapa15_pesos_bienestar.csv"), show_col_types = FALSE) |>
  filter(candidato %in% c("n13_mezcla_final", "n3_logit", "encuesta_corta_8")) |>
  transmute(
    modelo = recode(candidato, n13_mezcla_final = "Mejor modelo",
                    n3_logit = "Logit, 48", encuesta_corta_8 = "Encuesta corta"),
    `Cerca` = excluidos_cerca_linea * subcobertura,
    `Extrema` = excluidos_pobreza_extrema * subcobertura,
    `Media` = subcobertura - `Cerca` - `Extrema`,
    subcobertura
  ) |>
  pivot_longer(-c(modelo, subcobertura), names_to = "grupo", values_to = "share") |>
  mutate(
    modelo = factor(modelo, levels = c("Encuesta corta", "Logit, 48", "Mejor modelo")),
    grupo = factor(grupo, levels = c("Extrema", "Media", "Cerca"))
  )

grafico_excluidos <- ggplot(bienestar, aes(share, modelo, fill = grupo)) +
  geom_col(width = 0.65) +
  geom_text(data = distinct(bienestar, modelo, subcobertura),
            aes(subcobertura, modelo, label = formato_pct(subcobertura)),
            inherit.aes = FALSE, hjust = -0.15) +
  scale_fill_manual(values = c(COLOR_ROJO, COLOR_GRIS_MEDIO, COLOR_ACENTO)) +
  scale_x_continuous(limits = c(0, 0.32), expand = expansion(mult = c(0, 0))) +
  coord_cartesian(clip = "off") +
  labs(x = NULL, y = NULL, fill = NULL) +
  tema_slides() +
  theme(axis.text.x = element_blank(), panel.grid = element_blank(),
        legend.position = "top", legend.justification = "left", legend.location = "plot",
        legend.margin = margin(0, 0, 0, 0), legend.key.size = unit(11, "pt"))
guardar_figura_slides(grafico_excluidos, "excluidos_por_modelo", fraccion = 0.6, alto = 1.9)

# ---- 7. Encuesta corta: F1 según el número de preguntas ---------------------------
frontera <- read_csv(file.path(RUTA_NIVEL7, "frontera_encuesta_corta.csv"), show_col_types = FALSE)
f1_logit_48 <- leer_f1("leaderboard_interno.csv", "n3_logit", "Candidato", "F1 holdout")
f1_mejor <- leer_f1("nivel7/etapa18_resumen.csv", "n18_mezcla_n13_limpio")
referencias <- tibble(
  f1 = c(f1_mejor, f1_logit_48),
  etiqueta = c(paste0("Mejor modelo: ", formato_f1(f1_mejor)),
               paste0("Logit, 48: ", formato_f1(f1_logit_48)))
)
marcados <- frontera |> filter(n_variables == 8)
x_maximo <- max(frontera$n_variables)

grafico_frontera <- ggplot(frontera, aes(n_variables, f1_holdout)) +
  geom_hline(data = referencias, aes(yintercept = f1), linetype = "dashed", colour = COLOR_GRIS) +
  geom_text(data = referencias, aes(x = x_maximo, y = f1, label = etiqueta), hjust = 1,
            vjust = c(-0.4, 1.4), colour = COLOR_GRIS) +
  geom_line(colour = COLOR_ACENTO, linewidth = 1.1) +
  geom_point(data = marcados, colour = COLOR_ROJO, size = 3.2) +
  geom_text(data = marcados, aes(label = paste0("8 preguntas: ", formato_f1(f1_holdout))),
            hjust = -0.12, vjust = 1.3, colour = COLOR_ROJO) +
  scale_y_continuous(labels = formato_num, limits = c(0.47, 0.84), breaks = c(0.5, 0.6, 0.7, 0.8)) +
  labs(x = "Preguntas en la encuesta", y = "F1") +
  tema_slides() +
  theme(panel.grid.major.y = element_line(colour = "grey92"))
guardar_figura_slides(grafico_frontera, "frontera_encuesta", fraccion = 0.54, alto = 2.05)

# ---- 8. Umbral del modelo final (n18): curva de cobertura y escenarios ----------
# Los umbrales se eligen con las predicciones OOF (máximo F1 y el umbral más alto
# que cubre al menos al 90 % de los pobres) y se evalúan en el holdout.
predicciones_n18 <- arrow::read_parquet(
  file.path(RUTA_PROCESADOS, "nivel7", "predicciones_n18_mezcla_n13_limpio.parquet")
) |>
  filter(tipo %in% c("oof", "holdout")) |>
  inner_join(transmute(train, id, verdad = as.integer(pobre == "pobre")), by = "id")
stopifnot(nrow(predicciones_n18) == 131967 + 32993)

# Precisión, recall y F1 de clasificar como pobre a quien tenga probabilidad >= umbral.
# Argumentos: verdad (0/1), probabilidad, umbral.
# Devuelve: tibble de una fila con umbral, f1, precision, recall, filtracion,
#   subcobertura y share_atendida.
metricas_umbral <- function(verdad, probabilidad, umbral) {
  incluido <- probabilidad >= umbral
  vp <- sum(incluido & verdad == 1)
  precision <- vp / sum(incluido)
  recall <- vp / sum(verdad == 1)
  tibble(umbral = umbral, f1 = 2 * precision * recall / (precision + recall),
         precision = precision, recall = recall, filtracion = 1 - precision,
         subcobertura = 1 - recall, share_atendida = mean(incluido))
}

oof_n18 <- filter(predicciones_n18, tipo == "oof")
holdout_n18 <- filter(predicciones_n18, tipo == "holdout")
curva_oof <- map(seq(0.05, 0.80, by = 0.01),
                 \(u) metricas_umbral(oof_n18$verdad, oof_n18$probabilidad, u)) |>
  list_rbind()
umbrales_n18 <- c(
  "Máximo F1" = curva_oof$umbral[which.max(curva_oof$f1)],
  "Cobertura >= 90%" = max(curva_oof$umbral[curva_oof$recall >= 0.90])
)
escenarios_n18 <- imap(umbrales_n18, \(u, criterio) {
  metricas_umbral(holdout_n18$verdad, holdout_n18$probabilidad, u) |>
    mutate(criterio = criterio, .before = 1)
}) |>
  list_rbind()
print(escenarios_n18)
write_csv(escenarios_n18, file.path(RUTA_NIVEL7, "umbral_n18_escenarios.csv"))

# Curva de cobertura en el holdout: hogares atendidos de mayor a menor probabilidad
cobertura_n18 <- holdout_n18 |>
  arrange(desc(probabilidad)) |>
  mutate(share_atendida = row_number() / n(),
         share_pobres_cubiertos = cumsum(verdad) / sum(verdad))
write_csv(
  cobertura_n18 |> filter(row_number() %% 33 == 0) |> select(share_atendida, share_pobres_cubiertos),
  file.path(RUTA_NIVEL7, "umbral_n18_curva_cobertura.csv")
)
# Las etiquetas van abajo a la derecha de cada punto, escalonadas para no cruzarse
# con la curva ni entre sí.
puntos_n18 <- escenarios_n18 |>
  mutate(etiqueta = paste0("umbral ", formatC(umbral, format = "f", digits = 2, decimal.mark = ",")),
         x_etiqueta = share_atendida + 0.02,
         y_etiqueta = if_else(criterio == "Máximo F1", recall - 0.06, recall - 0.03))
tasa_holdout <- mean(holdout_n18$verdad)

grafico_cobertura <- ggplot(cobertura_n18, aes(share_atendida, share_pobres_cubiertos)) +
  annotate("segment", x = 0, y = 0, xend = tasa_holdout, yend = 1, linetype = "dotted", colour = COLOR_GRIS) +
  annotate("segment", x = tasa_holdout, y = 1, xend = 0.6, yend = 1, linetype = "dotted", colour = COLOR_GRIS) +
  geom_line(colour = COLOR_ACENTO, linewidth = 1.2) +
  geom_point(data = puntos_n18, aes(share_atendida, recall), colour = COLOR_NEGRO, size = 3.2) +
  geom_text(data = puntos_n18, aes(x_etiqueta, y_etiqueta, label = etiqueta),
            hjust = 0, vjust = 1) +
  scale_x_continuous(labels = formato_pct, limits = c(0, 0.6), breaks = c(0, 0.2, 0.4, 0.6)) +
  scale_y_continuous(labels = formato_pct, limits = c(0, 1.02), breaks = c(0, 0.5, 1)) +
  labs(x = "Hogares atendidos", y = "Pobres cubiertos") +
  tema_slides() +
  theme(panel.grid.major.y = element_line(colour = "grey92"))
guardar_figura_slides(grafico_cobertura, "curva_cobertura_slides", fraccion = 0.54, alto = 2.2)
