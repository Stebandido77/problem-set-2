# ==============================================================================
# 13_figuras_presentacion.R
# Figuras nuevas para los decks best y compare (document/slides/). Todas las
# cifras salen de las tablas de views/tablas/ y de la base de hogares de train;
# no se entrena nada. Colores de la plantilla Beamer de Uniandes.
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

# Tema de las slides: el del proyecto con letra más grande, porque las figuras
# se ven a media pantalla.
# Argumentos: tamano_base = tamaño de letra base.
# Devuelve: tema de ggplot2.
tema_slides <- function(tamano_base = 14) {
  tema_proyecto(tamano_base) +
    theme(
      plot.title = element_text(face = "bold", size = rel(1.05)),
      plot.subtitle = element_text(colour = COLOR_GRIS),
      axis.title = element_text(colour = COLOR_GRIS),
      panel.grid.major.y = element_blank(),
      strip.text = element_text(face = "bold", hjust = 0)
    )
}

# Formatos con coma decimal, como en el texto de las slides.
formato_f1 <- function(x) formatC(x, format = "f", digits = 3, decimal.mark = ",")
formato_pct <- scales::label_percent(accuracy = 1, decimal.mark = ",")
formato_pct1 <- scales::label_percent(accuracy = 0.1, decimal.mark = ",")
formato_num <- scales::label_number(decimal.mark = ",")

# Guarda una figura en document/slides/figuras en PDF (para LaTeX) y PNG.
# Argumentos: grafico, nombre (sin extensión), ancho y alto en pulgadas,
#   escala (< 1 agranda el texto frente al lienzo; las figuras se ven chicas en
#   una columna de la slide).
# Devuelve: el gráfico, invisible.
guardar_figura_slides <- function(grafico, nombre, ancho = 8, alto = 4.2, escala = 0.68) {
  ggsave(file.path(RUTA_FIGURAS_SLIDES, paste0(nombre, ".pdf")), grafico,
         width = ancho, height = alto, scale = escala, device = cairo_pdf)
  ggsave(file.path(RUTA_FIGURAS_SLIDES, paste0(nombre, ".png")), grafico,
         width = ancho, height = alto, scale = escala, dpi = 300, bg = "white")
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
tasa_media <- mean(train$pobre == "pobre")

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

correlatos <- bind_rows(
  train |>
    filter(tenencia != "otra") |>
    mutate(tenencia = fct_recode(
      factor(tenencia), "Propia\npagada" = "propia_pagada",
      "Propia\npagando" = "propia_pagando", "Arriendo" = "arriendo",
      "Usufructo" = "usufructo", "Sin\ntítulo" = "posesion_sin_titulo"
    )) |>
    tasa_por(tenencia, "Tenencia de la vivienda"),
  train |>
    mutate(tramo = cut(prop_ocupados, c(-Inf, 0, 0.34, 0.5, 0.67, Inf),
                       labels = c("0", "≤ 1/3", "1/3–1/2", "1/2–2/3", "> 2/3"))) |>
    tasa_por(tramo, "Proporción de ocupados en el hogar"),
  train |>
    mutate(tramo = cut(tasa_dependencia, c(-Inf, 0, 0.5, 1, 2, Inf),
                       labels = c("0", "≤ 0,5", "0,5–1", "1–2", "> 2"))) |>
    tasa_por(tramo, "Dependientes por adulto en edad de trabajar"),
  train |>
    filter(jefe_educ %in% c("ninguno", "primaria", "secundaria", "media", "superior")) |>
    mutate(jefe_educ = factor(str_to_sentence(as.character(jefe_educ)),
                              levels = c("Ninguno", "Primaria", "Secundaria", "Media", "Superior"))) |>
    tasa_por(jefe_educ, "Educación del jefe de hogar")
) |>
  mutate(panel = factor(panel, levels = unique(panel)))

# El orden de las categorías se fija con un factor por panel
correlatos <- correlatos |>
  group_by(panel) |>
  mutate(orden = row_number()) |>
  ungroup() |>
  mutate(etiqueta = fct_reorder(paste(panel, categoria, sep = "__"), as.numeric(panel) * 10 + orden))

grafico_correlatos <- ggplot(correlatos, aes(etiqueta, tasa)) +
  geom_col(fill = COLOR_ACENTO, width = 0.7) +
  geom_hline(yintercept = tasa_media, linetype = "dashed", colour = COLOR_GRIS) +
  geom_text(aes(label = formato_pct(tasa)), vjust = -0.3,
            size = 3.6, colour = COLOR_NEGRO) +
  facet_wrap(~panel, scales = "free_x", nrow = 2) +
  scale_x_discrete(labels = function(x) str_remove(x, "^.*__")) +
  scale_y_continuous(labels = formato_pct, expand = expansion(mult = c(0, 0.15))) +
  labs(
    title = NULL,
    subtitle = sprintf("Tasa de pobreza en train (164.960 hogares). Línea punteada: promedio (%s)",
                       formato_pct1(tasa_media)),
    x = NULL, y = "Tasa de pobreza"
  ) +
  tema_slides(13) +
  theme(panel.grid.major.x = element_blank(), panel.grid.major.y = element_line(colour = "grey92"))
guardar_figura_slides(grafico_correlatos, "correlatos_pobreza", ancho = 9.5, alto = 5.6, escala = 0.85)

# ---- 2. Escalera completa: cambios de variables vs. de algoritmo -----------------
escalera <- tribble(
  ~paso, ~tipo, ~f1,
  "Todos pobres", "Piso", leer_f1("leaderboard_interno.csv", "n1_todos_pobres", "Candidato", "F1 holdout"),
  "Logit con 5 variables teóricas", "Piso", leer_f1("leaderboard_interno.csv", "n2_logit", "Candidato", "F1 holdout"),
  "Logit + 48 variables desde personas", "Variables", leer_f1("leaderboard_interno.csv", "n3_logit", "Candidato", "F1 holdout"),
  "XGBoost, mismas 48 variables", "Algoritmo", leer_f1("leaderboard_interno.csv", "n4_xgboost", "Candidato", "F1 holdout"),
  "LightGBM afinado (50 configuraciones)", "Algoritmo", leer_f1("leaderboard_interno.csv", "n6_lightgbm", "Candidato", "F1 holdout"),
  "+ variables laborales y de composición", "Variables", leer_f1("nivel7/ablacion_etapa1.csv", "n7_a1_laborales"),
  "+ auxiliar de ingreso laboral", "Variables", leer_f1("nivel7/ablacion_etapa1.csv", "n7_a4_aux_personas"),
  "+ auxiliar de ingreso total", "Variables", leer_f1("nivel7/resumen_etapa2.csv", "n7_n6param_a4_total_bag5"),
  "+ auxiliares por fuente de ingreso", "Variables", leer_f1("nivel7/etapa6_ablacion.csv", "n7_v3_fuentes_bag5"),
  "+ 1.026 agregados por rol en el hogar", "Variables", leer_f1("nivel7/etapa7_a.csv", "n7_a_agregados_bag5"),
  "Mezcla con redes y modelo limpio (final)", "Algoritmo", leer_f1("nivel7/etapa18_resumen.csv", "n18_mezcla_n13_limpio")
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
  geom_segment(aes(x = 0.3, xend = f1, yend = paso), linewidth = 0.6, alpha = 0.35) +
  geom_point(size = 3.6) +
  geom_text(aes(label = formato_f1(f1)), hjust = -0.35, size = 3.8, colour = COLOR_NEGRO) +
  scale_colour_manual(
    values = c("Piso" = COLOR_GRIS_MEDIO, "Variables" = COLOR_ACENTO, "Algoritmo" = COLOR_NEGRO),
    labels = c("Piso", "Cambia variables", "Cambia algoritmo"), name = NULL
  ) +
  scale_x_continuous(limits = c(0.3, 0.86), breaks = seq(0.3, 0.8, 0.1), labels = formato_num) +
  labs(x = "F1 en el holdout", y = NULL, colour = NULL) +
  tema_slides(13) +
  theme(panel.grid.major.y = element_blank(), legend.position = "top", legend.location = "plot",
        legend.background = element_rect(fill = "white", colour = NA))
guardar_figura_slides(grafico_escalera, "escalera_completa", ancho = 9, alto = 4.8, escala = 0.7)

# ---- 3. Por qué hay techo -----------------------------------------------------------
oraculo <- read_csv(file.path(RUTA_NIVEL7, "techo_oraculo.csv"), show_col_types = FALSE) |>
  mutate(escenario = recode(
    escenario,
    "auxiliares predichos (modelo final)" = "Predichos\n(lo que\ntenemos)",
    "laboral observado" = "Laboral\nobservado",
    "total observado" = "Total\nobservado",
    "laboral y total observados" = "Laboral y\ntotal\nobservados"
  ), escenario = fct_inorder(escenario))

grafico_oraculo <- ggplot(oraculo, aes(escenario, f1_holdout,
                                       fill = escenario == first(levels(escenario)))) +
  geom_col(width = 0.65) +
  geom_text(aes(label = formato_f1(f1_holdout)), vjust = -0.3, size = 3.8) +
  scale_fill_manual(values = c(`TRUE` = COLOR_ACENTO, `FALSE` = COLOR_GRIS_MEDIO), guide = "none") +
  scale_y_continuous(labels = formato_num) +
  coord_cartesian(ylim = c(0.6, 0.93)) +
  labs(title = "A. Con el ingreso observado", x = NULL, y = "F1 en holdout") +
  tema_slides(12) +
  theme(panel.grid.major.x = element_blank(), axis.text.x = element_text(size = 8.5, lineheight = 0.9))

bandas <- read_csv(file.path(RUTA_NIVEL7, "techo_bandas.csv"), show_col_types = FALSE) |>
  filter(banda %in% c("dentro [0.8; 1.2]", "fuera [0.8; 1.2]")) |>
  mutate(
    share_pobres = pobres / sum(pobres),
    banda = recode(banda, "dentro [0.8; 1.2]" = "Ingreso a ±20 %\nde la línea",
                   "fuera [0.8; 1.2]" = "Resto de\nhogares"),
    etiqueta = paste0(formato_f1(f1), "\n(", formato_pct(share_pobres), " de\nlos pobres)")
  )

grafico_bandas <- ggplot(bandas, aes(banda, f1, fill = banda)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = etiqueta), vjust = -0.15, size = 3.4, lineheight = 0.9) +
  scale_fill_manual(values = c(COLOR_ROJO, COLOR_GRIS_MEDIO), guide = "none") +
  scale_y_continuous(labels = formato_num) +
  coord_cartesian(ylim = c(0.5, 1)) +
  labs(title = "B. Cerca de la línea", x = NULL, y = "F1 en holdout") +
  tema_slides(12) +
  theme(panel.grid.major.x = element_blank(), axis.text.x = element_text(size = 8.5, lineheight = 0.9))

curva <- read_csv(file.path(RUTA_NIVEL7, "techo_curva_aprendizaje.csv"), show_col_types = FALSE)
grafico_curva <- ggplot(curva, aes(hogares_entrenamiento / 1000, f1_holdout)) +
  geom_line(colour = COLOR_ACENTO, linewidth = 1) +
  geom_point(colour = COLOR_ACENTO, size = 2.8) +
  geom_text(aes(label = formato_f1(f1_holdout)), vjust = -0.9, size = 3.4) +
  scale_y_continuous(labels = formato_num) +
  coord_cartesian(ylim = c(0.74, 0.80), xlim = c(25, 140)) +
  scale_x_continuous(breaks = round(curva$hogares_entrenamiento / 1000)) +
  labs(title = "C. Con más datos", x = "Miles de hogares", y = "F1 en holdout") +
  tema_slides(12) +
  theme(panel.grid.major.y = element_line(colour = "grey92"))

if (requireNamespace("patchwork", quietly = TRUE)) {
  library(patchwork)
  grafico_techo <- grafico_oraculo + grafico_bandas + grafico_curva +
    plot_layout(widths = c(1.35, 0.9, 1))
  guardar_figura_slides(grafico_techo, "techo_tres_paneles", ancho = 11, alto = 4.4, escala = 0.72)
}

# ---- 4. Importancia de variables del modelo base (n7_a, 76 % de la mezcla) ---------
importancia <- read_csv(file.path(RUTA_NIVEL7, "etapa7_a_importancia.csv"), show_col_types = FALSE) |>
  mutate(familia = case_when(
    str_detect(variable, "ingreso_pred|laboral_pc|pensiones_pc|_pc_lp_v3|_pc_v3") ~ "Ingreso predicho (auxiliares)",
    str_detect(variable, "arriendo|tenencia|vivienda|cuartos|dormitorios") ~ "Vivienda y arriendo",
    str_detect(variable, "^ag_") ~ "Agregados por rol",
    TRUE ~ "Demografía, trabajo y ubicación"
  ))

participacion_familias <- importancia |>
  group_by(familia) |>
  summarise(share_gain = sum(share_gain), n_variables = n(), .groups = "drop") |>
  arrange(desc(share_gain))
print(participacion_familias)
write_csv(participacion_familias, file.path(RUTA_FIGURAS_SLIDES, "importancia_familias.csv"))

nombres_legibles <- c(
  log_ingreso_pred_pc_lp_v3 = "Ingreso per cápita predicho / línea (fuentes)",
  log_ingreso_pred_pc_lp_total = "Ingreso per cápita predicho / línea (total)",
  log_ingreso_pred_pc_v3 = "Ingreso per cápita predicho (fuentes)",
  arriendo_por_persona_lp = "Arriendo por persona / línea",
  tenencia_arriendo = "Vive en arriendo",
  log_arriendo_relativo = "Arriendo relativo a su ciudad",
  log_ingreso_pred_pc_total = "Ingreso per cápita predicho (total)",
  tenencia_propia_pagada = "Vivienda propia pagada",
  log_costo_vivienda_relativo = "Costo de vivienda relativo",
  tenencia_usufructo = "Vivienda en usufructo"
)

top_importancia <- importancia |>
  slice_max(share_gain, n = 10) |>
  mutate(nombre = coalesce(nombres_legibles[variable], variable),
         nombre = fct_reorder(nombre, share_gain))

grafico_importancia <- ggplot(top_importancia, aes(share_gain, nombre, fill = familia)) +
  geom_col(width = 0.7) +
  geom_text(aes(label = formato_pct1(share_gain)), hjust = -0.15, size = 3.6) +
  scale_fill_manual(values = c("Ingreso predicho (auxiliares)" = COLOR_ACENTO,
                               "Vivienda y arriendo" = COLOR_GRIS)) +
  scale_x_continuous(labels = formato_pct, expand = expansion(mult = c(0, 0.22))) +
  labs(x = "Participación en la ganancia de los árboles", y = NULL, fill = NULL) +
  tema_slides(13) +
  theme(legend.position = c(0.75, 0.3), legend.direction = "vertical",
        legend.background = element_rect(fill = "white", colour = NA))
guardar_figura_slides(grafico_importancia, "importancia_n7_a", ancho = 9, alto = 4.6, escala = 0.8)

# ---- 5. Búsqueda del techo: todos los enfoques contra la base ----------------------
resumen16 <- read_csv(file.path(RUTA_NIVEL7, "etapa16_fase1_resumen.csv"), show_col_types = FALSE)
base_holdout <- resumen16$f1_holdout[resumen16$candidato == "n7_a_agregados_bag5"]
emd_holdout <- resumen16$mde_holdout[resumen16$candidato == "n13_mezcla_final"]

enfoques <- tribble(
  ~enfoque, ~f1,
  "Rediseño desde cero", leer_f1("nivel7/etapa18_resumen.csv", "n18_limpio"),
  "Cuantiles del ingreso", leer_f1("nivel7/etapa12_resumen.csv", "n12_cuantiles_directo"),
  "DeepSets (red)", leer_f1("nivel7/etapa12_resumen.csv", "n12_conjuntos_bag3"),
  "GA-bagging", leer_f1("nivel7/etapa9_candidatos.csv", "n9_ga_bagging"),
  "AutoGluon, base cruda", leer_f1("nivel7/etapa12_resumen.csv", "n12_autogluon_a"),
  "TabM (red)", leer_f1("nivel7/etapa10_candidatos.csv", "n10_tabm_bag3"),
  "ExtraTrees + abejas", leer_f1("nivel7/etapa7_b.csv", "n7_b2_extratrees_abc_agr"),
  "GPBoost por ciudad", leer_f1("nivel7/etapa12_resumen.csv", "n12_gpboost_dominio_1s"),
  "RealMLP (red)", leer_f1("nivel7/etapa10_candidatos.csv", "n10_realmlp_bag5"),
  "EBM", leer_f1("nivel7/etapa15_resumen.csv", "n15_ebm_top60_s0"),
  "AutoGluon, nuestra base", leer_f1("nivel7/etapa12_resumen.csv", "n12_autogluon_b"),
  "Auxiliares con Optuna", leer_f1("nivel7/etapa8_candidatos.csv", "n8_aux_tuneados_bag5"),
  "Programación genética", leer_f1("nivel7/etapa14_resumen.csv", "n14_a_gp_lgb_s0"),
  "CatBoost", leer_f1("nivel7/etapa7_b.csv", "n7_b3_catboost_agr"),
  "Vecinos cercanos", leer_f1("nivel7/etapa16_fase1_resumen.csv", "A + KNN (bag5)"),
  "Pseudoetiquetado", leer_f1("nivel7/etapa15_resumen.csv", "n15_pseudo_s0"),
  "Mezcla final", leer_f1("nivel7/etapa18_resumen.csv", "n18_mezcla_n13_limpio")
) |>
  mutate(enfoque = fct_reorder(enfoque, f1), es_final = enfoque == "Mezcla final")

# Banda amarilla: LightGBM base ± efecto mínimo detectable en holdout (se explica en la slide)
grafico_enfoques <- ggplot(enfoques, aes(f1, enfoque, colour = es_final)) +
  annotate("rect", xmin = base_holdout - emd_holdout, xmax = base_holdout + emd_holdout,
           ymin = -Inf, ymax = Inf, fill = COLOR_AMARILLO, alpha = 0.3) +
  geom_vline(xintercept = base_holdout, colour = COLOR_GRIS, linetype = "dashed") +
  geom_point(size = 3) +
  geom_text(aes(label = formato_f1(f1)), hjust = 1.35, size = 3.4, colour = COLOR_GRIS) +
  scale_colour_manual(values = c(`FALSE` = COLOR_NEGRO, `TRUE` = COLOR_ACENTO), guide = "none") +
  scale_x_continuous(limits = c(0.742, 0.785), breaks = seq(0.75, 0.78, 0.01), labels = formato_num) +
  labs(x = "F1 en el holdout", y = NULL) +
  tema_slides(13)
guardar_figura_slides(grafico_enfoques, "busqueda_techo", ancho = 6.4, alto = 4.6, escala = 0.66)

# ---- 6. Política: a quién excluye cada modelo ------------------------------------
bienestar <- read_csv(file.path(RUTA_NIVEL7, "etapa15_pesos_bienestar.csv"), show_col_types = FALSE) |>
  filter(candidato %in% c("n13_mezcla_final", "n3_logit", "encuesta_corta_8")) |>
  transmute(
    modelo = recode(candidato, n13_mezcla_final = "Mejor modelo",
                    n3_logit = "Logit, 48 variables", encuesta_corta_8 = "Encuesta corta, 8 variables"),
    `Cerca de la línea\n(≥ 0,8 de la línea)` = excluidos_cerca_linea * subcobertura,
    `Pobreza extrema\n(< 0,5 de la línea)` = excluidos_pobreza_extrema * subcobertura,
    `Intermedios` = subcobertura - `Cerca de la línea\n(≥ 0,8 de la línea)` - `Pobreza extrema\n(< 0,5 de la línea)`,
    subcobertura
  ) |>
  pivot_longer(-c(modelo, subcobertura), names_to = "grupo", values_to = "share") |>
  mutate(
    modelo = factor(modelo, levels = c("Encuesta corta, 8 variables", "Logit, 48 variables", "Mejor modelo")),
    grupo = factor(grupo, levels = c("Pobreza extrema\n(< 0,5 de la línea)", "Intermedios",
                                     "Cerca de la línea\n(≥ 0,8 de la línea)"))
  )

grafico_excluidos <- ggplot(bienestar, aes(share, modelo, fill = grupo)) +
  geom_col(width = 0.6) +
  geom_text(data = distinct(bienestar, modelo, subcobertura),
            aes(subcobertura, modelo, label = formato_pct1(subcobertura)),
            inherit.aes = FALSE, hjust = -0.2, size = 3.8) +
  scale_fill_manual(values = c(COLOR_ROJO, COLOR_GRIS_MEDIO, COLOR_ACENTO)) +
  scale_x_continuous(labels = formato_pct, limits = c(0, 0.3)) +
  labs(x = "Pobres excluidos (% de los pobres del holdout)", y = NULL, fill = NULL) +
  tema_slides(13) +
  guides(fill = guide_legend(reverse = TRUE))
guardar_figura_slides(grafico_excluidos, "excluidos_por_modelo", ancho = 9, alto = 3.6)

# ---- 7. Encuesta corta: F1 según el número de preguntas ---------------------------
frontera <- read_csv(file.path(RUTA_NIVEL7, "frontera_encuesta_corta.csv"), show_col_types = FALSE)
f1_logit_48 <- leer_f1("leaderboard_interno.csv", "n3_logit", "Candidato", "F1 holdout")
f1_mejor <- leer_f1("nivel7/etapa18_resumen.csv", "n18_mezcla_n13_limpio")
referencias <- tibble(
  f1 = c(f1_mejor, f1_logit_48),
  etiqueta = c(paste0("Mejor modelo (", formato_f1(f1_mejor), ")"),
               paste0("Logit con 48 variables (", formato_f1(f1_logit_48), ")"))
)
marcados <- frontera |> filter(n_variables %in% c(5, 8, 15))

grafico_frontera <- ggplot(frontera, aes(n_variables, f1_holdout)) +
  geom_hline(data = referencias, aes(yintercept = f1), linetype = "dashed", colour = COLOR_GRIS) +
  geom_text(data = referencias, aes(x = c(30, 50), y = f1, label = etiqueta), hjust = 0, vjust = c(-0.6, 1.6),
            size = 3.6, colour = COLOR_GRIS) +
  geom_line(colour = COLOR_ACENTO, linewidth = 0.9) +
  geom_point(colour = COLOR_ACENTO, size = 1.8) +
  geom_point(data = marcados, colour = COLOR_ROJO, size = 3) +
  geom_text(data = marcados, aes(label = paste0(n_variables, " preguntas: ", formato_f1(f1_holdout))),
            hjust = -0.12, vjust = 1.3, size = 3.6, colour = COLOR_ROJO) +
  scale_y_continuous(labels = formato_num, limits = c(0.47, 0.80)) +
  labs(x = "Número de variables de la encuesta", y = "F1 en holdout (Logit)") +
  tema_slides(14) +
  theme(panel.grid.major.y = element_line(colour = "grey92"))
guardar_figura_slides(grafico_frontera, "frontera_encuesta", ancho = 8, alto = 5.2)

# ---- 8. Curva de cobertura: pobres cubiertos según hogares atendidos --------------
cobertura <- read_csv(file.path(RUTA_NIVEL7, "etapa15_curva_cobertura.csv"), show_col_types = FALSE) |>
  mutate(modelo = recode(candidato, n13_mezcla_final = "Mejor modelo",
                         n3_logit = "Logit, 48 variables", encuesta_corta_8 = "Encuesta corta, 8 variables"),
         modelo = factor(modelo, levels = c("Mejor modelo", "Logit, 48 variables", "Encuesta corta, 8 variables")))
puntos_cobertura <- read_csv(file.path(RUTA_NIVEL7, "etapa15_curva_cobertura_puntos.csv"), show_col_types = FALSE) |>
  mutate(modelo = recode(candidato, n13_mezcla_final = "Mejor modelo",
                         n3_logit = "Logit, 48 variables", encuesta_corta_8 = "Encuesta corta, 8 variables"),
         modelo = factor(modelo, levels = levels(cobertura$modelo)))
tasa_holdout <- 0.2

grafico_cobertura <- ggplot(cobertura, aes(share_atendida, share_pobres_cubiertos, colour = modelo)) +
  annotate("segment", x = 0, y = 0, xend = tasa_holdout, yend = 1, linetype = "dotted", colour = COLOR_GRIS) +
  annotate("segment", x = tasa_holdout, y = 1, xend = 1, yend = 1, linetype = "dotted", colour = COLOR_GRIS) +
  annotate("text", x = 0.215, y = 1.02, label = "focalización perfecta", hjust = 0, size = 3.6,
           colour = COLOR_GRIS) +
  geom_line(linewidth = 1) +
  geom_point(data = puntos_cobertura, size = 3) +
  scale_colour_manual(values = c(COLOR_ACENTO, COLOR_NEGRO, COLOR_ROJO)) +
  scale_x_continuous(labels = formato_pct, limits = c(0, 0.6), breaks = seq(0, 0.6, 0.1)) +
  scale_y_continuous(labels = formato_pct, limits = c(0, 1.04)) +
  labs(x = "Hogares atendidos (de mayor a menor probabilidad)", y = "Pobres cubiertos",
       colour = NULL) +
  tema_slides(14) +
  theme(panel.grid.major.y = element_line(colour = "grey92"), legend.position = c(0.68, 0.22),
        legend.background = element_rect(fill = "white", colour = NA))
guardar_figura_slides(grafico_cobertura, "curva_cobertura_slides", ancho = 8, alto = 5.4)
