# ==============================================================================
# 10_mejor_modelo.R
# Deck best: importancia de variables, trade-off entre filtración y subcobertura,
# y comparación de política contra el modelo de encuesta corta (nivel E).
#
# El mejor modelo se elige siempre por F1 en el holdout interno, nunca por el F1
# público de Kaggle: el público se calcula sobre una parte del test y tiene ruido,
# y elegir con él sería ajustar al leaderboard.
#   - Mejor de R (niveles 1-6): sale de su .rds.
#   - Mejor del nivel 7 (Python): no tiene .rds; su importancia y su curva de umbral
#     las exporta python/etapa4_leaderboard.py en views/tablas/nivel7/.
# ==============================================================================

source(here::here("scripts", "00_configuracion.R"))

leaderboard <- read_csv(file.path(RUTA_RESULTADOS, "leaderboard_interno.csv"),
                        show_col_types = FALSE, col_types = cols(nivel = col_character()))

# ---- Mejor modelo de R ---------------------------------------------------------
nombre_mejor <- leaderboard |>
  filter(algoritmo != "Regla", nivel != "7") |>
  filter(file.exists(file.path(RUTA_MODELOS, paste0(candidato, ".rds")))) |>
  slice_max(f1_holdout, n = 1, with_ties = FALSE) |>
  pull(candidato)

mejor <- readRDS(file.path(RUTA_MODELOS, paste0(nombre_mejor, ".rds")))
message("Mejor modelo de R: ", mejor$especificacion)

if (!is.null(mejor$importancia)) {
  guardar_figura(graficar_importancia(mejor$importancia, nombre_mejor), "importancia_variables")
}

# Curva con las predicciones fuera de fold (la misma con la que se eligió el umbral)
guardar_figura(graficar_tradeoff(mejor$curva_umbral, mejor$umbral),
               "tradeoff_filtracion_subcobertura")
guardar_tabla(tabla_escenarios_politica(mejor$curva_umbral, mejor$umbral), "escenarios_politica")

# ---- Mejor modelo del nivel 7 (Python) -----------------------------------------
# El de mayor F1 en holdout entre los que tienen CSV de submission
ruta_nivel7 <- file.path(RUTA_TABLAS, "nivel7")
nombre_mejor_n7 <- leaderboard |>
  filter(nivel == "7", !is.na(archivo_submission)) |>
  slice_max(f1_holdout, n = 1, with_ties = FALSE) |>
  pull(candidato)

if (length(nombre_mejor_n7) == 1) {
  importancia_n7 <- read_csv(file.path(ruta_nivel7, "mejor_n7_importancia.csv"), show_col_types = FALSE)
  curva_n7 <- read_csv(file.path(ruta_nivel7, "mejor_n7_curva_umbral.csv"), show_col_types = FALSE)
  stopifnot(
    "Las tablas del nivel 7 son de otro candidato: corran python/etapa4_leaderboard.py" =
      all(importancia_n7$candidato == nombre_mejor_n7) && all(curva_n7$candidato == nombre_mejor_n7)
  )
  umbral_n7 <- leaderboard |> filter(candidato == nombre_mejor_n7) |> pull(umbral)
  message("Mejor modelo del nivel 7: ", nombre_mejor_n7)

  guardar_figura(graficar_importancia(importancia_n7, nombre_mejor_n7), "importancia_variables_n7")
  guardar_figura(graficar_tradeoff(curva_n7, umbral_n7), "tradeoff_filtracion_subcobertura_n7")
  guardar_tabla(tabla_escenarios_politica(curva_n7, umbral_n7), "escenarios_politica_n7")
}

# ---- Mejor modelo vs. encuesta corta (holdout) -------------------------------
tabla_encuesta_corta <- leaderboard |>
  filter(candidato %in% c(nombre_mejor, nombre_mejor_n7) | nivel == "E") |>
  transmute(
    candidato, algoritmo, n_variables, f1_holdout, precision_holdout, recall_holdout,
    filtracion_n, subcobertura_n, pobres_holdout
  )
guardar_tabla(tabla_encuesta_corta, "mejor_vs_encuesta_corta")
print(tabla_encuesta_corta)
