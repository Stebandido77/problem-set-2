# ==============================================================================
# 10_mejor_modelo.R
# Deck best: importancia de variables, trade-off entre filtración y subcobertura,
# y comparación de política contra el modelo de encuesta corta (nivel E).
# ==============================================================================

source(here::here("scripts", "00_configuracion.R"))

leaderboard <- read_csv(file.path(RUTA_RESULTADOS, "leaderboard_interno.csv"),
                        show_col_types = FALSE, col_types = cols(nivel = col_character()))
bitacora <- read_csv(file.path(RUTA_SUBMISSIONS, "bitacora_kaggle.csv"),
                     show_col_types = FALSE, col_types = cols(nivel = col_character()))

# Mejor modelo: el de mayor F1 público si ya hay scores de Kaggle; si no, el de
# mayor F1 en holdout
candidatos <- leaderboard |>
  filter(algoritmo != "Regla") |>
  left_join(select(bitacora, candidato, f1_kaggle), by = "candidato")
nombre_mejor <- if (any(!is.na(candidatos$f1_kaggle))) {
  candidatos |> slice_max(f1_kaggle, n = 1, with_ties = FALSE) |> pull(candidato)
} else {
  candidatos |> slice_max(f1_holdout, n = 1, with_ties = FALSE) |> pull(candidato)
}

mejor <- readRDS(file.path(RUTA_MODELOS, paste0(nombre_mejor, ".rds")))
message("Mejor modelo: ", mejor$especificacion)

# ---- Importancia de variables ------------------------------------------------
if (!is.null(mejor$importancia)) {
  grafico_importancia <- mejor$importancia |>
    slice_max(importancia, n = 15) |>
    mutate(variable = fct_reorder(variable, importancia)) |>
    ggplot(aes(importancia, variable)) +
    geom_col(fill = "#2a78d6", width = 0.7) +
    labs(
      title = "¿Qué predice la pobreza?",
      subtitle = paste0(nombre_mejor, " | importancia relativa (máximo = 1)"),
      x = "Importancia relativa", y = NULL
    )
  guardar_figura(grafico_importancia, "importancia_variables")
}

# ---- Trade-off de política: filtración vs. subcobertura ----------------------
# Curva con las predicciones fuera de fold (la misma con la que se eligió el umbral)
grafico_umbral <- mejor$curva_umbral |>
  select(umbral, F1 = f1, Filtración = filtracion, Subcobertura = subcobertura) |>
  pivot_longer(-umbral, names_to = "metrica", values_to = "valor") |>
  ggplot(aes(umbral, valor, color = metrica)) +
  geom_line(linewidth = 1) +
  geom_vline(xintercept = mejor$umbral, linetype = "dashed", color = "#52514e") +
  scale_color_manual(values = c(F1 = "#2a78d6", Filtración = "#eb6834", Subcobertura = "#1baf7a")) +
  scale_y_continuous(labels = scales::percent) +
  labs(
    title = "Mover el umbral cambia a quién se deja por fuera",
    subtitle = "Predicciones fuera de fold; la línea punteada es el umbral que maximiza F1",
    x = "Umbral de probabilidad para clasificar como pobre",
    y = NULL, color = NULL
  )
guardar_figura(grafico_umbral, "tradeoff_filtracion_subcobertura")

# Escenario alternativo: un hacedor de política que tolera más filtración
# a cambio de cubrir al menos 90% de los pobres
escenario_cobertura <- mejor$curva_umbral |>
  filter(recall >= 0.90) |>
  slice_max(umbral, n = 1)

tabla_politica <- bind_rows(
  mejor$curva_umbral |> filter(umbral == mejor$umbral) |> mutate(criterio = "Máximo F1"),
  escenario_cobertura |> mutate(criterio = "Cobertura >= 90%")
) |>
  select(criterio, umbral, f1, precision, recall, filtracion, subcobertura)
guardar_tabla(tabla_politica, "escenarios_politica")

# ---- Mejor modelo vs. encuesta corta (holdout) -------------------------------
tabla_encuesta_corta <- leaderboard |>
  filter(candidato == nombre_mejor | nivel == "E") |>
  transmute(
    candidato, algoritmo, n_variables, f1_holdout, precision_holdout, recall_holdout,
    filtracion_n, subcobertura_n, pobres_holdout
  )
guardar_tabla(tabla_encuesta_corta, "mejor_vs_encuesta_corta")
print(tabla_encuesta_corta)
