# ==============================================================================
# 08_mejor_modelo.R
# Stages 3 y 4 / deck best: importancia de variables, trade-off entre
# filtración y subcobertura, y qué pasaría con otro umbral de política.
# ==============================================================================

source(here::here("scripts", "00_configuracion.R"))

# Actualizar con el .rds de la corrida que tuvo el mejor F1 público en Kaggle
ARCHIVO_MEJOR <- read_csv(file.path(RUTA_RESULTADOS, "metricas_cv.csv"), show_col_types = FALSE) |>
  slice_max(f1, n = 1, with_ties = FALSE) |>
  pull(archivo_rds)

mejor <- readRDS(file.path(RUTA_RESULTADOS, ARCHIVO_MEJOR))
message("Mejor modelo: ", mejor$especificacion)

# ---- Importancia de variables ------------------------------------------------
# El LPM se guarda como lm(); los demás como workflow de tidymodels
modelo_ajustado <- if (inherits(mejor$modelo_final, "workflow")) {
  extract_fit_parsnip(mejor$modelo_final)
} else {
  mejor$modelo_final
}

grafico_importancia <- modelo_ajustado |>
  vip(num_features = 15, aesthetics = list(fill = "#2c7fb8")) +
  labs(
    title = "¿Qué predice la pobreza?",
    subtitle = mejor$especificacion,
    x = NULL, y = "Importancia"
  )
guardar_figura(grafico_importancia, "importancia_variables")

# ---- Trade-off de política: filtración vs. subcobertura ----------------------
grafico_umbral <- mejor$curva_umbral |>
  select(umbral, F1 = f1, Filtración = filtracion, Subcobertura = subcobertura) |>
  pivot_longer(-umbral, names_to = "metrica", values_to = "valor") |>
  ggplot(aes(umbral, valor, color = metrica)) +
  geom_line(linewidth = 1) +
  geom_vline(xintercept = mejor$umbral, linetype = "dashed") +
  scale_y_continuous(labels = scales::percent) +
  labs(
    title = "Mover el umbral cambia a quién se deja por fuera",
    subtitle = "La línea punteada es el umbral que maximiza F1",
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
