# ==============================================================================
# 07_comparacion.R
# Stage 2 / deck compare: tabla y gráfico de la mejor corrida de cada algoritmo.
# Combina las métricas de CV con el F1 público que se anota en la bitácora.
# ==============================================================================

source(here::here("scripts", "00_configuracion.R"))

metricas <- read_csv(file.path(RUTA_RESULTADOS, "metricas_cv.csv"), show_col_types = FALSE)
bitacora <- read_csv(file.path(RUTA_SUBMISSIONS, "bitacora_kaggle.csv"), show_col_types = FALSE)

# Mejor corrida por algoritmo según F1 de validación cruzada
mejor_por_algoritmo <- metricas |>
  group_by(algoritmo) |>
  slice_max(f1, n = 1, with_ties = FALSE) |>
  ungroup() |>
  arrange(desc(f1))

tabla_comparacion <- mejor_por_algoritmo |>
  select(algoritmo, umbral, f1, precision, recall, filtracion, subcobertura) |>
  left_join(
    bitacora |>
      group_by(algoritmo) |>
      summarise(f1_publico = max(f1_publico, na.rm = TRUE), .groups = "drop") |>
      mutate(f1_publico = na_if(f1_publico, -Inf)),
    by = "algoritmo"
  )
guardar_tabla(tabla_comparacion, "comparacion_algoritmos")

grafico_comparacion <- tabla_comparacion |>
  select(algoritmo, Precisión = precision, Recall = recall, F1 = f1) |>
  pivot_longer(-algoritmo, names_to = "metrica", values_to = "valor") |>
  mutate(algoritmo = fct_reorder(algoritmo, valor, .fun = max)) |>
  ggplot(aes(valor, algoritmo, color = metrica)) +
  geom_point(size = 3) +
  scale_x_continuous(limits = c(0, 1)) +
  labs(
    title = "Desempeño fuera de muestra por algoritmo",
    subtitle = "Validación cruzada de 5 folds, umbral que maximiza F1",
    x = "Valor de la métrica", y = NULL, color = "Métrica"
  )
guardar_figura(grafico_comparacion, "comparacion_algoritmos")

# Conteo de submissions: el enunciado exige >= 15 con >= 5 algoritmos
resumen_submissions <- bitacora |> count(algoritmo, name = "submissions")
message(
  "Submissions: ", nrow(bitacora), " / 15 | Algoritmos: ",
  n_distinct(bitacora$algoritmo), " / 5"
)
