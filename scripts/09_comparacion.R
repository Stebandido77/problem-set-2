# ==============================================================================
# 09_comparacion.R
# Deck compare: los algoritmos en cancha nivelada (misma base del nivel 3, misma
# receta). Combina las métricas del holdout interno con el F1 público que se
# anota a mano en la bitácora (f1_kaggle).
# ==============================================================================

source(here::here("scripts", "00_configuracion.R"))

leaderboard <- read_csv(file.path(RUTA_RESULTADOS, "leaderboard_interno.csv"),
                        show_col_types = FALSE, col_types = cols(nivel = col_character()))
bitacora <- read_csv(file.path(RUTA_SUBMISSIONS, "bitacora_kaggle.csv"),
                     show_col_types = FALSE, col_types = cols(nivel = col_character()))

# Cancha nivelada: niveles 3 y 4 comparten variables y preprocesamiento
tabla_comparacion <- leaderboard |>
  filter(nivel %in% c("3", "4")) |>
  left_join(select(bitacora, candidato, f1_kaggle), by = "candidato") |>
  group_by(algoritmo) |>
  slice_max(f1_holdout, n = 1, with_ties = FALSE) |>
  ungroup() |>
  arrange(desc(f1_holdout)) |>
  select(algoritmo, candidato, umbral, f1_cv, f1_cv_sd, f1_holdout, precision_holdout,
         recall_holdout, filtracion_pct, subcobertura_pct, f1_kaggle)
guardar_tabla(tabla_comparacion, "comparacion_algoritmos")

grafico_comparacion <- tabla_comparacion |>
  select(algoritmo, Precisión = precision_holdout, Recall = recall_holdout, F1 = f1_holdout) |>
  pivot_longer(-algoritmo, names_to = "metrica", values_to = "valor") |>
  mutate(
    algoritmo = fct_reorder(algoritmo, valor, .fun = max),
    metrica = factor(metrica, levels = c("F1", "Precisión", "Recall"))
  ) |>
  ggplot(aes(valor, algoritmo, color = metrica)) +
  geom_point(size = 3) +
  scale_color_manual(values = c(F1 = "#2a78d6", Precisión = "#eb6834", Recall = "#1baf7a")) +
  scale_x_continuous(limits = c(0, 1)) +
  labs(
    title = "Desempeño fuera de muestra por algoritmo, con las mismas variables",
    subtitle = "Holdout del 20%; umbral elegido con predicciones fuera de fold para maximizar F1",
    x = "Valor de la métrica", y = NULL, color = "Métrica"
  )
guardar_figura(grafico_comparacion, "comparacion_algoritmos")

# Conteo de submissions: el enunciado exige >= 15 con >= 5 algoritmos
subidas <- bitacora |> filter(!is.na(f1_kaggle))
message(
  "Subidas a Kaggle: ", nrow(subidas), " / 15 | Algoritmos: ",
  n_distinct(subidas$algoritmo), " / 5"
)
