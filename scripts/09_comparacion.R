# ==============================================================================
# 09_comparacion.R
# Deck compare: mejor candidato por algoritmo entre los niveles 3, 4 y 4+.
# El nivel 4+ agrega variables laborales y de hacinamiento. Combina métricas del
# holdout con el F1 público anotado en la bitácora (f1_kaggle).
# ==============================================================================

source(here::here("scripts", "00_configuracion.R"))

leaderboard <- read_csv(file.path(RUTA_RESULTADOS, "leaderboard_interno.csv"),
                        show_col_types = FALSE, col_types = cols(nivel = col_character()))
bitacora <- read_csv(file.path(RUTA_SUBMISSIONS, "bitacora_kaggle.csv"),
                     show_col_types = FALSE, col_types = cols(nivel = col_character()))

# Niveles 3 y 4 comparten variables; 4+ evalúa el bloque complementario.
tabla_comparacion <- leaderboard |>
  filter(nivel %in% c("3", "4", "4+")) |>
  left_join(select(bitacora, candidato, f1_kaggle), by = "candidato") |>
  group_by(algoritmo) |>
  slice_max(f1_holdout, n = 1, with_ties = FALSE) |>
  ungroup() |>
  arrange(desc(f1_holdout)) |>
  select(nivel, algoritmo, candidato, umbral, f1_cv, f1_cv_sd, f1_holdout, precision_holdout,
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
    title = "Mejor desempeño por algoritmo y conjunto de variables",
    subtitle = "Holdout del 20%; el nivel 4+ agrega variables laborales y de hacinamiento",
    x = "Valor de la métrica", y = NULL, color = "Métrica"
  )
guardar_figura(grafico_comparacion, "comparacion_algoritmos")

# Conteo de submissions: el enunciado exige >= 15 con >= 5 algoritmos
subidas <- bitacora |> filter(!is.na(f1_kaggle))
message(
  "Subidas a Kaggle: ", nrow(subidas), " / 15 | Algoritmos: ",
  n_distinct(subidas$algoritmo), " / 5"
)
