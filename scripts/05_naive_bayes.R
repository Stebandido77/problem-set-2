# ==============================================================================
# 05_naive_bayes.R
# Stage 2: Naive Bayes sobre la receta común.
# ==============================================================================

source(here::here("scripts", "00_configuracion.R"))
registrar_nucleos()

bases  <- cargar_bases_modelado()
folds  <- crear_folds(bases$train)
receta <- crear_receta(bases$train, balanceo = "ninguno")

grilla_naive_bayes <- grid_regular(
  smoothness(range = c(0.5, 1.5)),
  Laplace(range = c(0, 3)),
  levels = 4
)

resultado_naive_bayes <- correr_algoritmo(
  algoritmo = "Naive Bayes",
  modelo = naive_Bayes(smoothness = tune(), Laplace = tune()) |>
    set_engine("naivebayes"),
  receta = receta, folds = folds, grilla = grilla_naive_bayes,
  datos_train = bases$train, datos_test = bases$test,
  especificacion = "Naive Bayes | receta base | grid 4x4 | umbral óptimo F1"
)
