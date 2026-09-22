# ==============================================================================
# 03_modelos_lineales.R
# Stage 2: LPM, Logit y Elastic Net sobre la receta común.
# Cada corrida genera una submission y queda en la bitácora.
# ==============================================================================

source(here::here("scripts", "00_configuracion.R"))
registrar_nucleos()

bases  <- cargar_bases_modelado()
folds  <- crear_folds(bases$train)
receta <- crear_receta(bases$train, balanceo = "ninguno")

# ---- LPM ---------------------------------------------------------------------
resultado_lpm <- correr_lpm(
  receta, folds, bases$train, bases$test,
  especificacion = "LPM | receta base | umbral óptimo F1"
)

# ---- Logit -------------------------------------------------------------------
resultado_logit <- correr_algoritmo(
  algoritmo = "Logit",
  modelo = logistic_reg() |> set_engine("glm"),
  receta = receta, folds = folds, grilla = NULL,
  datos_train = bases$train, datos_test = bases$test,
  especificacion = "Logit | receta base | umbral óptimo F1"
)

# ---- Elastic Net -------------------------------------------------------------
# Búsqueda en grilla: penalty en escala log, mixture de ridge (0) a lasso (1)
grilla_elastic_net <- grid_regular(
  penalty(range = c(-5, 0)),
  mixture(range = c(0, 1)),
  levels = c(penalty = 20, mixture = 6)
)

resultado_elastic_net <- correr_algoritmo(
  algoritmo = "Elastic Net",
  modelo = logistic_reg(penalty = tune(), mixture = tune()) |> set_engine("glmnet"),
  receta = receta, folds = folds, grilla = grilla_elastic_net,
  datos_train = bases$train, datos_test = bases$test,
  especificacion = "Elastic Net | receta base | grid 20x6 | umbral óptimo F1"
)
