# ==============================================================================
# 04_modelos_arboles.R
# Stage 2: CART, Random Forest y Boosting sobre la receta común.
# ==============================================================================

source(here::here("scripts", "00_configuracion.R"))
registrar_nucleos()

bases  <- cargar_bases_modelado()
folds  <- crear_folds(bases$train)
receta <- crear_receta(bases$train, balanceo = "ninguno")

# ---- CART --------------------------------------------------------------------
grilla_cart <- grid_regular(
  cost_complexity(range = c(-5, -1)),
  tree_depth(range = c(3, 15)),
  levels = 6
)

resultado_cart <- correr_algoritmo(
  algoritmo = "CART",
  modelo = decision_tree(cost_complexity = tune(), tree_depth = tune()) |>
    set_engine("rpart") |>
    set_mode("classification"),
  receta = receta, folds = folds, grilla = grilla_cart,
  datos_train = bases$train, datos_test = bases$test,
  especificacion = "CART | receta base | grid 6x6 | umbral óptimo F1"
)

# ---- Random Forest -----------------------------------------------------------
# Búsqueda aleatoria: más eficiente que la grilla con 2+ hiperparámetros
set.seed(SEMILLA)
grilla_rf <- grid_space_filling(
  mtry(range = c(3, 20)),
  min_n(range = c(5, 50)),
  size = 15
)

resultado_rf <- correr_algoritmo(
  algoritmo = "Random Forest",
  modelo = rand_forest(mtry = tune(), min_n = tune(), trees = 500) |>
    set_engine("ranger", importance = "impurity", seed = SEMILLA) |>
    set_mode("classification"),
  receta = receta, folds = folds, grilla = grilla_rf,
  datos_train = bases$train, datos_test = bases$test,
  especificacion = "Random Forest | receta base | 500 árboles | búsqueda space-filling 15"
)

# ---- Boosting ----------------------------------------------------------------
set.seed(SEMILLA)
grilla_boosting <- grid_space_filling(
  trees(range = c(200, 1000)),
  tree_depth(range = c(3, 8)),
  learn_rate(range = c(-3, -1)),
  size = 20
)

resultado_boosting <- correr_algoritmo(
  algoritmo = "Boosting",
  modelo = boost_tree(trees = tune(), tree_depth = tune(), learn_rate = tune()) |>
    set_engine("xgboost") |>
    set_mode("classification"),
  receta = receta, folds = folds, grilla = grilla_boosting,
  datos_train = bases$train, datos_test = bases$test,
  especificacion = "Boosting (XGBoost) | receta base | búsqueda space-filling 20"
)
