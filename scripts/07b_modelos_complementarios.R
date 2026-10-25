# ============================================================================== 
# 07b_modelos_complementarios.R
# 3 candidatos de complejidad media/alta: tuning real de hiperparámetros
# ============================================================================== 

source(here::here("scripts", "00_configuracion.R"))

# Desactivar explícitamente paralelización pesada para evitar saturación de RAM
suppressPackageStartupMessages(library(future))
plan(sequential)

message("-> Cargando datos de entrenamiento y prueba...")
bases <- cargar_bases_modelado()

# Receta liviana: reduce dimensiones reduciendo multicolinealidad
receta_balanceada <- recipe(pobre ~ ., data = bases$train) |>
  update_role(id, new_role = "id") |>
  step_impute_median(all_numeric_predictors()) |>
  step_impute_mode(all_nominal_predictors()) |>
  step_novel(all_nominal_predictors()) |>
  step_dummy(all_nominal_predictors()) |>
  step_zv(all_predictors()) |>
  step_corr(all_numeric_predictors(), threshold = 0.80) |> # Conserva solo predictores muy limpios
  step_normalize(all_numeric_predictors())

folds <- crear_folds(bases$train, v = N_FOLDS)

# ------------------------------------------------------------------------------
# MODELO 1: Elastic Net (Súper rápido: ~1-2 mins)
# ------------------------------------------------------------------------------
message("\n[1/3] Entrenando Elastic Net...")
t1 <- Sys.time()
gc(verbose = FALSE)

grilla_en <- grid_regular(
  penalty(range = c(-3, -1)),
  mixture(range = c(0.1, 0.9)),
  levels = c(penalty = 5, mixture = 3) # 15 combinaciones livianas
)

res_elastic_net <- correr_algoritmo(
  algoritmo = "Elastic Net Balanceado",
  modelo = logistic_reg(penalty = tune(), mixture = tune()) |> set_engine("glmnet"),
  receta = receta_balanceada,
  folds = folds,
  grilla = grilla_en,
  datos_train = bases$train,
  datos_test = bases$test,
  especificacion = "Elastic Net | Receta Optimizada | Grid 5x3"
)
message("   -> Tiempo transcurrido: ", round(difftime(Sys.time(), t1, units = "mins"), 2), " minutos.")

# ------------------------------------------------------------------------------
# MODELO 2: Random Forest (Control de hilos: ~3 mins)
# ------------------------------------------------------------------------------
message("\n[2/3] Entrenando Random Forest...")
t2 <- Sys.time()
gc(verbose = FALSE)

receta_prep <- prep(receta_balanceada, training = bases$train)
n_preds <- ncol(bake(receta_prep, new_data = NULL)) - 2

set.seed(SEMILLA)
grilla_rf <- grid_space_filling(
  mtry(range = c(2, min(10, n_preds))),
  min_n(range = c(10, 30)),
  size = 6 # 6 evaluaciones de espacio lleno
)

res_rf <- correr_algoritmo(
  algoritmo = "Random Forest Balanceado",
  modelo = rand_forest(mtry = tune(), min_n = tune(), trees = 200) |> # 200 árboles es suficiente y vuela
    set_engine("ranger", importance = "impurity", seed = SEMILLA, num.threads = 2) |>
    set_mode("classification"),
  receta = receta_balanceada,
  folds = folds,
  grilla = grilla_rf,
  datos_train = bases$train,
  datos_test = bases$test,
  especificacion = "Random Forest | 200 arboles | Space-filling 6"
)
message("   -> Tiempo transcurrido: ", round(difftime(Sys.time(), t2, units = "mins"), 2), " minutos.")

# ------------------------------------------------------------------------------
# MODELO 3: XGBoost (Control de hilos: ~3 mins)
# ------------------------------------------------------------------------------
message("\n[3/3] Entrenando XGBoost...")
t3 <- Sys.time()
gc(verbose = FALSE)

set.seed(SEMILLA)
grilla_xgb <- grid_space_filling(
  trees(range = c(100, 300)),
  tree_depth(range = c(3, 5)),
  learn_rate(range = c(-2, -1)),
  size = 6
)

res_xgb <- correr_algoritmo(
  algoritmo = "XGBoost Balanceado",
  modelo = boost_tree(trees = tune(), tree_depth = tune(), learn_rate = tune()) |>
    set_engine("xgboost", nthread = 2) |>
    set_mode("classification"),
  receta = receta_balanceada,
  folds = folds,
  grilla = grilla_xgb,
  datos_train = bases$train,
  datos_test = bases$test,
  especificacion = "XGBoost | Receta Optimizada | Space-filling 6"
)
message("   -> Tiempo transcurrido: ", round(difftime(Sys.time(), t3, units = "mins"), 2), " minutos.")

# Limpieza final
gc(verbose = FALSE)
message("\n>>> ¡Proceso finalizado con éxito! Archivos generados en stores/submissions/")