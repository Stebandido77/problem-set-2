# ==============================================================================
# 06_balanceo_clases.R
# Stage 3: ¿el manejo del desbalance cambia el resultado? Re-estima el mejor
# algoritmo con SMOTE y con submuestreo, manteniendo todo lo demás igual.
# Cambien ALGORITMO_BASE y el modelo según lo que haya ganado en Kaggle.
# ==============================================================================

source(here::here("scripts", "00_configuracion.R"))
registrar_nucleos()

bases <- cargar_bases_modelado()
folds <- crear_folds(bases$train)

ALGORITMO_BASE <- "Boosting"  # <- actualizar con el ganador

modelo_base <- boost_tree(trees = tune(), tree_depth = tune(), learn_rate = tune()) |>
  set_engine("xgboost") |>
  set_mode("classification")

set.seed(SEMILLA)
grilla_base <- grid_space_filling(
  trees(range = c(200, 1000)),
  tree_depth(range = c(3, 8)),
  learn_rate(range = c(-3, -1)),
  size = 20
)

resultados_balanceo <- map(c("smote", "submuestreo"), function(estrategia) {
  correr_algoritmo(
    algoritmo = ALGORITMO_BASE,
    modelo = modelo_base,
    receta = crear_receta(bases$train, balanceo = estrategia),
    folds = folds, grilla = grilla_base,
    datos_train = bases$train, datos_test = bases$test,
    especificacion = paste(ALGORITMO_BASE, "| balanceo:", estrategia)
  )
})
