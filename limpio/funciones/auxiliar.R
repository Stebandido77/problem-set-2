# =============================================================================
# auxiliar.R — Variante con objetivo auxiliar de ingreso.
# Un regresor LightGBM aprende z = log(1 + Ingpcug / Lp) con las mismas
# variables de la base. Pobre = 1 equivale a z < log(2), así que la predicción
# de z resume en una sola columna la distancia esperada a la línea. Ingpcug solo
# existe en train: la columna se construye siempre fuera de fold, de forma
# anidada, para que ninguna fila use un regresor que vio su propio ingreso.
# =============================================================================

PARAMETROS_REGRESOR <- list(
  objective = "regression",
  metric = "rmse",
  num_threads = NUM_HILOS,
  verbose = -1,
  learning_rate = 0.05,
  num_leaves = 63,
  min_data_in_leaf = 50,
  feature_fraction = 0.8,
  bagging_fraction = 0.8,
  bagging_freq = 1,
  lambda_l2 = 1
)

# Objetivo auxiliar continuo.
# Argumentos: ingreso_pc (Ingpcug), lp (línea de pobreza del hogar).
# Devuelve: vector z = log(1 + ingreso_pc / lp).
objetivo_auxiliar <- function(ingreso_pc, lp) {
  log1p(ingreso_pc / lp)
}

# Entrena el regresor auxiliar y predice sobre filas nuevas.
# Argumentos: x, z (entrenamiento), x_nueva, rondas.
# Devuelve: vector de predicciones de z para x_nueva.
ajustar_y_predecir_aux <- function(x, z, x_nueva, rondas) {
  modelo <- entrenar_lgb(x, z, list(), rondas, base = PARAMETROS_REGRESOR)
  predict(modelo, x_nueva)
}

# Construye la columna auxiliar anidada para una partición.
# Para cada fold externo k:
#   - filas de los folds j != k: predicción de un regresor entrenado en los
#     folds distintos de k y de j (OOF interna, es la columna con la que se
#     entrena el clasificador del fold k);
#   - filas del fold k: predicción de un regresor entrenado en todos los folds
#     distintos de k (es la columna con la que se valida).
# Ningún regresor que produce el valor de una fila vio el ingreso de esa fila,
# y el clasificador del fold k nunca usa información del fold k.
# Argumentos:
#   x: matriz del 80 %; z: objetivo auxiliar; fold: 1..5; entrena: lógico,
#   filas que pueden entrenar (escenario); rondas: rondas del regresor.
# Devuelve: lista con `columnas` (5 vectores, uno por fold externo) y
# `errores` (data.table con RMSE interno y externo por fold, para el test de
# fuga).
aux_anidada <- function(x, z, fold, entrena, rondas) {
  columnas <- vector("list", N_FOLDS)
  errores <- vector("list", N_FOLDS)
  for (k in seq_len(N_FOLDS)) {
    columna <- rep(NA_real_, length(z))
    for (j in setdiff(seq_len(N_FOLDS), k)) {
      tr <- !fold %in% c(k, j) & entrena
      columna[fold == j] <- ajustar_y_predecir_aux(
        x[tr, ], z[tr], x[fold == j, ], rondas
      )
    }
    tr <- fold != k & entrena
    columna[fold == k] <- ajustar_y_predecir_aux(
      x[tr, ], z[tr], x[fold == k, ], rondas
    )
    rmse <- function(filas) sqrt(mean((columna[filas] - z[filas])^2))
    errores[[k]] <- data.table(
      fold = k,
      rmse_interno = rmse(fold != k & entrena),
      rmse_externo = rmse(fold == k & entrena)
    )
    columnas[[k]] <- columna
  }
  list(columnas = columnas, errores = rbindlist(errores))
}

# Columna auxiliar para el ajuste final sobre el 80 %: OOF de 5 folds en el 80 %
# (columna de entrenamiento del clasificador final) y regresor ajustado en todo
# el 80 % para el holdout y el test.
# Argumentos: x, z, fold (del 80 %), entrena, x_nuevas (lista de matrices a
# predecir: holdout, test), rondas.
# Devuelve: lista con `entrenamiento` y una entrada por cada matriz nueva.
aux_final <- function(x, z, fold, entrena, x_nuevas, rondas) {
  entrenamiento <- rep(NA_real_, length(z))
  for (k in seq_len(N_FOLDS)) {
    tr <- fold != k & entrena
    entrenamiento[fold == k] <- ajustar_y_predecir_aux(
      x[tr, ], z[tr], x[fold == k, ], rondas
    )
  }
  modelo <- entrenar_lgb(x[entrena, ], z[entrena], list(), rondas,
                         base = PARAMETROS_REGRESOR)
  c(list(entrenamiento = entrenamiento),
    lapply(x_nuevas, function(m) predict(modelo, m)))
}
