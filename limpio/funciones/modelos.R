# =============================================================================
# modelos.R — Clasificador LightGBM: parámetros, búsqueda aleatoria y
# predicciones fuera de fold (OOF).
# =============================================================================

# Parámetros fijos del clasificador; la búsqueda solo varía los de
# muestrear_configuraciones().
PARAMETROS_BASE_CLASIFICADOR <- list(
  objective = "binary",
  metric = "binary_logloss",
  num_threads = NUM_HILOS,
  verbose = -1,
  bagging_freq = 1
)

# Muestrea configuraciones de hiperparámetros al azar dentro de rangos fijos.
# Rangos: learning_rate en [0,03; 0,10] (log), num_leaves en {15..127},
# min_data_in_leaf en {20..300}, feature_fraction en [0,5; 1],
# bagging_fraction en [0,6; 1], lambda_l2 en [0; 10].
# Argumentos: n, número de configuraciones; semilla.
# Devuelve: lista de listas de parámetros (con su número de configuración).
muestrear_configuraciones <- function(n, semilla = SEMILLA) {
  set.seed(semilla)
  lapply(seq_len(n), function(i) {
    list(
      configuracion = i,
      learning_rate = exp(runif(1, log(0.03), log(0.10))),
      num_leaves = sample(15:127, 1),
      min_data_in_leaf = sample(20:300, 1),
      feature_fraction = runif(1, 0.5, 1),
      bagging_fraction = runif(1, 0.6, 1),
      lambda_l2 = runif(1, 0, 10)
    )
  })
}

# Escenarios de entrenamiento: con todo el train o sin Bogotá (ausente en test).
ESCENARIOS <- c("con_bogota", "sin_bogota")
VARIANTES <- c("parsimonioso", "auxiliar")

# Carga las bases y deja listas las piezas que usan los scripts 02 a 05.
# Argumentos: ninguno (lee de RUTA_SALIDAS y RUTA_IDS).
# Devuelve: lista con matrices y vectores del 80 % (`x`, `y`, `z`, `bogota`),
# del holdout (`x_holdout`, `y_holdout`, `bogota_holdout`), del test
# (`x_test`, `ids_test`), los 3 vectores de folds del 80 % y las filas que
# entrenan en cada escenario (`entrena`).
cargar_datos_modelo <- function() {
  base_train <- readRDS(file.path(RUTA_SALIDAS, "base_train.rds"))
  base_test <- readRDS(file.path(RUTA_SALIDAS, "base_test.rds"))
  variables <- variables_explicativas(base_train)
  particiones <- cargar_particiones(base_train$id)
  en_80 <- !particiones$holdout
  b80 <- base_train[en_80]
  bho <- base_train[particiones$holdout]
  bogota <- b80$es_bogota == 1L
  list(
    variables = variables,
    x = preparar_matriz(b80, variables),
    y = b80$Pobre,
    z = objetivo_auxiliar(b80$Ingpcug, b80$lp),
    bogota = bogota,
    folds = lapply(particiones$folds, function(f) f[en_80]),
    entrena = list(con_bogota = rep(TRUE, nrow(b80)), sin_bogota = !bogota),
    x_holdout = preparar_matriz(bho, variables),
    y_holdout = bho$Pobre,
    bogota_holdout = bho$es_bogota == 1L,
    x_test = preparar_matriz(base_test, variables),
    ids_test = base_test$id,
    ids_80 = b80$id,
    ids_holdout = bho$id
  )
}

# Convierte la base en matriz numérica para LightGBM.
# Argumentos: base, data.table; variables, columnas a usar.
# Devuelve: matriz numérica con nombres de columna.
preparar_matriz <- function(base, variables) {
  data.matrix(base[, ..variables])
}

# Entrena un booster de LightGBM.
# Argumentos:
#   x, y: matriz y objetivo de entrenamiento.
#   parametros: lista de hiperparámetros (se agregan los base).
#   rondas: número máximo de rondas.
#   x_valida, y_valida: opcionales; si se dan, hay parada temprana.
#   base: parámetros fijos (clasificador por defecto).
# Devuelve: booster entrenado (con best_iter si hubo parada temprana).
entrenar_lgb <- function(x, y, parametros, rondas, x_valida = NULL,
                         y_valida = NULL,
                         base = PARAMETROS_BASE_CLASIFICADOR) {
  parametros$configuracion <- NULL
  parametros <- modifyList(base, parametros)
  parametros$seed <- SEMILLA
  categoricas <- intersect(VARIABLES_CATEGORICAS, colnames(x))
  datos <- lgb.Dataset(x, label = y, categorical_feature = categoricas,
                       free_raw_data = FALSE)
  if (is.null(x_valida)) {
    return(lgb.train(parametros, datos, nrounds = rondas, verbose = -1))
  }
  valida <- lgb.Dataset.create.valid(datos, x_valida, label = y_valida)
  lgb.train(parametros, datos, nrounds = rondas,
            valids = list(valida = valida),
            early_stopping_rounds = PARADA_TEMPRANA, verbose = -1)
}

# Predicciones fuera de fold del clasificador sobre el 80 %.
# Argumentos:
#   x: matriz de las filas del 80 % (en el orden de `fold`).
#   y: objetivo de esas filas.
#   fold: vector 1..5 de esas filas.
#   entrena: lógico, filas que pueden entrar al entrenamiento (escenario con o
#     sin Bogotá); la validación usa siempre todas las filas del fold.
#   parametros, rondas: hiperparámetros y rondas.
#   parada_temprana: si TRUE, usa el fold de validación para parar (solo en la
#     búsqueda); si FALSE, entrena con `rondas` fijas.
#   auxiliar: NULL o lista de 5 vectores (uno por fold externo) con la columna
#     auxiliar anidada de aux_anidada().
# Devuelve: lista con `oof` (probabilidades) y `mejores_rondas` (por fold).
oof_clasificador <- function(x, y, fold, entrena, parametros, rondas,
                             parada_temprana = FALSE, auxiliar = NULL) {
  oof <- rep(NA_real_, length(y))
  mejores <- integer(N_FOLDS)
  for (k in seq_len(N_FOLDS)) {
    x_k <- if (is.null(auxiliar)) x else cbind(x, ingreso_aux = auxiliar[[k]])
    tr <- fold != k & entrena
    va <- fold == k
    modelo <- if (parada_temprana) {
      entrenar_lgb(x_k[tr, ], y[tr], parametros, rondas,
                   x_k[va, ], y[va])
    } else {
      entrenar_lgb(x_k[tr, ], y[tr], parametros, rondas)
    }
    mejores[k] <- if (parada_temprana) modelo$best_iter else rondas
    oof[va] <- predict(modelo, x_k[va, ])
  }
  list(oof = oof, mejores_rondas = mejores)
}
