# =============================================================================
# 07_exportar_etapa18.R — Exporta el modelo limpio ganador (auxiliar con
# Bogotá) al formato del pipeline de python/ para la etapa 18. Escribe en
# limpio/salidas/etapa18/:
#   variables_limpio.parquet      id + las 74 variables de la base (train + test)
#   aux_limpio_contextos.parquet  columna auxiliar l_ingreso_aux por partición
#                                 (0 = original de R, 1-3 = repetidas de la etapa
#                                 16) y contexto (cv_k, holdout, test), anidada:
#                                 las filas de entrenamiento de un contexto tienen
#                                 OOF interna y las de evaluación la predicción de
#                                 un regresor que no las vio
#   predicciones_n18_limpio.parquet  id, tipo (oof/holdout/test), fold,
#                                 probabilidad: OOF en la partición original de R,
#                                 holdout (ajuste en el 80 %) y test (100 %)
#   oof_limpio_repetida.parquet   OOF guardadas de 03 en las particiones 1-3
# Hiperparámetros y rondas: los de configuraciones_elegidas.csv (03); el
# reajuste en el 80 % y en el 100 % usa las rondas por 1,1, como en 03 y 06.
# =============================================================================

source(here::here("limpio", "00_configuracion.R"))
registrar("07 | Exportación para la etapa 18")

RUTA_ETAPA18 <- file.path(RUTA_SALIDAS, "etapa18")
dir.create(RUTA_ETAPA18, showWarnings = FALSE)
FACTOR_RONDAS_FINAL <- 1.1
ESCENARIO <- "con_bogota"

fila <- fread(file.path(RUTA_SALIDAS, "configuraciones_elegidas.csv"))[
  variante == "auxiliar" & escenario == ESCENARIO]
parametros <- as.list(fila[, .(learning_rate, num_leaves, min_data_in_leaf,
                               feature_fraction, bagging_fraction, lambda_l2)])
rondas_aux <- readRDS(file.path(RUTA_SALIDAS, "rondas_regresor.rds"))

base_train <- readRDS(file.path(RUTA_SALIDAS, "base_train.rds"))
base_test <- readRDS(file.path(RUTA_SALIDAS, "base_test.rds"))
variables <- variables_explicativas(base_train)
x <- preparar_matriz(base_train, variables)
x_test <- preparar_matriz(base_test, variables)
y <- base_train$Pobre
z <- objetivo_auxiliar(base_train$Ingpcug, base_train$lp)

# Grupos de la partición original de R: fold 1-5 en el 80 %, 6 en el holdout
original <- as.data.table(arrow::read_parquet(here::here("stores", "procesados", "particion_nivel7.parquet")))
original <- original[base == "train"][match(base_train$id, id)]
stopifnot(identical(original$id, base_train$id))
grupo <- ifelse(original$holdout, 6L, as.integer(original$fold_id))
en_80 <- grupo != 6L
stopifnot(!anyNA(grupo), setequal(base_train$id[!en_80], fread(file.path(RUTA_IDS, "holdout_ids.csv"))$id))

# --- Variables del modelo limpio -------------------------------------------------
arrow::write_parquet(rbind(base_train[, c("id", variables), with = FALSE],
                           base_test[, c("id", variables), with = FALSE]),
                     file.path(RUTA_ETAPA18, "variables_limpio.parquet"))

# --- Columna auxiliar por partición y contexto -----------------------------------
# Convierte la salida de aux_anidada() (5 vectores sobre el 80 %) en filas largas.
a_contextos <- function(columnas, ids, particion) {
  rbindlist(lapply(seq_along(columnas), function(k) {
    data.table(particion = particion, contexto = paste0("cv_", k), id = ids,
               l_ingreso_aux = columnas[[k]])
  }))
}

# Columna de un contexto de entrenamiento final: OOF por grupo sobre las filas
# de entrenamiento y regresor con todas ellas para las de evaluación.
# Argumentos: filas_ent (lógico sobre x), grupo_ent (grupo de esas filas),
# x_eval, contexto, ids_eval.
contexto_final <- function(filas_ent, grupo_ent, x_eval, contexto, ids_eval) {
  oof <- rep(NA_real_, sum(filas_ent))
  x_ent <- x[filas_ent, ]
  z_ent <- z[filas_ent]
  for (g in sort(unique(grupo_ent))) {
    oof[grupo_ent == g] <- ajustar_y_predecir_aux(
      x_ent[grupo_ent != g, ], z_ent[grupo_ent != g], x_ent[grupo_ent == g, ], rondas_aux)
  }
  eval <- ajustar_y_predecir_aux(x_ent, z_ent, x_eval, rondas_aux)
  list(
    tabla = rbind(
      data.table(particion = 0L, contexto = contexto, id = base_train$id[filas_ent], l_ingreso_aux = oof),
      data.table(particion = 0L, contexto = contexto, id = ids_eval, l_ingreso_aux = eval)
    ),
    entrenamiento = oof, evaluacion = eval
  )
}

archivo_aux <- file.path(RUTA_ETAPA18, "aux_limpio_contextos.parquet")
archivo_original <- file.path(RUTA_SALIDAS, "aux_original_con_bogota.rds")
if (!file.exists(archivo_original)) {
  saveRDS(aux_anidada(x[en_80, ], z[en_80], grupo[en_80], rep(TRUE, sum(en_80)), rondas_aux),
          archivo_original)
}
aux_original <- readRDS(archivo_original)
ids_80 <- base_train$id[en_80]
ctx_holdout <- contexto_final(en_80, grupo[en_80], x[!en_80, ], "holdout", base_train$id[!en_80])
ctx_test <- contexto_final(rep(TRUE, nrow(x)), grupo, x_test, "test", base_test$id)
registrar("07 | Columna auxiliar de la partición original lista")

datos <- cargar_datos_modelo()
repetidas <- rbindlist(lapply(seq_len(N_PARTICIONES), function(p) {
  aux <- readRDS(file.path(RUTA_SALIDAS, sprintf("aux_p%d_%s.rds", p, ESCENARIO)))
  a_contextos(aux$columnas, datos$ids_80, p)
}))
aux_tabla <- rbind(a_contextos(aux_original$columnas, ids_80, 0L),
                   ctx_holdout$tabla, ctx_test$tabla, repetidas)
stopifnot(!anyNA(aux_tabla$l_ingreso_aux),
          !anyDuplicated(aux_tabla[, .(particion, contexto, id)]))
arrow::write_parquet(aux_tabla, archivo_aux)

# --- Predicciones del clasificador limpio ------------------------------------------
oof <- oof_clasificador(x[en_80, ], y[en_80], grupo[en_80], rep(TRUE, sum(en_80)),
                        parametros, fila$rondas, auxiliar = aux_original$columnas)$oof
rondas_final <- round(fila$rondas * FACTOR_RONDAS_FINAL)
modelo_holdout <- entrenar_lgb(cbind(x[en_80, ], ingreso_aux = ctx_holdout$entrenamiento),
                               y[en_80], parametros, rondas_final)
p_holdout <- predict(modelo_holdout, cbind(x[!en_80, ], ingreso_aux = ctx_holdout$evaluacion))
modelo_test <- entrenar_lgb(cbind(x, ingreso_aux = ctx_test$entrenamiento), y, parametros, rondas_final)
p_test <- predict(modelo_test, cbind(x_test, ingreso_aux = ctx_test$evaluacion))

predicciones <- rbind(
  data.table(id = ids_80, tipo = "oof", fold = grupo[en_80], probabilidad = oof),
  data.table(id = base_train$id[!en_80], tipo = "holdout", fold = NA_integer_, probabilidad = p_holdout),
  data.table(id = base_test$id, tipo = "test", fold = NA_integer_, probabilidad = p_test)
)
arrow::write_parquet(predicciones, file.path(RUTA_ETAPA18, "predicciones_n18_limpio.parquet"))
umbral_original <- buscar_umbral_f1(y[en_80], oof)
registrar(sprintf("07 | OOF original: F1 %.4f (umbral %.4f); holdout F1 %.4f con ese umbral",
                  umbral_original$f1, umbral_original$umbral,
                  metricas_en_umbral(y[!en_80], p_holdout, umbral_original$umbral)$f1))

# --- OOF repetidas de 03 -------------------------------------------------------------
oof_repetida <- rbindlist(lapply(seq_len(N_PARTICIONES), function(p) {
  data.table(particion = p, id = datos$ids_80, fold = datos$folds[[p]],
             probabilidad = readRDS(file.path(RUTA_SALIDAS, sprintf(
               "oof_auxiliar_%s_p%d.rds", ESCENARIO, p))))
}))
arrow::write_parquet(oof_repetida, file.path(RUTA_ETAPA18, "oof_limpio_repetida.parquet"))
registrar("07 | Exportación lista")
