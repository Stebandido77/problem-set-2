# =============================================================================
# 06_submission_final.R — Reajusta la combinación ganadora (según 05) sobre el
# 100 % del train (80 % + holdout) con los mismos hiperparámetros, rondas y
# umbral del ajuste final en el 80 %, y escribe la submission validada en
# limpio/resultados/submission_limpio.csv. Copia además las tablas livianas de
# resultados a limpio/resultados/ (lo que se versiona).
# =============================================================================

source(here::here("limpio", "00_configuracion.R"))
registrar("06 | Submission final sobre el 100 % del train")

RUTA_RESULTADOS <- here::here("limpio", if (MODO_PRUEBA) "resultados_prueba" else "resultados")
dir.create(RUTA_RESULTADOS, showWarnings = FALSE)
FACTOR_RONDAS_FINAL <- 1.1 # el mismo de 03

tabla <- fread(file.path(RUTA_SALIDAS, "tabla_comparacion.csv"))
ganador <- tabla[1]
fila <- fread(file.path(RUTA_SALIDAS, "configuraciones_elegidas.csv"))[
  variante == ganador$variante & escenario == ganador$escenario]
umbral <- readRDS(file.path(RUTA_SALIDAS, sprintf(
  "final_%s_%s.rds", ganador$variante, ganador$escenario)))$umbral
rondas <- round(fila$rondas * FACTOR_RONDAS_FINAL)
parametros <- as.list(fila[, .(learning_rate, num_leaves, min_data_in_leaf,
                               feature_fraction, bagging_fraction, lambda_l2)])

base_train <- readRDS(file.path(RUTA_SALIDAS, "base_train.rds"))
base_test <- readRDS(file.path(RUTA_SALIDAS, "base_test.rds"))
variables <- variables_explicativas(base_train)
x <- preparar_matriz(base_train, variables)
x_test <- preparar_matriz(base_test, variables)
y <- base_train$Pobre
entrena <- if (ganador$escenario == "sin_bogota") base_train$es_bogota == 0L else rep(TRUE, nrow(x))

if (ganador$variante == "auxiliar") {
  # Columna auxiliar del 100 %: OOF de 5 folds (folds de la partición 1 en el
  # 80 %; el holdout se reparte al azar entre los 5) y regresor en todo el
  # train para el test.
  particiones <- cargar_particiones(base_train$id)
  fold <- particiones$folds[[1]]
  set.seed(SEMILLA)
  fold[particiones$holdout] <- sample(rep_len(seq_len(N_FOLDS), sum(particiones$holdout)))
  z <- objetivo_auxiliar(base_train$Ingpcug, base_train$lp)
  rondas_aux <- readRDS(file.path(RUTA_SALIDAS, "rondas_regresor.rds"))
  aux <- aux_final(x, z, fold, entrena, list(test = x_test), rondas_aux)
  x <- cbind(x, ingreso_aux = aux$entrenamiento)
  x_test <- cbind(x_test, ingreso_aux = aux$test)
}

modelo <- entrenar_lgb(x[entrena, ], y[entrena], parametros, rondas)
p_test <- predict(modelo, x_test)
submission <- data.table(id = base_test$id, pobre = as.integer(p_test >= umbral))

# Validación contra el formato de Kaggle
ejemplo <- fread(file.path(RUTA_RAW, "sample_submission.csv"))
stopifnot(
  identical(names(submission), c("id", "pobre")),
  MODO_PRUEBA || nrow(submission) == nrow(ejemplo),
  MODO_PRUEBA || nrow(submission) == 66168,
  !anyNA(submission),
  all(submission$pobre %in% c(0L, 1L)),
  !anyDuplicated(submission$id),
  MODO_PRUEBA || setequal(submission$id, ejemplo$id)
)
fwrite(submission, file.path(RUTA_RESULTADOS, "submission_limpio.csv"))

# Concordancia con la predicción del ajuste en el 80 %
p_80 <- readRDS(file.path(RUTA_SALIDAS, sprintf(
  "final_%s_%s.rds", ganador$variante, ganador$escenario)))$test
registrar(sprintf(
  "06 | %s %s: %d rondas, umbral %.4f, tasa predicha test %.4f (ajuste 80 %%: %.4f), concordancia con 80 %% %.4f",
  ganador$variante, ganador$escenario, rondas, umbral, mean(submission$pobre),
  mean(p_80 >= umbral), mean(submission$pobre == as.integer(p_80 >= umbral))))

for (archivo in c("tabla_comparacion.csv", "diferencias_pareadas.csv",
                  "cv_confirmacion.csv", "holdout.csv", "busqueda.csv",
                  "configuraciones_elegidas.csv", "fuga_resumen.csv",
                  "fuga_brecha_rmse.csv", "faltantes_base.csv")) {
  file.copy(file.path(RUTA_SALIDAS, archivo), file.path(RUTA_RESULTADOS, archivo),
            overwrite = TRUE)
}
file.copy(file.path(RUTA_SALIDAS, "log.txt"),
          file.path(RUTA_RESULTADOS, "log_corrida.txt"), overwrite = TRUE)
