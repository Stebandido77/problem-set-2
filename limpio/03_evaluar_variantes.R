# =============================================================================
# 03_evaluar_variantes.R — Mismo protocolo para las dos variantes
# (parsimonioso y auxiliar) en los dos escenarios (con y sin Bogotá):
#   1. Búsqueda aleatoria de N_CONFIGURACIONES en la partición 1, con parada
#      temprana; se elige por F1 OOF sin Bogotá.
#   2. Confirmación con rondas fijas en las particiones 1, 2 y 3.
#   3. Ajuste final en el 80 % y evaluación única en el holdout, completo y
#      sin Bogotá, con el umbral promedio de la CV sin Bogotá.
# Regla de decisión (escrita antes de correr): gana la combinación
# variante × escenario con mayor F1 medio de la CV sin Bogotá en las tres
# particiones. El holdout se reporta, no se usa para elegir.
# =============================================================================

source(here::here("limpio", "00_configuracion.R"))
datos <- cargar_datos_modelo()
registrar("03 | Evaluación de variantes")

FACTOR_RONDAS_FINAL <- 1.1 # el ajuste final ve 5/4 de los datos de cada fold

# Columna auxiliar anidada de una partición y escenario (NULL si no aplica).
columnas_auxiliares <- function(variante, p, escenario) {
  if (variante == "parsimonioso") return(NULL)
  readRDS(file.path(RUTA_SALIDAS, sprintf("aux_p%d_%s.rds", p, escenario)))$columnas
}

# Métricas OOF sin Bogotá y completas, cada una con su umbral óptimo.
# Argumentos: oof, y, bogota (lógico).
# Devuelve: data.table de una fila.
evaluar_oof <- function(oof, y, bogota) {
  sin <- buscar_umbral_f1(y[!bogota], oof[!bogota])
  com <- buscar_umbral_f1(y, oof)
  data.table(
    f1_sin_bogota = sin$f1, precision_sin_bogota = sin$precision,
    recall_sin_bogota = sin$recall, umbral_sin_bogota = sin$umbral,
    f1_completo = com$f1, umbral_completo = com$umbral
  )
}

# --- 1. Búsqueda en la partición 1 -------------------------------------------
configuraciones <- muestrear_configuraciones(N_CONFIGURACIONES)
archivo_busqueda <- file.path(RUTA_SALIDAS, "busqueda.csv")
busqueda <- if (file.exists(archivo_busqueda)) fread(archivo_busqueda) else data.table()

for (var_actual in VARIANTES) {
  for (esc_actual in ESCENARIOS) {
    aux <- columnas_auxiliares(var_actual, 1, esc_actual)
    for (cfg in configuraciones) {
      hecho <- nrow(busqueda) > 0 && nrow(busqueda[
        variante == var_actual & escenario == esc_actual &
          configuracion == cfg$configuracion]) > 0
      if (hecho) next
      inicio <- Sys.time()
      r <- oof_clasificador(datos$x, datos$y, datos$folds[[1]],
                            datos$entrena[[esc_actual]], cfg, MAX_RONDAS,
                            parada_temprana = TRUE, auxiliar = aux)
      fila <- cbind(
        data.table(variante = var_actual, escenario = esc_actual),
        as.data.table(cfg),
        rondas = round(mean(r$mejores_rondas)),
        evaluar_oof(r$oof, datos$y, datos$bogota),
        minutos = as.numeric(difftime(Sys.time(), inicio, units = "mins"))
      )
      busqueda <- rbind(busqueda, fila, fill = TRUE)
      fwrite(busqueda, archivo_busqueda)
      registrar(sprintf("03 | búsqueda %s %s cfg %d: F1 sin Bogotá %.4f (%d rondas, %.1f min)",
                        var_actual, esc_actual, cfg$configuracion,
                        fila$f1_sin_bogota, fila$rondas, fila$minutos))
    }
  }
}

elegidas <- busqueda[order(-f1_sin_bogota), .SD[1], by = .(variante, escenario)]
fwrite(elegidas, file.path(RUTA_SALIDAS, "configuraciones_elegidas.csv"))

parametros_de <- function(fila) {
  as.list(fila[, .(learning_rate, num_leaves, min_data_in_leaf,
                   feature_fraction, bagging_fraction, lambda_l2)])
}

# --- 2. Confirmación con rondas fijas en las tres particiones ---------------
archivo_cv <- file.path(RUTA_SALIDAS, "cv_confirmacion.csv")
cv <- if (file.exists(archivo_cv)) fread(archivo_cv) else data.table()

for (i in seq_len(nrow(elegidas))) {
  fila <- elegidas[i]
  for (p in seq_len(N_PARTICIONES)) {
    archivo_oof <- file.path(RUTA_SALIDAS, sprintf("oof_%s_%s_p%d.rds",
                                                   fila$variante, fila$escenario, p))
    if (file.exists(archivo_oof)) next
    r <- oof_clasificador(datos$x, datos$y, datos$folds[[p]],
                          datos$entrena[[fila$escenario]], parametros_de(fila),
                          fila$rondas,
                          auxiliar = columnas_auxiliares(fila$variante, p, fila$escenario))
    saveRDS(r$oof, archivo_oof)
    cv <- rbind(cv, cbind(fila[, .(variante, escenario)], particion = p,
                          evaluar_oof(r$oof, datos$y, datos$bogota)))
    fwrite(cv, archivo_cv)
    registrar(sprintf("03 | CV %s %s p%d: F1 sin Bogotá %.4f, completo %.4f",
                      fila$variante, fila$escenario, p,
                      tail(cv$f1_sin_bogota, 1), tail(cv$f1_completo, 1)))
  }
}

# --- 3. Ajuste final en el 80 % y holdout -----------------------------------
archivo_holdout <- file.path(RUTA_SALIDAS, "holdout.csv")
holdout <- if (file.exists(archivo_holdout)) fread(archivo_holdout) else data.table()

for (i in seq_len(nrow(elegidas))) {
  fila <- elegidas[i]
  hecho <- nrow(holdout) > 0 && nrow(holdout[
    variante == fila$variante & escenario == fila$escenario]) > 0
  if (hecho) next
  entrena <- datos$entrena[[fila$escenario]]
  x_tr <- datos$x
  x_ho <- datos$x_holdout
  x_te <- datos$x_test
  if (fila$variante == "auxiliar") {
    aux <- readRDS(file.path(RUTA_SALIDAS, sprintf("aux_final_%s.rds", fila$escenario)))
    x_tr <- cbind(x_tr, ingreso_aux = aux$entrenamiento)
    x_ho <- cbind(x_ho, ingreso_aux = aux$holdout)
    x_te <- cbind(x_te, ingreso_aux = aux$test)
  }
  modelo <- entrenar_lgb(x_tr[entrena, ], datos$y[entrena], parametros_de(fila),
                         round(fila$rondas * FACTOR_RONDAS_FINAL))
  umbral <- mean(cv[variante == fila$variante & escenario == fila$escenario,
                    umbral_sin_bogota])
  p_ho <- predict(modelo, x_ho)
  p_te <- predict(modelo, x_te)
  saveRDS(list(holdout = p_ho, test = p_te, umbral = umbral),
          file.path(RUTA_SALIDAS, sprintf("final_%s_%s.rds", fila$variante, fila$escenario)))
  completo <- metricas_en_umbral(datos$y_holdout, p_ho, umbral)
  sin <- metricas_en_umbral(datos$y_holdout[!datos$bogota_holdout],
                            p_ho[!datos$bogota_holdout], umbral)
  holdout <- rbind(holdout, data.table(
    variante = fila$variante, escenario = fila$escenario, umbral = umbral,
    f1_completo = completo$f1, precision_completo = completo$precision,
    recall_completo = completo$recall,
    f1_sin_bogota = sin$f1, precision_sin_bogota = sin$precision,
    recall_sin_bogota = sin$recall,
    tasa_predicha_test = mean(p_te >= umbral)
  ))
  fwrite(holdout, archivo_holdout)
  registrar(sprintf("03 | holdout %s %s: F1 completo %.4f, sin Bogotá %.4f",
                    fila$variante, fila$escenario, completo$f1, sin$f1))
}
