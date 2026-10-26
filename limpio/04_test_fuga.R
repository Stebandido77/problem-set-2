# =============================================================================
# 04_test_fuga.R — Pruebas de fuga de la variante auxiliar.
#   A. Ids: holdout y 80 % son disjuntos (también se verifica al cargar).
#   B. Brecha de error: el RMSE de la columna con la que se entrena (OOF
#      interna) no puede ser menor que el de la columna de validación. Si lo
#      fuera, el clasificador vería una columna más limpia en entrenamiento
#      que en validación.
#   C. Control negativo: con el ingreso permutado al azar, la variante auxiliar
#      no debe superar al mismo clasificador sin la columna (mismos
#      hiperparámetros y rondas), porque la columna ya no lleva información.
#   D. Control positivo: una columna construida CON fuga (regresor ajustado en
#      todo el 80 %, incluido el fold de validación) debe inflar el F1. Muestra
#      que el protocolo detecta una fuga cuando existe.
#   E. Deriva: distribución de la columna en entrenamiento, holdout y test
#      (estadístico de Kolmogorov-Smirnov).
# Las pruebas C y D usan la partición 1 y el escenario y la configuración
# elegidos para la variante auxiliar.
# =============================================================================

source(here::here("limpio", "00_configuracion.R"))
datos <- cargar_datos_modelo()
registrar("04 | Test de fuga")

TOLERANCIA_CONTROL_NEGATIVO <- 0.003
TOLERANCIA_BRECHA_RMSE <- 0.02 # relativa

RONDAS_AUX <- readRDS(file.path(RUTA_SALIDAS, "rondas_regresor.rds"))
elegidas <- fread(file.path(RUTA_SALIDAS, "configuraciones_elegidas.csv"))
cv <- fread(file.path(RUTA_SALIDAS, "cv_confirmacion.csv"))
resultados <- list()

# A. Ids disjuntos
resultados$ids <- data.table(
  prueba = "A. holdout y 80 % disjuntos",
  valor = length(intersect(datos$ids_holdout, datos$ids_80)),
  pasa = length(intersect(datos$ids_holdout, datos$ids_80)) == 0
)

# B. Brecha de RMSE interno vs externo en todas las particiones y escenarios
brechas <- rbindlist(lapply(seq_len(N_PARTICIONES), function(p) {
  rbindlist(lapply(ESCENARIOS, function(e) {
    aux <- readRDS(file.path(RUTA_SALIDAS, sprintf("aux_p%d_%s.rds", p, e)))
    cbind(particion = p, escenario = e, aux$errores)
  }))
}))
brechas[, brecha_relativa := (rmse_interno - rmse_externo) / rmse_externo]
fwrite(brechas, file.path(RUTA_SALIDAS, "fuga_brecha_rmse.csv"))
# El criterio usa el promedio de los 5 folds por partición y escenario: fold a
# fold la brecha tiene ruido de muestreo simétrico.
brecha_media <- brechas[, .(brecha = (mean(rmse_interno) - mean(rmse_externo)) /
                              mean(rmse_externo)), by = .(particion, escenario)]
resultados$brecha <- data.table(
  prueba = "B. RMSE interno >= externo (brecha relativa media mínima)",
  valor = min(brecha_media$brecha),
  pasa = min(brecha_media$brecha) > -TOLERANCIA_BRECHA_RMSE
)

# Configuración y escenario de la variante auxiliar
fila_aux <- elegidas[variante == "auxiliar"][order(-f1_sin_bogota)][1]
esc <- fila_aux$escenario
parametros <- as.list(fila_aux[, .(learning_rate, num_leaves, min_data_in_leaf,
                                   feature_fraction, bagging_fraction, lambda_l2)])
fold1 <- datos$folds[[1]]
entrena <- datos$entrena[[esc]]
f1_parsimonioso <- cv[variante == "parsimonioso" & escenario == esc &
                        particion == 1, f1_sin_bogota]
f1_auxiliar <- cv[variante == "auxiliar" & escenario == esc & particion == 1,
                  f1_sin_bogota]

f1_con_columna <- function(columnas) {
  r <- oof_clasificador(datos$x, datos$y, fold1, entrena, parametros,
                        fila_aux$rondas, auxiliar = columnas)
  buscar_umbral_f1(datos$y[!datos$bogota], r$oof[!datos$bogota])$f1
}

# C. Control negativo: ingreso permutado
archivo_c <- file.path(RUTA_SALIDAS, "fuga_control_negativo.rds")
if (!file.exists(archivo_c)) {
  set.seed(SEMILLA)
  z_permutado <- sample(datos$z)
  aux_perm <- aux_anidada(datos$x, z_permutado, fold1, entrena, RONDAS_AUX)
  saveRDS(f1_con_columna(aux_perm$columnas), archivo_c)
}
f1_permutado <- readRDS(archivo_c)
# Referencia con los mismos hiperparámetros y rondas, sin la columna auxiliar,
# para que la comparación aísle el efecto de la columna.
archivo_ref <- file.path(RUTA_SALIDAS, "fuga_referencia_sin_columna.rds")
if (!file.exists(archivo_ref)) saveRDS(f1_con_columna(NULL), archivo_ref)
f1_referencia <- readRDS(archivo_ref)
resultados$negativo <- data.table(
  prueba = "C. control negativo: F1 permutado - F1 sin columna (mismos parámetros)",
  valor = f1_permutado - f1_referencia,
  pasa = f1_permutado - f1_referencia <= TOLERANCIA_CONTROL_NEGATIVO
)

# D. Control positivo: columna con fuga
archivo_d <- file.path(RUTA_SALIDAS, "fuga_control_positivo.rds")
if (!file.exists(archivo_d)) {
  modelo <- entrenar_lgb(datos$x[entrena, ], datos$z[entrena], list(),
                         RONDAS_AUX, base = PARAMETROS_REGRESOR)
  columna_fugada <- predict(modelo, datos$x)
  saveRDS(f1_con_columna(rep(list(columna_fugada), N_FOLDS)), archivo_d)
}
f1_fugado <- readRDS(archivo_d)
resultados$positivo <- data.table(
  prueba = "D. control positivo: F1 con fuga - F1 auxiliar anidado",
  valor = f1_fugado - f1_auxiliar,
  pasa = f1_fugado - f1_auxiliar > 0
)

# E. Deriva de la columna final
aux_final <- readRDS(file.path(RUTA_SALIDAS, sprintf("aux_final_%s.rds", esc)))
sin_bog <- !datos$bogota
ks_holdout <- suppressWarnings(ks.test(aux_final$entrenamiento[sin_bog],
                                       aux_final$holdout[!datos$bogota_holdout])$statistic)
ks_test <- suppressWarnings(ks.test(aux_final$entrenamiento[sin_bog],
                                    aux_final$test)$statistic)
resultados$deriva <- data.table(
  prueba = c("E. KS entrenamiento vs holdout (sin Bogotá)",
             "E. KS entrenamiento vs test"),
  valor = c(ks_holdout, ks_test),
  pasa = c(ks_holdout, ks_test) < 0.05
)

tabla <- rbindlist(resultados)
tabla[, valor := round(valor, 4)]
fwrite(tabla, file.path(RUTA_SALIDAS, "fuga_resumen.csv"))
registrar("04 | F1 p1 ", esc, ": parsimonioso ", sprintf("%.4f", f1_parsimonioso),
          ", sin columna con parámetros de auxiliar ", sprintf("%.4f", f1_referencia),
          ", auxiliar ", sprintf("%.4f", f1_auxiliar), ", permutado ",
          sprintf("%.4f", f1_permutado), ", con fuga ", sprintf("%.4f", f1_fugado))
for (i in seq_len(nrow(tabla))) {
  registrar("04 | ", tabla$prueba[i], ": ", tabla$valor[i],
            if (tabla$pasa[i]) " (pasa)" else " (NO PASA)")
}
