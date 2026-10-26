# =============================================================================
# 05_resumen.R — Tabla comparativa de variantes y escenarios, diferencias
# pareadas por partición, decisión según la regla de 03 y predicciones del
# test de la combinación ganadora (formato de Kaggle, en limpio/salidas/; no se
# registra como submission).
# =============================================================================

source(here::here("limpio", "00_configuracion.R"))
registrar("05 | Resumen")

cv <- fread(file.path(RUTA_SALIDAS, "cv_confirmacion.csv"))
holdout <- fread(file.path(RUTA_SALIDAS, "holdout.csv"))

resumen_cv <- cv[, .(
  f1_cv_sin_bogota = mean(f1_sin_bogota), de_cv_sin_bogota = sd(f1_sin_bogota),
  precision_cv_sin_bogota = mean(precision_sin_bogota),
  recall_cv_sin_bogota = mean(recall_sin_bogota),
  f1_cv_completo = mean(f1_completo)
), by = .(variante, escenario)]
tabla <- merge(resumen_cv, holdout[, .(variante, escenario, umbral,
                                       f1_holdout_completo = f1_completo,
                                       f1_holdout_sin_bogota = f1_sin_bogota,
                                       precision_holdout_sin_bogota = precision_sin_bogota,
                                       recall_holdout_sin_bogota = recall_sin_bogota,
                                       tasa_predicha_test)],
               by = c("variante", "escenario"))
setorder(tabla, -f1_cv_sin_bogota)
num <- names(tabla)[sapply(tabla, is.numeric)]
tabla[, (num) := lapply(.SD, round, 4), .SDcols = num]
fwrite(tabla, file.path(RUTA_SALIDAS, "tabla_comparacion.csv"))

# Diferencias pareadas por partición (auxiliar - parsimonioso, mismo escenario)
pareadas <- dcast(cv, escenario + particion ~ variante, value.var = "f1_sin_bogota")
pareadas[, diferencia := round(auxiliar - parsimonioso, 4)]
fwrite(pareadas, file.path(RUTA_SALIDAS, "diferencias_pareadas.csv"))

ganador <- tabla[1]
registrar(sprintf("05 | Ganador por CV sin Bogotá: %s %s (CV %.4f, holdout sin Bogotá %.4f)",
                  ganador$variante, ganador$escenario, ganador$f1_cv_sin_bogota,
                  ganador$f1_holdout_sin_bogota))

final <- readRDS(file.path(RUTA_SALIDAS, sprintf("final_%s_%s.rds",
                                                 ganador$variante, ganador$escenario)))
ids_test <- readRDS(file.path(RUTA_SALIDAS, "base_test.rds"))$id
fwrite(data.table(id = ids_test, pobre = as.integer(final$test >= final$umbral)),
       file.path(RUTA_SALIDAS, sprintf("prediccion_test_%s_%s.csv",
                                       ganador$variante, ganador$escenario)))
print(tabla)
print(pareadas)
