# =============================================================================
# 02_columna_auxiliar.R — Construye la columna auxiliar de ingreso
# (predicción de log(1 + Ingpcug / Lp)) fuera de fold y anidada, para cada
# partición y escenario, y la versión final del 80 % para holdout y test.
# Cada resultado se guarda en su propio .rds y se salta si ya existe, para que
# la corrida sea reanudable.
# =============================================================================

source(here::here("limpio", "00_configuracion.R"))
datos <- cargar_datos_modelo()
registrar("02 | Columna auxiliar anidada")

# Rondas del regresor: una sola vez, con parada temprana entrenando en los
# folds 2-5 de la partición 1 y validando en su fold 1. Se fijan para todo lo
# demás (no se reajustan por fold).
archivo_rondas <- file.path(RUTA_SALIDAS, "rondas_regresor.rds")
if (!file.exists(archivo_rondas)) {
  f1 <- datos$folds[[1]]
  modelo <- entrenar_lgb(
    datos$x[f1 != 1, ], datos$z[f1 != 1], list(), MAX_RONDAS,
    datos$x[f1 == 1, ], datos$z[f1 == 1], base = PARAMETROS_REGRESOR
  )
  saveRDS(modelo$best_iter, archivo_rondas)
}
RONDAS_AUX <- readRDS(archivo_rondas)
registrar("02 | Rondas del regresor auxiliar: ", RONDAS_AUX)

for (p in seq_len(N_PARTICIONES)) {
  for (escenario in ESCENARIOS) {
    archivo <- file.path(RUTA_SALIDAS, sprintf("aux_p%d_%s.rds", p, escenario))
    if (file.exists(archivo)) next
    inicio <- Sys.time()
    aux <- aux_anidada(datos$x, datos$z, datos$folds[[p]],
                       datos$entrena[[escenario]], RONDAS_AUX)
    saveRDS(aux, archivo)
    registrar(sprintf("02 | aux p%d %s listo en %.1f min", p, escenario,
                      as.numeric(difftime(Sys.time(), inicio, units = "mins"))))
  }
}

for (escenario in ESCENARIOS) {
  archivo <- file.path(RUTA_SALIDAS, sprintf("aux_final_%s.rds", escenario))
  if (file.exists(archivo)) next
  aux <- aux_final(datos$x, datos$z, datos$folds[[1]],
                   datos$entrena[[escenario]],
                   list(holdout = datos$x_holdout, test = datos$x_test),
                   RONDAS_AUX)
  saveRDS(aux, archivo)
  registrar("02 | aux final ", escenario, " listo")
}
