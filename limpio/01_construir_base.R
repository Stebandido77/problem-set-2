# =============================================================================
# 01_construir_base.R — Construye la base parsimoniosa de hogar para train y
# test y verifica su consistencia. Guarda base_train.rds y base_test.rds.
# =============================================================================

source(here::here("limpio", "00_configuracion.R"))

registrar("01 | Construcción de la base", if (MODO_PRUEBA) " (prueba)" else "")

train <- leer_muestra("train")
test <- leer_muestra("test")

if (MODO_PRUEBA) {
  set.seed(SEMILLA)
  submuestrear <- function(m, fraccion) {
    ids <- sample(m$hogares$id, round(fraccion * nrow(m$hogares)))
    list(hogares = m$hogares[id %in% ids], personas = m$personas[id %in% ids])
  }
  train <- submuestrear(train, 0.05)
  test <- submuestrear(test, 0.05)
}

referencias <- crear_referencias(train$hogares)

# Verificación: la unidad de gasto reconstruida coincide con Npersug
n_ug <- train$personas[!P6050 %in% PARENTESCOS_FUERA_UG, .N, by = id]
control <- merge(train$hogares[, .(id, Npersug)], n_ug, by = "id")
stopifnot(all(control$Npersug == control$N))

base_train <- construir_base_hogar(train, referencias)
base_test <- construir_base_hogar(test, referencias)

variables <- variables_explicativas(base_train)
stopifnot(identical(variables, variables_explicativas(base_test)))
stopifnot(!anyDuplicated(base_train$id), !anyDuplicated(base_test$id))

faltantes <- data.table(
  variable = variables,
  na_train = sapply(base_train[, ..variables], function(v) mean(is.na(v))),
  na_test = sapply(base_test[, ..variables], function(v) mean(is.na(v)))
)
fwrite(faltantes, file.path(RUTA_SALIDAS, "faltantes_base.csv"))
registrar("01 | ", nrow(base_train), " hogares train, ", nrow(base_test),
          " test, ", length(variables), " variables")
registrar("01 | Variables con NA > 1 %: ",
          paste(faltantes[na_train > 0.01, variable], collapse = ", "))

saveRDS(base_train, file.path(RUTA_SALIDAS, "base_train.rds"))
saveRDS(base_test, file.path(RUTA_SALIDAS, "base_test.rds"))
