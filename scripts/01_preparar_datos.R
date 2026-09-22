# ==============================================================================
# 01_preparar_datos.R
# Stage 1: limpia y une hogares con personas, y deja las bases listas en
# stores/procesados/. Solo conserva variables disponibles en train Y test.
# ==============================================================================

source(here::here("scripts", "00_configuracion.R"))

datos <- cargar_datos_crudos()

# ¿Qué variables quitaron del test? Esto también es insumo para la slide de datos.
ausentes_hogares  <- variables_solo_en_train(datos$train_hogares, datos$test_hogares)
ausentes_personas <- variables_solo_en_train(datos$train_personas, datos$test_personas)
guardar_tabla(ausentes_hogares, "variables_ausentes_test_hogares")
guardar_tabla(ausentes_personas, "variables_ausentes_test_personas")

train_hogar <- construir_base_hogar(datos$train_hogares, datos$train_personas)
test_hogar  <- construir_base_hogar(datos$test_hogares, datos$test_personas)

# Cancha nivelada: el train solo puede tener lo que también existe en el test
train_hogar <- train_hogar |>
  select(id, pobre, any_of(names(test_hogar))) |>
  preparar_objetivo()

stopifnot(
  !anyDuplicated(train_hogar$id),
  !anyDuplicated(test_hogar$id),
  setequal(setdiff(names(train_hogar), "pobre"), names(test_hogar))
)

saveRDS(train_hogar, file.path(RUTA_PROCESADOS, "train_hogar.rds"))
saveRDS(test_hogar,  file.path(RUTA_PROCESADOS, "test_hogar.rds"))

message("Train: ", nrow(train_hogar), " hogares | Test: ", nrow(test_hogar), " hogares")
