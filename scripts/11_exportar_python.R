# ==============================================================================
# 11_exportar_python.R
# Nivel 7: exporta a stores/procesados/*.parquet todo lo que necesita el código
# de python/ para trabajar sobre exactamente los mismos datos, folds y holdout
# que la escalera de R. Python NO rehace la partición: la lee de aquí.
#
# Archivos que produce:
#   hogar_nivel7.parquet        base de hogar del nivel 3 (train + test) más las
#                               columnas crudas de hogar presentes en test que el
#                               nivel 3 no usa (P5100, Npersug, Li, Fex_c, Fex_dpto)
#   personas_nivel7.parquet     personas de train y test: columnas crudas presentes
#                               en ambos + las variables de limpiar_personas()
#   ingreso_personas_train.parquet  ingresos individuales (solo train; objetivo
#                               del modelo auxiliar, nunca predictores)
#   ingreso_hogar_train.parquet ingreso per cápita del hogar (solo train)
#   particion_nivel7.parquet    id, pobre, fold_id (1-5; NA en holdout y test),
#                               holdout (TRUE/FALSE) y base (train/test)
#   n6_lightgbm_predicciones.parquet  OOF y holdout de n6_lightgbm (referencia)
#   resumen_particion_r.csv     conteos por fold y tasa de pobreza, para el test
#                               de python/tests/ que verifica la partición
# ==============================================================================

source(here::here("scripts", "00_configuracion.R"))
pacman::p_load(arrow)

datos <- cargar_datos_crudos()
entrada <- cargar_particion()
particion <- entrada$particion
folds <- entrada$folds

# ---- Partición -----------------------------------------------------------------
# Cada hogar del 80% recibe el fold en el que quedó en la parte de evaluación
fold_por_id <- map2_dfr(folds$splits, seq_along(folds$splits), function(division, k) {
  tibble(id = assessment(division)$id, fold_id = k)
})

tabla_particion <- bind_rows(
  particion$entrenamiento |> select(id, pobre) |> mutate(holdout = FALSE),
  particion$holdout |> select(id, pobre) |> mutate(holdout = TRUE)
) |>
  left_join(fold_por_id, by = "id") |>
  mutate(base = "train", pobre = as.integer(pobre == "pobre")) |>
  bind_rows(tibble(id = particion$test$id, pobre = NA_integer_, holdout = FALSE,
                   fold_id = NA_integer_, base = "test"))

stopifnot(
  !anyDuplicated(tabla_particion$id),
  all(!is.na(tabla_particion$fold_id[tabla_particion$base == "train" & !tabla_particion$holdout])),
  all(is.na(tabla_particion$fold_id[tabla_particion$holdout]))
)

resumen_particion_r <- tabla_particion |>
  filter(base == "train") |>
  mutate(grupo = if_else(holdout, "holdout", paste0("fold_", fold_id))) |>
  group_by(grupo) |>
  summarise(hogares = n(), pobres = sum(pobre), tasa_pobreza = mean(pobre), .groups = "drop")

# ---- Hogares -------------------------------------------------------------------
columnas_extra_hogar <- c("P5100", "Npersug", "Li", "Fex_c", "Fex_dpto")
extra_hogar <- bind_rows(datos$train_hogares, datos$test_hogares) |>
  select(id, all_of(columnas_extra_hogar))

hogar <- bind_rows(
  particion$completo |> select(-pobre),
  particion$test
) |>
  left_join(extra_hogar, by = "id")
stopifnot(nrow(hogar) == nrow(tabla_particion), !anyDuplicated(hogar$id))

# ---- Personas ------------------------------------------------------------------
# Solo columnas crudas que existen en ambas bases; los ingresos de train van aparte
columnas_comunes <- intersect(names(datos$train_personas), names(datos$test_personas))

preparar_personas <- function(personas) {
  crudas <- personas |> select(all_of(columnas_comunes))
  limpias <- limpiar_personas(personas) |> select(-id)
  bind_cols(crudas, limpias)
}
personas <- bind_rows(
  preparar_personas(datos$train_personas),
  preparar_personas(datos$test_personas)
)
stopifnot(!anyDuplicated(personas[c("id", "Orden")]))

columnas_ingreso <- c("Impa", "Isa", "Ie", "Imdi", "Impaes", "Isaes", "Iees", "Imdies",
                      "Iof1", "Iof2", "Iof3h", "Iof3i", "Iof6", "Ingtotob", "Ingtotes",
                      "Ingtot", "P6500", "P7070")
ingreso_personas <- datos$train_personas |> select(id, Orden, any_of(columnas_ingreso))
ingreso_hogar <- datos$train_hogares |> select(id, Ingtotug, Ingpcug, Lp, Pobre)

# ---- Predicciones de referencia de n6_lightgbm ---------------------------------
n6 <- readRDS(file.path(RUTA_MODELOS, "n6_lightgbm.rds"))
n6_predicciones <- bind_rows(
  n6$oof |>
    mutate(id = particion$entrenamiento$id[.row], tipo = "oof") |>
    select(id, tipo, fold, probabilidad),
  n6$prediccion_holdout |> transmute(id, tipo = "holdout", fold = NA_character_, probabilidad)
)

# ---- Escritura -----------------------------------------------------------------
write_parquet(hogar, file.path(RUTA_PROCESADOS, "hogar_nivel7.parquet"))
write_parquet(personas, file.path(RUTA_PROCESADOS, "personas_nivel7.parquet"))
write_parquet(ingreso_personas, file.path(RUTA_PROCESADOS, "ingreso_personas_train.parquet"))
write_parquet(ingreso_hogar, file.path(RUTA_PROCESADOS, "ingreso_hogar_train.parquet"))
write_parquet(tabla_particion, file.path(RUTA_PROCESADOS, "particion_nivel7.parquet"))
write_parquet(n6_predicciones, file.path(RUTA_PROCESADOS, "n6_lightgbm_predicciones.parquet"))
write_csv(resumen_particion_r, file.path(RUTA_PROCESADOS, "resumen_particion_r.csv"))

print(resumen_particion_r)
message("Exportado: ", nrow(hogar), " hogares y ", nrow(personas), " personas")
