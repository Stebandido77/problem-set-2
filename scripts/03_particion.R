# ==============================================================================
# 03_particion.R
# Separa el holdout estratificado del 20% (a nivel hogar) y crea los folds de CV
# sobre el 80% restante. Todos los candidatos de la escalera usan exactamente
# esta partición y estos folds (requisito para comparar y para el stacking).
# ==============================================================================

source(here::here("scripts", "00_configuracion.R"))

bases <- cargar_bases_modelado()
particion <- crear_particion_holdout(bases$train, bases$test)
folds <- crear_folds(particion$entrenamiento)

# ---- Chequeos anti-fuga ------------------------------------------------------
ids_folds <- unique(unlist(map(folds$splits, ~ analysis(.x)$id)))
stopifnot(
  "El holdout se cruza con el entrenamiento" =
    !any(particion$holdout$id %in% particion$entrenamiento$id),
  "Hay ids del holdout dentro de los folds" =
    !any(particion$holdout$id %in% ids_folds),
  "Hogares duplicados" = !anyDuplicated(particion$entrenamiento$id)
)

resumen_particion <- tibble(
  base = c("entrenamiento (CV)", "holdout", "reajuste final", "test"),
  hogares = c(nrow(particion$entrenamiento), nrow(particion$holdout),
              nrow(particion$completo), nrow(particion$test)),
  tasa_pobreza = c(
    mean(particion$entrenamiento$pobre == "pobre"),
    mean(particion$holdout$pobre == "pobre"),
    mean(particion$completo$pobre == "pobre"),
    NA
  )
)
print(resumen_particion)
guardar_tabla(resumen_particion, "resumen_particion")

archivo <- if (MODO_RAPIDO) "particion_rapida.rds" else "particion.rds"
saveRDS(list(particion = particion, folds = folds), file.path(RUTA_PROCESADOS, archivo))
