# =============================================================================
# particiones.R — Lectura de los ids fijos de limpio/ids/ y su alineación con la
# base de train.
# =============================================================================

N_PARTICIONES <- 3
N_FOLDS <- 5

# Lee el holdout y las tres particiones de 5 folds y las alinea con los ids.
# Argumentos: ids, vector de ids de la base de train (en su orden).
# Devuelve: lista con `holdout` (lógico) y `folds` (lista de 3 vectores enteros,
# fold 1..5 para el 80 % y NA para el holdout). Verifica que los folds sean
# disjuntos y cubran exactamente el 80 %.
cargar_particiones <- function(ids) {
  leer_ids <- function(archivo) fread(file.path(RUTA_IDS, archivo))$id
  ids_holdout <- leer_ids("holdout_ids.csv")
  ids_80 <- leer_ids("entrenamiento_80_ids.csv")
  stopifnot(length(intersect(ids_holdout, ids_80)) == 0)
  folds <- lapply(seq_len(N_PARTICIONES), function(p) {
    fold <- rep(NA_integer_, length(ids))
    for (k in seq_len(N_FOLDS)) {
      ids_k <- leer_ids(sprintf("particion_%d_fold_%d.csv", p, k))
      posiciones <- match(ids_k, ids)
      posiciones <- posiciones[!is.na(posiciones)]
      stopifnot(all(is.na(fold[posiciones])))
      fold[posiciones] <- k
    }
    stopifnot(identical(!is.na(fold), ids %in% ids_80))
    fold
  })
  list(holdout = ids %in% ids_holdout, folds = folds)
}
