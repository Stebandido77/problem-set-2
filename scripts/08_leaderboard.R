# ==============================================================================
# 08_leaderboard.R
# Junta todos los candidatos de la escalera: leaderboard interno ordenado por F1
# en holdout, gráfico de la escalera, top 3 para Kaggle, validación de todos los
# CSV, inventario de submissions y bitácora.
# ==============================================================================

source(here::here("scripts", "00_configuracion.R"))

resultados <- cargar_resultados_escalera()
leaderboard <- construir_leaderboard(resultados)
write_csv(leaderboard, file.path(RUTA_RESULTADOS, "leaderboard_interno.csv"))

# Versión compacta para las slides
leaderboard |>
  transmute(
    Nivel = etiqueta_nivel, Candidato = candidato, Algoritmo = algoritmo,
    Umbral = umbral, `F1 CV` = f1_cv, `sd CV` = f1_cv_sd, `F1 holdout` = f1_holdout,
    Precisión = precision_holdout, Recall = recall_holdout, Accuracy = accuracy_holdout,
    `Filtración (hogares)` = filtracion_n, `Subcobertura (hogares)` = subcobertura_n
  ) |>
  guardar_tabla("leaderboard_interno")

guardar_figura(graficar_escalera(leaderboard), "escalera_f1")

# ---- Sobreajuste -------------------------------------------------------------
con_sobreajuste <- leaderboard |> filter(sobreajuste)
if (nrow(con_sobreajuste) > 0) {
  warning(
    "Brecha CV - holdout > ", UMBRAL_SOBREAJUSTE, " en: ",
    paste(con_sobreajuste$candidato, collapse = ", "), call. = FALSE
  )
}

# ---- Top 3 para Kaggle -------------------------------------------------------
# Cada candidato ya se reajustó sobre el 100% del train con sus hiperparámetros
# y su umbral OOF (evaluar_candidato()); aquí se copian y se validan sus CSV.
seleccion <- elegir_top3(leaderboard, resultados)
carpeta_candidatos <- file.path(RUTA_SUBMISSIONS, "candidatos")
carpeta_top3 <- file.path(RUTA_SUBMISSIONS, "top3")
unlink(list.files(carpeta_top3, pattern = "\\.csv$", full.names = TRUE))

leer_submission <- function(ruta) {
  read_csv(ruta, col_types = cols(id = col_character(), pobre = col_integer()))
}

pwalk(seleccion$top, function(posicion, candidato) {
  origen <- file.path(carpeta_candidatos, resultados[[candidato]]$archivo_submission)
  destino <- file.path(carpeta_top3, paste0("top", posicion, "_", candidato, ".csv"))
  file.copy(origen, destino, overwrite = TRUE)
  validar_submission(leer_submission(destino))
})

if (nrow(seleccion$saltados) > 0) {
  message("Saltados por repetir algoritmo (concordancia de clases en holdout con el elegido):")
  print(seleccion$saltados)
}

# ---- Validación e inventario de todos los candidatos ------------------------
archivos_candidatos <- file.path(carpeta_candidatos, leaderboard$archivo_submission)
walk(archivos_candidatos, ~ validar_submission(leer_submission(.x)))

algoritmos_curso <- c(
  "LPM", "Logit", "Elastic Net", "CART", "Random Forest", "Boosting", "Naive Bayes"
)
inventario <- leaderboard |>
  count(algoritmo, name = "submissions") |>
  mutate(cuenta_para_el_curso = algoritmo %in% algoritmos_curso)
print(inventario)

n_submissions_curso <- sum(inventario$submissions[inventario$cuenta_para_el_curso])
n_algoritmos_curso <- sum(inventario$cuenta_para_el_curso)
message(
  "Inventario: ", n_submissions_curso, " CSV con algoritmos del curso (mínimo 15) y ",
  n_algoritmos_curso, " algoritmos distintos (mínimo 5)"
)
if (n_submissions_curso < 15 || n_algoritmos_curso < 5) {
  warning("El inventario no cumple el mínimo del enunciado", call. = FALSE)
}

# ---- Bitácora ----------------------------------------------------------------
actualizar_bitacora(leaderboard, seleccion$top)

# ---- Resumen en consola ------------------------------------------------------
leaderboard |>
  left_join(seleccion$top, by = "candidato") |>
  select(posicion, candidato, algoritmo, umbral, f1_cv, f1_cv_sd, f1_holdout,
         precision_holdout, recall_holdout, filtracion_n, subcobertura_n,
         brecha_cv_holdout, tiempo_min) |>
  mutate(across(where(is.double), ~ round(.x, 4))) |>
  print(n = Inf, width = Inf)
