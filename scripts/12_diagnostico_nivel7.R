# ==============================================================================
# 12_diagnostico_nivel7.R
# Nivel 7, etapa 0: diagnóstico antes de construir variables nuevas.
#   1. Qué columnas de hogares y personas existen en train y en test, con su
#      cobertura (% sin NA) en cada base.
#   2. Importancia (gain) de n6_lightgbm sumada por variable original (las
#      dummies de una categórica se suman).
#   3. Dónde se concentran los errores de n6_lightgbm en el holdout: tamaño del
#      hogar, zona, tenencia, departamento y ciudad.
# Salida: views/tablas/nivel7/diagnostico_*.csv
# ==============================================================================

source(here::here("scripts", "00_configuracion.R"))

ruta_salida <- file.path(RUTA_TABLAS, "nivel7")
dir.create(ruta_salida, recursive = TRUE, showWarnings = FALSE)

# Cobertura de cada columna: share de filas sin NA.
# Argumentos: datos = data frame.
# Devuelve: tibble (variable, cobertura).
calcular_cobertura <- function(datos) {
  tibble(variable = names(datos), cobertura = map_dbl(datos, ~ mean(!is.na(.x))))
}

# Precisión, recall, F1 y cuota de errores de un candidato por subgrupo.
# Argumentos: datos = tibble con verdad (0/1), prediccion (0/1) y la columna de
#   grupo; grupo = columna por la que se agrupa.
# Devuelve: tibble con una fila por grupo.
resumir_errores <- function(datos, grupo) {
  datos |>
    group_by(grupo = as.character({{ grupo }})) |>
    summarise(
      hogares = n(), pobres = sum(verdad), tasa_pobreza = mean(verdad),
      vp = sum(verdad == 1 & prediccion == 1),
      fn = sum(verdad == 1 & prediccion == 0),
      fp = sum(verdad == 0 & prediccion == 1),
      .groups = "drop"
    ) |>
    mutate(
      recall = vp / pmax(vp + fn, 1),
      precision = vp / pmax(vp + fp, 1),
      f1 = 2 * vp / pmax(2 * vp + fp + fn, 1),
      cuota_fn = fn / sum(fn),
      cuota_fp = fp / sum(fp)
    )
}

# ---- 1. Columnas en train y test -------------------------------------------------
datos <- cargar_datos_crudos()
for (nivel in c("hogares", "personas")) {
  cobertura <- full_join(
    calcular_cobertura(datos[[paste0("train_", nivel)]]) |> rename(cobertura_train = cobertura),
    calcular_cobertura(datos[[paste0("test_", nivel)]]) |> rename(cobertura_test = cobertura),
    by = "variable"
  ) |>
    mutate(en_train = !is.na(cobertura_train), en_test = !is.na(cobertura_test))
  write_csv(cobertura, file.path(ruta_salida, paste0("diagnostico_columnas_", nivel, ".csv")))
}
rm(datos)
gc(verbose = FALSE)

# ---- 2. Importancia de n6_lightgbm por variable original -------------------------
n6 <- readRDS(file.path(RUTA_MODELOS, "n6_lightgbm.rds"))
categoricas <- c("jefe_educ", "jefe_posicion", "jefe_oficio", "clase", "dominio",
                 "departamento", "tenencia")
importancia <- n6$importancia |>
  mutate(variable_original = map_chr(variable, function(v) {
    prefijo <- categoricas[str_starts(v, paste0(categoricas, "_"))]
    if (length(prefijo) > 0) prefijo[1] else v
  })) |>
  group_by(variable_original) |>
  summarise(gain = sum(importancia), .groups = "drop") |>
  mutate(pct_gain = 100 * gain / sum(gain)) |>
  arrange(desc(pct_gain))
write_csv(importancia, file.path(ruta_salida, "diagnostico_importancia_n6.csv"))

# ---- 3. Errores de n6_lightgbm en el holdout -------------------------------------
holdout <- cargar_particion()$particion$holdout
errores <- n6$prediccion_holdout |>
  rename(prediccion = clase) |>
  left_join(holdout |> select(id, pobre, n_personas, zona = clase, departamento, dominio, tenencia),
            by = "id") |>
  mutate(
    verdad = as.integer(pobre == "pobre"),
    tamano = cut(n_personas, c(0, 1, 2, 3, 4, 5, 6, 8, Inf),
                 labels = c("1", "2", "3", "4", "5", "6", "7-8", "9+"))
  )
stopifnot(nrow(errores) == nrow(holdout), !anyNA(errores$verdad))

walk(c("tamano", "zona", "tenencia", "departamento", "dominio"), function(grupo) {
  write_csv(resumir_errores(errores, .data[[grupo]]),
            file.path(ruta_salida, paste0("diagnostico_errores_", grupo, ".csv")))
})

# Probabilidad de los falsos negativos: ¿casi aciertos o invisibles para el modelo?
errores |>
  filter(verdad == 1, prediccion == 0) |>
  summarise(
    n = n(), p10 = quantile(probabilidad, 0.1), mediana = median(probabilidad),
    p90 = quantile(probabilidad, 0.9), cerca_umbral = mean(probabilidad > n6$umbral - 0.1)
  ) |>
  write_csv(file.path(ruta_salida, "diagnostico_probabilidad_fn.csv"))

message("Diagnóstico del nivel 7 en ", ruta_salida)
