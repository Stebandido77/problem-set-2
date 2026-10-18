# ==============================================================================
# 07_nivel_6_complejo.R
# Escalera, nivel 6 y nivel E:
#   6. LightGBM con búsqueda amplia (latin hypercube de 7 hiperparámetros) y un
#      stacking de los mejores candidatos, construido con predicciones OOF sobre
#      los mismos folds.
#   E. Encuesta corta: el mejor algoritmo del nivel 4 y un Logit con solo 10
#      variables baratas y verificables. Es el contrapunto de política: ¿cuánto F1
#      se pierde con un instrumento que se puede aplicar en campo?
# ==============================================================================

source(here::here("scripts", "00_configuracion.R"))

entrada <- cargar_particion()
particion <- entrada$particion
folds <- entrada$folds
niveles <- definir_niveles()

# ---- Nivel 6: LightGBM -------------------------------------------------------
correr_candidato_escalera("lightgbm", "n6_lightgbm", "6", niveles$nivel_3, particion, folds)

# ---- Nivel 6: stacking -------------------------------------------------------
# Miembros: el mejor candidato de cada algoritmo (por F1 de CV) entre los niveles
# 3 a 6 con la base completa, y de ahí los 4 mejores. Un solo representante por
# algoritmo para que el stacking combine modelos que se equivocan distinto.
candidatos_stacking <- cargar_resultados_escalera("^n[3-6]_") |>
  purrr::discard(~ .x$algoritmo == "Stacking")  # scales::discard() tapa al de purrr

miembros <- tibble(
  nombre = names(candidatos_stacking),
  algoritmo = map_chr(candidatos_stacking, "algoritmo"),
  f1_cv = map_dbl(candidatos_stacking, ~ .x$metricas_cv$f1_cv)
) |>
  group_by(algoritmo) |>
  slice_max(f1_cv, n = 1, with_ties = FALSE) |>
  ungroup() |>
  slice_max(f1_cv, n = 4) |>
  pull(nombre)

correr_stacking("n6_stacking", candidatos_stacking[miembros], particion)

# ---- Nivel E: encuesta corta -------------------------------------------------
ganador_nivel_4 <- mejor_por_cv(cargar_resultados_escalera("^n4_"))
id_ganador <- str_remove(ganador_nivel_4$nombre, "^n4_")

correr_candidato_escalera(id_ganador, paste0("nE_", id_ganador), "E", niveles$nivel_e,
                          particion, folds)
correr_candidato_escalera("logit", "nE_logit", "E", niveles$nivel_e, particion, folds)

plan(sequential)
