# ==============================================================================
# main.R
# Reproduce todo el análisis de principio a fin.
# Requisito: los CSV de Kaggle en stores/raw/.
# Para validar rápido de punta a punta: Sys.setenv(PS2_MODO_RAPIDO = "TRUE")
# antes de correr (o MODO_RAPIDO <- TRUE en 00_configuracion.R).
# Los candidatos ya evaluados se cargan desde stores/modelos/ sin recalcular.
# ==============================================================================

scripts <- c(
  "01_preparar_datos.R",
  "02_exploracion.R",
  "03_particion.R",
  "04_niveles_1_2_3.R",
  "05_nivel_4_algoritmos.R",
  "06_nivel_5_balanceo.R",
  "07_nivel_6_complejo.R",
  "08_leaderboard.R",
  "09_comparacion.R",
  "10_mejor_modelo.R"
)

for (script in scripts) {
  message("\n==== Corriendo ", script, " ====")
  source(here::here("scripts", script), local = new.env())
}
