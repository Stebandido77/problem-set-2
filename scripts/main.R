# ==============================================================================
# main.R
# Reproduce todo el análisis de principio a fin.
# Requisito: los CSV de Kaggle en stores/raw/.
# ==============================================================================

scripts <- c(
  "01_preparar_datos.R",
  "02_exploracion.R",
  "03_modelos_lineales.R",
  "04_modelos_arboles.R",
  "05_naive_bayes.R",
  "06_balanceo_clases.R",
  "07_comparacion.R",
  "08_mejor_modelo.R"
)

for (script in scripts) {
  message("\n==== Corriendo ", script, " ====")
  source(here::here("scripts", script), local = new.env())
}
