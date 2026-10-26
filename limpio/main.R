# =============================================================================
# main.R — Etapa 17 (diseño limpio) de punta a punta.
#   "C:\Program Files\R\R-4.6.0\bin\Rscript.exe" limpio/main.R [--prueba]
# Cada script guarda sus resultados en limpio/salidas/ (o salidas_prueba/) y se
# salta lo que ya existe, así que una corrida interrumpida se reanuda.
# =============================================================================

argumentos <- commandArgs(trailingOnly = TRUE)
rscript <- file.path(R.home("bin"), "Rscript.exe")
salidas <- here::here("limpio", if ("--prueba" %in% argumentos) "salidas_prueba" else "salidas")
dir.create(salidas, showWarnings = FALSE)
marca <- file.path(salidas, "fin.txt")
unlink(marca)
for (script in c("01_construir_base.R", "02_columna_auxiliar.R",
                 "03_evaluar_variantes.R", "04_test_fuga.R", "05_resumen.R",
                 "06_submission_final.R")) {
  estado <- system2(rscript, c(shQuote(here::here("limpio", script)), argumentos))
  if (estado != 0) {
    writeLines(paste("FALLO", script, format(Sys.time())), marca)
    stop("Falló ", script)
  }
}
writeLines(paste("OK", format(Sys.time())), marca)
