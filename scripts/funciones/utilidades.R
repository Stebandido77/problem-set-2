# ==============================================================================
# funciones/utilidades.R
# Métricas, gráficos, exportación de tablas y registro de submissions.
# ==============================================================================

# Tema gráfico común para que todas las figuras de las slides se vean iguales.
tema_proyecto <- function(tamano_base = 13) {
  theme_minimal(base_size = tamano_base) +
    theme(
      plot.title       = element_text(face = "bold"),
      plot.title.position = "plot",
      legend.position  = "bottom",
      panel.grid.minor = element_blank()
    )
}

# Guarda una figura en views/figuras con tamaño listo para diapositiva 16:9.
guardar_figura <- function(grafico, nombre, ancho = 10, alto = 5.6) {
  ggsave(
    file.path(RUTA_FIGURAS, paste0(nombre, ".pdf")),
    grafico, width = ancho, height = alto
  )
  ggsave(
    file.path(RUTA_FIGURAS, paste0(nombre, ".png")),
    grafico, width = ancho, height = alto, dpi = 300
  )
  invisible(grafico)
}

# Exporta una tabla a CSV y a LaTeX (para meterla en las slides sin pantallazos).
guardar_tabla <- function(tabla, nombre, digitos = 3) {
  write_csv(tabla, file.path(RUTA_TABLAS, paste0(nombre, ".csv")))
  tabla_tex <- knitr::kable(tabla, format = "latex", booktabs = TRUE, digits = digitos)
  writeLines(tabla_tex, file.path(RUTA_TABLAS, paste0(nombre, ".tex")))
  invisible(tabla)
}

# Calcula F1, precisión y recall con 1 = pobre.
# Argumentos: verdad, prediccion = vectores 0/1.
# Devuelve: tibble con f1, precision, recall, filtracion y subcobertura.
calcular_metricas <- function(verdad, prediccion) {
  vp <- sum(verdad == 1 & prediccion == 1)
  fp <- sum(verdad == 0 & prediccion == 1)
  fn <- sum(verdad == 1 & prediccion == 0)

  precision <- if (vp + fp == 0) 0 else vp / (vp + fp)
  recall    <- if (vp + fn == 0) 0 else vp / (vp + fn)
  f1        <- if (vp == 0) 0 else 2 * vp / (2 * vp + fp + fn)

  tibble(
    f1 = f1,
    precision = precision,
    recall = recall,
    # Filtración: share de beneficiarios que no son pobres
    filtracion = 1 - precision,
    # Subcobertura: share de pobres que quedan por fuera
    subcobertura = 1 - recall
  )
}

# Busca el umbral de probabilidad que maximiza el F1 en predicciones fuera de muestra.
# Argumentos: verdad = vector 0/1; probabilidad = P(pobre) predicha.
# Devuelve: lista con el umbral óptimo y la curva completa (útil para la slide de política).
buscar_umbral_f1 <- function(verdad, probabilidad, grilla = seq(0.05, 0.95, by = 0.01)) {
  curva <- map_dfr(grilla, function(u) {
    calcular_metricas(verdad, as.integer(probabilidad >= u)) |>
      mutate(umbral = u, .before = 1)
  })

  list(
    umbral = curva$umbral[which.max(curva$f1)],
    curva  = curva
  )
}

# Escribe el CSV para Kaggle y registra la submission en la bitácora.
# Argumentos:
#   ids, clases    = id del hogar y predicción 0/1
#   algoritmo      = uno de los 7 permitidos por el enunciado
#   especificacion = descripción corta (variables, hiperparámetros, balanceo)
#   f1_cv          = F1 de validación cruzada
# Devuelve: ruta del archivo escrito.
guardar_submission <- function(ids, clases, algoritmo, especificacion, f1_cv) {
  algoritmos_validos <- c(
    "LPM", "Logit", "Elastic Net", "CART", "Random Forest", "Boosting", "Naive Bayes"
  )
  stopifnot(algoritmo %in% algoritmos_validos)

  marca <- format(Sys.time(), "%Y%m%d_%H%M%S")
  nombre_archivo <- paste0(
    str_replace_all(str_to_lower(algoritmo), " ", "_"), "_", marca, ".csv"
  )

  # Formato de Kaggle: verificar contra sample_submission.csv
  write_csv(
    tibble(id = ids, pobre = as.integer(clases)),
    file.path(RUTA_SUBMISSIONS, nombre_archivo)
  )

  ruta_bitacora <- file.path(RUTA_SUBMISSIONS, "bitacora_kaggle.csv")
  registro <- tibble(
    fecha = as.character(Sys.Date()),
    archivo = nombre_archivo,
    algoritmo = algoritmo,
    especificacion = especificacion,
    f1_cv = round(f1_cv, 4),
    f1_publico = NA_real_  # se llena a mano después de subir a Kaggle
  )
  write_csv(registro, ruta_bitacora, append = file.exists(ruta_bitacora))

  message("Submission guardada: ", nombre_archivo)
  invisible(file.path(RUTA_SUBMISSIONS, nombre_archivo))
}
