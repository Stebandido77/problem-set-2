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
    grafico, width = ancho, height = alto, dpi = 300, bg = "white"
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

# Calcula F1, precisión, recall y accuracy con 1 = pobre.
# Argumentos: verdad, prediccion = vectores 0/1.
# Devuelve: tibble con f1, precision, recall, accuracy, filtracion, subcobertura
#   y los conteos de la matriz de confusión (n_vp, n_fp, n_fn, n_vn).
calcular_metricas <- function(verdad, prediccion) {
  vp <- sum(verdad == 1 & prediccion == 1)
  fp <- sum(verdad == 0 & prediccion == 1)
  fn <- sum(verdad == 1 & prediccion == 0)
  vn <- sum(verdad == 0 & prediccion == 0)

  precision <- if (vp + fp == 0) 0 else vp / (vp + fp)
  recall    <- if (vp + fn == 0) 0 else vp / (vp + fn)
  f1        <- if (vp == 0) 0 else 2 * vp / (2 * vp + fp + fn)

  tibble(
    f1 = f1,
    precision = precision,
    recall = recall,
    accuracy = (vp + vn) / length(verdad),
    # Filtración: share de beneficiarios que no son pobres
    filtracion = 1 - precision,
    # Subcobertura: share de pobres que quedan por fuera
    subcobertura = 1 - recall,
    n_vp = vp, n_fp = fp, n_fn = fn, n_vn = vn
  )
}

# Busca el umbral de probabilidad que maximiza el F1 en predicciones fuera de muestra.
# Argumentos: verdad = vector 0/1; probabilidad = P(pobre) predicha;
#   grilla = umbrales a evaluar. Por defecto, 0,01 a 0,99 más los percentiles
#   0,5% a 99,5% de las probabilidades: así también funciona con modelos mal
#   calibrados (Naive Bayes concentra casi todo en 0 o 1 aunque ordene bien).
# Devuelve: lista con el umbral óptimo y la curva completa (útil para la slide de política).
buscar_umbral_f1 <- function(verdad, probabilidad, grilla = NULL) {
  if (is.null(grilla)) {
    percentiles <- quantile(probabilidad, seq(0.005, 0.995, by = 0.005), names = FALSE)
    grilla <- sort(unique(c(seq(0.01, 0.99, by = 0.01), percentiles)))
    # Un umbral <= 0 clasificaría a todos como pobres por construcción
    grilla <- grilla[grilla > 0]
  }

  # Versión vectorizada de calcular_metricas() para todos los umbrales a la vez:
  # cuántos pobres y no pobres tienen probabilidad >= u
  prob_pobres <- sort(probabilidad[verdad == 1])
  prob_no_pobres <- sort(probabilidad[verdad == 0])
  n_pobres <- length(prob_pobres)
  n_no_pobres <- length(prob_no_pobres)

  vp <- n_pobres - findInterval(grilla, prob_pobres, left.open = TRUE)
  fp <- n_no_pobres - findInterval(grilla, prob_no_pobres, left.open = TRUE)
  fn <- n_pobres - vp
  vn <- n_no_pobres - fp

  precision <- if_else(vp + fp == 0, 0, vp / (vp + fp))
  recall    <- if_else(vp + fn == 0, 0, vp / (vp + fn))

  curva <- tibble(
    umbral = grilla,
    f1 = if_else(vp == 0, 0, 2 * vp / (2 * vp + fp + fn)),
    precision = precision,
    recall = recall,
    accuracy = (vp + vn) / (n_pobres + n_no_pobres),
    filtracion = 1 - precision,
    subcobertura = 1 - recall,
    n_vp = vp, n_fp = fp, n_fn = fn, n_vn = vn
  )

  list(
    umbral = curva$umbral[which.max(curva$f1)],
    curva  = curva
  )
}

# Memoria en uso en este momento, en GB: la sesión de R que orquesta, la suma de
# sus procesos hijos (los workers de future) y el sistema completo.
# Devuelve: lista con sesion, workers, n_workers, sistema_usada y sistema_total
#   (NA si ps no puede leer algún proceso).
medir_ram <- function() {
  gb <- function(bytes) bytes / 1024^3
  tryCatch({
    sesion <- ps::ps_handle()
    hijos <- ps::ps_children(sesion, recursive = TRUE)
    rss_hijos <- map_dbl(hijos, function(proceso) {
      tryCatch(ps::ps_memory_info(proceso)[["rss"]], error = function(e) 0)
    })
    sistema <- ps::ps_system_memory()
    list(
      sesion = gb(ps::ps_memory_info(sesion)[["rss"]]),
      workers = gb(sum(rss_hijos)),
      n_workers = length(hijos),
      sistema_usada = gb(sistema$total - sistema$avail),
      sistema_total = gb(sistema$total)
    )
  }, error = function(e) {
    list(sesion = NA, workers = NA, n_workers = NA, sistema_usada = NA, sistema_total = NA)
  })
}

# Agrega una línea a stores/logs/corrida.log con la hora, el candidato, el evento
# y la RAM (ver medir_ram()). Sirve para ubicar en qué candidato y en qué etapa
# se dispara la memoria.
# Argumentos:
#   candidato      = nombre del candidato
#   evento         = "inicio", "fin de la CV", "fin", etc.
#   inicio         = hora de inicio (Sys.time()); si se da, agrega la duración
#   reiniciar_pico = si TRUE reinicia el máximo de gc(), para que el "pico heap R"
#                    del evento "fin" corresponda solo a este candidato
# Devuelve: la línea escrita (invisible). Nunca detiene la corrida si falla.
registrar_log <- function(candidato, evento, inicio = NULL, reiniciar_pico = FALSE) {
  memoria_gc <- gc(verbose = FALSE, reset = reiniciar_pico)
  pico_heap <- sum(memoria_gc[, ncol(memoria_gc)]) / 1024  # columna "max used (Mb)"
  ram <- medir_ram()

  linea <- sprintf(
    "%s | %-26s | %-20s | sesión R %5.2f GB | workers %5.2f GB (%s) | pico heap R %5.2f GB | sistema %5.1f / %4.1f GB%s",
    format(Sys.time(), "%Y-%m-%d %H:%M:%S"), candidato, evento,
    ram$sesion, ram$workers, ram$n_workers, pico_heap,
    ram$sistema_usada, ram$sistema_total,
    if (is.null(inicio)) "" else sprintf(" | %.1f min", as.numeric(difftime(Sys.time(), inicio, units = "mins")))
  )

  tryCatch(
    write(linea, file.path(RUTA_LOGS, "corrida.log"), append = TRUE),
    error = function(e) message("No se pudo escribir el log: ", conditionMessage(e))
  )
  invisible(linea)
}

# Revisa que una submission tenga exactamente el formato de sample_submission.csv:
# columnas id y pobre, una fila por hogar del test, mismos ids, sin NA y clases 0/1.
# Argumentos: submission = tibble con id y pobre; ruta_ejemplo = sample_submission.csv.
# Devuelve: TRUE invisible; se detiene con un error si algo no cumple.
validar_submission <- function(submission,
                               ruta_ejemplo = file.path(RUTA_CRUDOS, "sample_submission.csv")) {
  ejemplo <- read_csv(ruta_ejemplo, show_col_types = FALSE)

  stopifnot(
    "Las columnas deben ser id y pobre" = identical(names(submission), names(ejemplo)),
    "Número de filas distinto al test"  = nrow(submission) == nrow(ejemplo),
    "Los ids no coinciden con el test"  = setequal(submission$id, ejemplo$id),
    "Hay ids duplicados"                = !anyDuplicated(submission$id),
    "Hay NA en la submission"           = !anyNA(submission),
    "La clase debe ser entera 0/1"      = is.integer(submission$pobre) &&
                                          all(submission$pobre %in% c(0L, 1L))
  )
  invisible(TRUE)
}

# Escribe el CSV para Kaggle y (opcionalmente) registra la submission en la bitácora.
# Argumentos:
#   ids, clases    = id del hogar y predicción 0/1
#   algoritmo      = uno de los 7 permitidos por el enunciado, o "Regla" / "Stacking"
#                    (estos dos no cuentan para el mínimo de 5 algoritmos)
#   especificacion = descripción corta (variables, hiperparámetros, balanceo)
#   f1_cv          = F1 de validación cruzada
#   carpeta        = dónde escribir el CSV (por defecto stores/submissions)
#   nombre_archivo = nombre del CSV; si es NULL se arma con algoritmo y fecha
#   nivel          = nivel de la escalera (solo para la bitácora)
#   f1_holdout     = F1 en el holdout interno (solo para la bitácora)
#   registrar      = si TRUE agrega una fila a bitacora_kaggle.csv
# Devuelve: ruta del archivo escrito.
guardar_submission <- function(ids, clases, algoritmo, especificacion, f1_cv,
                               carpeta = RUTA_SUBMISSIONS, nombre_archivo = NULL,
                               nivel = NA_character_, f1_holdout = NA_real_,
                               registrar = TRUE) {
  algoritmos_validos <- c(
    "LPM", "Logit", "Elastic Net", "CART", "Random Forest", "Boosting", "Naive Bayes",
    "Regla", "Stacking"
  )
  stopifnot(algoritmo %in% algoritmos_validos)

  if (is.null(nombre_archivo)) {
    marca <- format(Sys.time(), "%Y%m%d_%H%M%S")
    nombre_archivo <- paste0(
      str_replace_all(str_to_lower(algoritmo), " ", "_"), "_", marca, ".csv"
    )
  }

  submission <- tibble(id = ids, pobre = as.integer(clases))
  validar_submission(submission)
  write_csv(submission, file.path(carpeta, nombre_archivo))

  if (registrar) {
    ruta_bitacora <- file.path(RUTA_SUBMISSIONS, "bitacora_kaggle.csv")
    registro <- tibble(
      fecha = as.character(Sys.Date()),
      archivo = nombre_archivo,
      nivel = as.character(nivel),
      candidato = NA_character_,
      algoritmo = algoritmo,
      especificacion = especificacion,
      f1_cv = round(f1_cv, 4),
      f1_holdout = round(f1_holdout, 4),
      top3 = NA_character_,
      f1_kaggle = NA_character_,  # F1 público; se llena a mano después de subir a Kaggle
      subida = "no"
    ) |>
      mutate(across(everything(), as.character))
    # Se une por nombre de columna: la bitácora tiene columnas anotadas a mano
    # (orden_subida, f1_kaggle_privado, ...) que un append desalinearía.
    previa <- if (file.exists(ruta_bitacora)) {
      read_csv(ruta_bitacora, show_col_types = FALSE, col_types = cols(.default = "c"))
    } else {
      NULL
    }
    write_csv(bind_rows(previa, registro), ruta_bitacora, na = "")
  }

  message("Submission guardada: ", nombre_archivo)
  invisible(file.path(carpeta, nombre_archivo))
}

# ---- Gráficos y tablas del mejor modelo ----------------------------------------
# Se usan dos veces en 10_mejor_modelo.R: para el mejor candidato de R (desde su
# .rds) y para el mejor del nivel 7 (desde las tablas que exporta Python).

# Importancia de las 15 variables más importantes.
# Argumentos: importancia = tibble (variable, importancia) normalizada a máximo 1;
#   nombre = candidato, para el subtítulo.
# Devuelve: objeto ggplot.
graficar_importancia <- function(importancia, nombre) {
  importancia |>
    slice_max(importancia, n = 15) |>
    mutate(variable = fct_reorder(variable, importancia)) |>
    ggplot(aes(importancia, variable)) +
    geom_col(fill = "#2a78d6", width = 0.7) +
    labs(
      title = "¿Qué predice la pobreza?",
      subtitle = paste0(nombre, " | importancia relativa (máximo = 1)"),
      x = "Importancia relativa", y = NULL
    )
}

# Trade-off entre filtración y subcobertura a lo largo del umbral.
# Argumentos: curva = curva de umbral OOF (umbral, f1, filtracion, subcobertura);
#   umbral_elegido = umbral elegido (línea punteada).
# Devuelve: objeto ggplot.
graficar_tradeoff <- function(curva, umbral_elegido) {
  curva |>
    select(umbral, F1 = f1, Filtración = filtracion, Subcobertura = subcobertura) |>
    pivot_longer(-umbral, names_to = "metrica", values_to = "valor") |>
    ggplot(aes(umbral, valor, color = metrica)) +
    geom_line(linewidth = 1) +
    geom_vline(xintercept = umbral_elegido, linetype = "dashed", color = "#52514e") +
    scale_color_manual(values = c(F1 = "#2a78d6", Filtración = "#eb6834", Subcobertura = "#1baf7a")) +
    scale_y_continuous(labels = scales::percent) +
    labs(
      title = "Mover el umbral cambia a quién se deja por fuera",
      subtitle = "Predicciones fuera de fold; la línea punteada es el umbral que maximiza F1",
      x = "Umbral de probabilidad para clasificar como pobre",
      y = NULL, color = NULL
    )
}

# Escenarios de política: máximo F1 frente a cubrir al menos 90% de los pobres
# (un hacedor de política que tolera más filtración a cambio de cobertura).
# Argumentos: curva = curva de umbral OOF; umbral_elegido = umbral elegido.
# Devuelve: tibble con una fila por criterio.
tabla_escenarios_politica <- function(curva, umbral_elegido) {
  bind_rows(
    curva |> slice_min(abs(umbral - umbral_elegido), n = 1, with_ties = FALSE) |>
      mutate(criterio = "Máximo F1"),
    curva |> filter(recall >= 0.90) |> slice_max(umbral, n = 1, with_ties = FALSE) |>
      mutate(criterio = "Cobertura >= 90%")
  ) |>
    select(criterio, umbral, f1, precision, recall, filtracion, subcobertura)
}
