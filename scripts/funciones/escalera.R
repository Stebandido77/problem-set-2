# ==============================================================================
# funciones/escalera.R
# Escalera de modelos: corre candidatos por nivel, arma el stacking, construye
# el leaderboard interno, elige el top 3 para Kaggle y mantiene la bitácora.
# ==============================================================================

# Etiquetas de los niveles de la escalera (en orden).
ETIQUETAS_NIVELES <- c(
  "1" = "1. Naive",
  "2" = "2. Base",
  "3" = "3. Base + variables",
  "4" = "4. Algoritmos",
  "5" = "5. Desbalance",
  "6" = "6. Complejo",
  "E" = "E. Encuesta corta"
)

# Carga la partición holdout/folds que produce 03_particion.R.
# Devuelve: lista con particion y folds.
cargar_particion <- function() {
  archivo <- if (MODO_RAPIDO) "particion_rapida.rds" else "particion.rds"
  readRDS(file.path(RUTA_PROCESADOS, archivo))
}

# Peso relativo de la clase pobre: n_no_pobre / n_pobre en el entrenamiento.
calcular_razon_pesos <- function(datos) {
  sum(datos$pobre == "no_pobre") / sum(datos$pobre == "pobre")
}

# Elige la fuente de paralelismo según el algoritmo (ver 00_configuracion.R):
#   - ranger, xgboost y lightgbm: plan secuencial; el motor usa N_NUCLEOS hilos.
#   - Logit, Elastic Net, CART y Naive Bayes: future multisession con a lo sumo
#     N_WORKERS_MAX workers.
#   - LPM y reglas: no pasan por tune (sus folds ya corren en secuencia).
# Argumento: id_algoritmo = "lpm" o un nombre válido de especificar_algoritmo().
# Devuelve: el número de workers registrados (invisible).
configurar_paralelismo <- function(id_algoritmo) {
  if (id_algoritmo %in% c("lpm", ALGORITMOS_CON_HILOS)) {
    plan(sequential)
    return(invisible(1L))
  }
  registrar_nucleos(min(N_NUCLEOS, N_WORKERS_MAX))
  invisible(min(N_NUCLEOS, N_WORKERS_MAX))
}

# Corre un candidato de la escalera con un algoritmo, un nivel (conjunto de
# variables) y una estrategia de balanceo. Arma la receta común y llama a
# correr_algoritmo() / correr_lpm() con el esquema de holdout.
# Argumentos:
#   id_algoritmo = "lpm" o un nombre válido de especificar_algoritmo()
#   nombre       = identificador único del candidato (nombre del .rds y del CSV)
#   nivel        = "1" a "6" o "E"
#   variables    = predictores del nivel (ver definir_niveles())
#   particion, folds = salidas de cargar_particion()
#   balanceo     = "ninguno", "smote", "submuestreo" o "pesos"
# Devuelve: salida de evaluar_candidato().
correr_candidato_escalera <- function(id_algoritmo, nombre, nivel, variables,
                                      particion, folds, balanceo = "ninguno") {
  # Si el candidato ya está en caché no hace falta levantar workers
  if (!file.exists(file.path(RUTA_MODELOS, paste0(nombre, ".rds")))) {
    configurar_paralelismo(id_algoritmo)
  }
  # Al terminar (o si falla) se cierran los workers y se libera su memoria
  on.exit({
    plan(sequential)
    gc(verbose = FALSE)
  }, add = TRUE)

  receta <- crear_receta(particion$entrenamiento, balanceo = balanceo, variables = variables)
  descripcion_nivel <- paste0(ETIQUETAS_NIVELES[[nivel]], " (", length(variables), " variables)")

  if (id_algoritmo == "lpm") {
    return(correr_lpm(
      receta, folds, particion$entrenamiento, particion$test,
      especificacion = paste("LPM |", descripcion_nivel, "| umbral óptimo F1 OOF"),
      datos_holdout = particion$holdout, nivel = nivel, nombre = nombre,
      variables = variables
    ))
  }

  razon_pesos <- if (balanceo == "pesos") calcular_razon_pesos(particion$entrenamiento)
  especificacion_modelo <- especificar_algoritmo(id_algoritmo, razon_pesos = razon_pesos)

  # Pesos por fila: la columna va en el 80% (y sus folds) y en el holdout, que
  # se suma al 80% en el reajuste final. Nunca cambia qué filas hay en cada parte.
  columna_pesos <- NULL
  if (especificacion_modelo$mecanismo_pesos == "case_weights") {
    columna_pesos <- "peso_caso"
    con_pesos <- agregar_pesos_caso_particion(particion, folds, especificacion_modelo$peso_caso)
    particion <- con_pesos$particion
    folds <- con_pesos$folds
    receta <- crear_receta(particion$entrenamiento, balanceo = balanceo, variables = variables)
  }
  descripcion_balanceo <- if (balanceo == "pesos") {
    paste0("pesos (", especificacion_modelo$mecanismo_pesos, ")")
  } else {
    balanceo
  }

  correr_algoritmo(
    algoritmo = especificacion_modelo$algoritmo,
    modelo = especificacion_modelo$modelo,
    receta = receta,
    folds = folds,
    grilla = especificacion_modelo$grilla,
    datos_train = particion$entrenamiento,
    datos_test = particion$test,
    especificacion = paste(
      especificacion_modelo$algoritmo, paste0("(", id_algoritmo, ")"), "|",
      descripcion_nivel, "| balanceo:", descripcion_balanceo, "|", especificacion_modelo$busqueda
    ),
    datos_holdout = particion$holdout,
    nivel = nivel,
    nombre = nombre,
    balanceo = balanceo,
    variables = variables,
    busqueda = especificacion_modelo$busqueda,
    # SMOTE es caro: mejor estimar la receta una sola vez por fold
    parallel_over = if (balanceo == "smote") "resamples" else "everything",
    columna_pesos = columna_pesos,
    mecanismo_pesos = especificacion_modelo$mecanismo_pesos,
    pesos_aplicados = especificacion_modelo$pesos_aplicados
  )
}

# Agrega los pesos por fila al 80%, a sus folds y al holdout (ver
# agregar_pesos_caso()). Los folds conservan exactamente las mismas filas: solo
# se reemplaza la base a la que apuntan por la misma base con la columna nueva.
# Argumentos: particion, folds = salidas de cargar_particion(); razon_pesos.
# Devuelve: lista con particion y folds.
agregar_pesos_caso_particion <- function(particion, folds, razon_pesos) {
  particion$entrenamiento <- agregar_pesos_caso(particion$entrenamiento, razon_pesos)
  particion$holdout <- agregar_pesos_caso(particion$holdout, razon_pesos)
  folds$splits <- map(folds$splits, function(division) {
    division$data <- particion$entrenamiento
    division
  })
  list(particion = particion, folds = folds)
}

# Regla naive del nivel 1 (constante o de una sola variable).
# Argumentos: nombre; variables = vector de candidatas (o "todos_pobres" /
#   "todos_no_pobres"); particion, folds.
# Devuelve: salida de evaluar_candidato().
correr_regla_naive <- function(nombre, variables, particion, folds) {
  es_constante <- all(variables %in% c("todos_pobres", "todos_no_pobres"))
  especificacion <- if (es_constante) {
    paste("Regla constante:", variables)
  } else {
    paste("Regla de una variable (corte en percentil óptimo); candidatas:",
          paste(variables, collapse = ", "))
  }

  evaluar_candidato(
    nombre = nombre, nivel = "1", algoritmo = "Regla", especificacion = especificacion,
    particion = particion, folds = folds,
    ajustar_predecir = ajustar_predecir_regla,
    configuraciones = tibble(variable = variables),
    variables = if (es_constante) character() else variables,
    busqueda = if (es_constante) "sin hiperparámetros" else "variable y percentil de corte"
  )
}

# Stacking: meta-logit con coeficientes no negativos sobre las probabilidades
# OOF de los miembros (todas sobre los mismos folds). El meta-modelo también se
# cruza por folds para que el umbral se elija con predicciones fuera de muestra.
# Para holdout y test usa las predicciones que cada miembro ya hizo con su
# reajuste sobre el 80% y el 100%, respectivamente.
# Argumentos: nombre; miembros = lista de salidas de evaluar_candidato(); particion.
# Devuelve: lista con la misma estructura que evaluar_candidato().
correr_stacking <- function(nombre, miembros, particion) {
  ruta_resultado <- file.path(RUTA_MODELOS, paste0(nombre, ".rds"))
  if (file.exists(ruta_resultado)) {
    message("[", nombre, "] cargado desde ", basename(ruta_resultado))
    registrar_log(nombre, "cargado de caché")
    return(readRDS(ruta_resultado))
  }

  inicio <- Sys.time()
  registrar_log(nombre, "inicio", reiniciar_pico = TRUE)
  nombres_miembros <- map_chr(miembros, "nombre")
  message("\n[", nombre, "] stacking de: ", paste(nombres_miembros, collapse = ", "))

  # Une las predicciones de los miembros por fila (.row), fold o id
  unir_predicciones <- function(elemento, llaves) {
    map(miembros, function(miembro) {
      miembro[[elemento]] |>
        select(all_of(llaves), probabilidad) |>
        rename(!!miembro$nombre := probabilidad)
    }) |>
      reduce(inner_join, by = llaves)
  }
  base_oof <- unir_predicciones("oof", c(".row", "fold", "verdad"))
  stopifnot("Los miembros no comparten los mismos folds" =
              nrow(base_oof) == nrow(miembros[[1]]$oof))

  a_logit <- function(datos) {
    matriz <- as.matrix(datos[nombres_miembros])
    qlogis(pmin(pmax(matriz, 1e-4), 1 - 1e-4))
  }
  ajustar_meta <- function(datos) {
    glmnet::glmnet(a_logit(datos), datos$verdad, family = "binomial",
                   alpha = 0, lambda = 1e-3, lower.limits = 0)
  }
  predecir_meta <- function(meta, datos) {
    as.numeric(predict(meta, a_logit(datos), type = "response"))
  }

  # OOF del meta-modelo: se entrena en k-1 folds y predice el fold restante
  oof <- map_dfr(unique(base_oof$fold), function(f) {
    evaluacion <- base_oof |> filter(fold == f)
    tibble(
      .row = evaluacion$.row, fold = f, verdad = evaluacion$verdad,
      probabilidad = predecir_meta(ajustar_meta(filter(base_oof, fold != f)), evaluacion)
    )
  })

  busqueda_umbral <- buscar_umbral_f1(oof$verdad, oof$probabilidad)
  umbral <- busqueda_umbral$umbral
  metricas_cv <- metricas_por_fold(oof, umbral)

  meta <- ajustar_meta(base_oof)
  pesos <- coef(meta)[-1, 1]

  base_holdout <- unir_predicciones("prediccion_holdout", "id")
  base_holdout <- base_holdout[match(particion$holdout$id, base_holdout$id), ]
  probabilidad_holdout <- predecir_meta(meta, base_holdout)
  clases_holdout <- as.integer(probabilidad_holdout >= umbral)
  metricas_holdout <- calcular_metricas(
    as.integer(particion$holdout$pobre == "pobre"), clases_holdout
  )

  base_test <- unir_predicciones("prediccion_test", "id")
  base_test <- base_test[match(particion$test$id, base_test$id), ]
  probabilidad_test <- predecir_meta(meta, base_test)
  clases_test <- as.integer(probabilidad_test >= umbral)

  resultado <- list(
    nombre = nombre,
    nivel = "6",
    algoritmo = "Stacking",
    especificacion = paste(
      "Stacking (meta-logit, pesos >= 0) de:", paste(nombres_miembros, collapse = ", ")
    ),
    balanceo = "ninguno",
    variables = unique(unlist(map(miembros, "variables"))),
    busqueda = "pesos del meta-logit estimados sobre OOF",
    n_configuraciones = 1L,
    hiperparametros = as_tibble(as.list(round(pesos, 3))),
    umbral = umbral,
    curva_umbral = busqueda_umbral$curva,
    seleccion = tibble(),
    metricas_cv = metricas_cv,
    metricas_holdout = metricas_holdout,
    oof = oof,
    prediccion_holdout = tibble(
      id = particion$holdout$id, probabilidad = probabilidad_holdout, clase = clases_holdout
    ),
    prediccion_test = tibble(
      id = particion$test$id, probabilidad = probabilidad_test, clase = clases_test
    ),
    importancia = tibble(variable = names(pesos), importancia = pesos / max(pesos)),
    archivo_submission = paste0(nombre, ".csv"),
    tiempo_min = as.numeric(difftime(Sys.time(), inicio, units = "mins"))
  )

  guardar_submission(
    ids = particion$test$id, clases = clases_test, algoritmo = "Stacking",
    especificacion = resultado$especificacion, f1_cv = metricas_cv$f1_cv,
    carpeta = file.path(RUTA_SUBMISSIONS, "candidatos"),
    nombre_archivo = resultado$archivo_submission, nivel = "6",
    f1_holdout = metricas_holdout$f1, registrar = FALSE
  )
  saveRDS(meta, file.path(RUTA_MODELOS, paste0(nombre, "_ajuste.rds")))
  saveRDS(resultado, ruta_resultado)

  message(sprintf(
    "[%s] F1 CV = %.4f (sd %.4f) | F1 holdout = %.4f | umbral = %.2f",
    nombre, metricas_cv$f1_cv, metricas_cv$f1_cv_sd, metricas_holdout$f1, umbral
  ))
  registrar_log(nombre, "fin", inicio = inicio)
  resultado
}

# Carga todos los candidatos evaluados (un .rds por candidato en stores/modelos).
# Argumento opcional: patron = regex sobre el nombre del candidato.
# Devuelve: lista nombrada de resultados.
cargar_resultados_escalera <- function(patron = ".") {
  archivos <- list.files(RUTA_MODELOS, pattern = "\\.rds$", full.names = TRUE)
  archivos <- archivos[!str_detect(basename(archivos), "_ajuste\\.rds$")]
  archivos <- archivos[str_detect(basename(archivos), patron)]
  resultados <- map(archivos, readRDS)
  set_names(resultados, map_chr(resultados, "nombre"))
}

# Mejor candidato según F1 de CV (las decisiones dentro de la escalera nunca
# miran el holdout). Argumentos: resultados = lista de candidatos.
mejor_por_cv <- function(resultados) {
  f1 <- map_dbl(resultados, ~ .x$metricas_cv$f1_cv)
  resultados[[which.max(f1)]]
}

# Una fila del leaderboard por candidato.
# Argumentos: resultado = salida de evaluar_candidato() o correr_stacking().
# Devuelve: tibble de una fila.
fila_leaderboard <- function(resultado) {
  cv <- resultado$metricas_cv
  holdout <- resultado$metricas_holdout

  tibble(
    nivel = resultado$nivel,
    etiqueta_nivel = ETIQUETAS_NIVELES[[resultado$nivel]],
    candidato = resultado$nombre,
    algoritmo = resultado$algoritmo,
    especificacion = resultado$especificacion,
    n_variables = length(resultado$variables),
    variables = paste(resultado$variables, collapse = ", "),
    balanceo = resultado$balanceo,
    mecanismo_pesos = resultado$mecanismo_pesos %||% "sin_pesos",
    # FALSE: se pidieron pesos de clase pero el motor no los aplica
    pesos_aplicados = resultado$pesos_aplicados %||% NA,
    busqueda = resultado$busqueda,
    n_configuraciones = resultado$n_configuraciones,
    hiperparametros = describir_hiperparametros(resultado$hiperparametros),
    umbral = resultado$umbral,
    f1_cv = cv$f1_cv,
    f1_cv_sd = cv$f1_cv_sd,
    precision_cv = cv$precision_cv,
    recall_cv = cv$recall_cv,
    accuracy_cv = cv$accuracy_cv,
    f1_holdout = holdout$f1,
    precision_holdout = holdout$precision,
    recall_holdout = holdout$recall,
    accuracy_holdout = holdout$accuracy,
    filtracion_pct = holdout$filtracion,
    filtracion_n = holdout$n_fp,
    subcobertura_pct = holdout$subcobertura,
    subcobertura_n = holdout$n_fn,
    pobres_holdout = holdout$n_vp + holdout$n_fn,
    predichos_pobres_holdout = holdout$n_vp + holdout$n_fp,
    brecha_cv_holdout = cv$f1_cv - holdout$f1,
    sobreajuste = (cv$f1_cv - holdout$f1) > UMBRAL_SOBREAJUSTE,
    tiempo_min = resultado$tiempo_min,
    archivo_submission = resultado$archivo_submission
  )
}

# Leaderboard interno ordenado por F1 en holdout.
construir_leaderboard <- function(resultados) {
  map_dfr(resultados, fila_leaderboard) |>
    arrange(desc(f1_holdout))
}

# Elige los n mejores candidatos para Kaggle recorriendo el ranking por F1 en
# holdout, con un solo representante por algoritmo (el stacking cuenta como
# familia propia): XGBoost con y sin balanceo, o XGBoost vs. LightGBM, son
# variaciones del mismo Boosting, y subir tres de ellas no enseña nada en Kaggle.
# Para cada salto se reporta qué tanto coincide con el elegido en el holdout.
# Las reglas naive no entran, ni un candidato con pesos de clase que su motor
# no aplica (pesos_aplicados = FALSE).
# Argumentos: leaderboard; resultados = lista de candidatos; n = cuántos elegir.
# Devuelve: lista con top (posición, candidato) y saltados (con su concordancia).
elegir_top3 <- function(leaderboard, resultados, n = 3) {
  ranking <- leaderboard |> filter(algoritmo != "Regla", !(pesos_aplicados %in% FALSE))
  elegidos <- character()
  saltados <- tibble(candidato = character(), parecido_a = character(), concordancia = numeric())

  for (candidato in ranking$candidato) {
    if (length(elegidos) == n) break
    algoritmo <- resultados[[candidato]]$algoritmo
    mismo <- purrr::keep(elegidos, ~ resultados[[.x]]$algoritmo == algoritmo)

    if (length(mismo) == 0) {
      elegidos <- c(elegidos, candidato)
    } else {
      concordancia <- mean(resultados[[candidato]]$prediccion_holdout$clase ==
                             resultados[[mismo[1]]]$prediccion_holdout$clase)
      saltados <- bind_rows(
        saltados,
        tibble(candidato = candidato, parecido_a = mismo[1], concordancia = concordancia)
      )
    }
  }

  list(
    top = tibble(posicion = seq_along(elegidos), candidato = elegidos),
    saltados = saltados
  )
}

# Gráfico de la escalera: F1 en holdout de cada candidato por nivel, con el mejor
# de cada nivel resaltado y rotulado, y el piso de "todos pobres" como referencia.
# Argumentos: leaderboard = salida de construir_leaderboard().
# Devuelve: objeto ggplot.
graficar_escalera <- function(leaderboard) {
  datos <- leaderboard |>
    filter(candidato != "n1_todos_no_pobres") |>
    mutate(etiqueta_nivel = factor(etiqueta_nivel, levels = ETIQUETAS_NIVELES)) |>
    group_by(etiqueta_nivel) |>
    mutate(es_mejor = f1_holdout == max(f1_holdout)) |>
    ungroup() |>
    mutate(rotulo = if_else(
      es_mejor,
      paste0(str_remove(candidato, "^n[0-9E]_"), "\n", formatC(f1_holdout, format = "f", digits = 3)),
      NA_character_
    ))

  piso <- leaderboard |> filter(candidato == "n1_todos_pobres") |> pull(f1_holdout)

  ggplot(datos, aes(etiqueta_nivel, f1_holdout)) +
    geom_hline(yintercept = piso, linetype = "dashed", color = "#52514e", linewidth = 0.4) +
    annotate("text", x = levels(datos$etiqueta_nivel)[1], y = piso,
             label = "Piso: clasificar a todos como pobres",
             hjust = 0.25, vjust = 1.6, size = 3.3, color = "#52514e") +
    geom_point(data = filter(datos, !es_mejor), color = "#a8a7a0", size = 2.6) +
    geom_point(data = filter(datos, es_mejor), color = "#2a78d6", size = 3.6) +
    geom_text(aes(label = rotulo), vjust = -0.6, size = 3.2, lineheight = 0.9,
              color = "#0b0b0b", na.rm = TRUE) +
    scale_y_continuous(limits = c(min(datos$f1_holdout, piso) - 0.02,
                                  max(datos$f1_holdout) + 0.04),
                       breaks = seq(0, 1, 0.05)) +
    labs(
      title = "F1 fuera de muestra a lo largo de la escalera de modelos",
      subtitle = "F1 en un holdout estratificado del 20% (nunca usado para elegir hiperparámetros ni umbral). Azul: mejor de cada nivel",
      x = NULL, y = "F1 en holdout"
    ) +
    theme(panel.grid.major.x = element_blank())
}

# Escribe la bitácora de Kaggle con una fila por CSV de candidato de R. Conserva
# todo lo que no genera este script: las columnas anotadas a mano (f1_kaggle,
# orden_subida, ...), la fecha en que cada archivo entró a la bitácora y las
# filas de otros orígenes (p. ej. el nivel 7 en Python, archivos nivel7/*.csv).
# Argumentos: leaderboard = candidatos de R; top = salida $top de elegir_top3().
# Devuelve: la bitácora (invisible).
actualizar_bitacora <- function(leaderboard, top) {
  ruta_bitacora <- file.path(RUTA_SUBMISSIONS, "bitacora_kaggle.csv")
  previa <- if (file.exists(ruta_bitacora)) {
    read_csv(ruta_bitacora, show_col_types = FALSE, col_types = cols(.default = "c"))
  } else {
    tibble(archivo = character())
  }

  bitacora <- leaderboard |>
    left_join(top, by = "candidato") |>
    transmute(
      archivo = file.path("candidatos", archivo_submission),
      nivel,
      candidato,
      algoritmo,
      especificacion,
      f1_cv = round(f1_cv, 4),
      f1_holdout = round(f1_holdout, 4),
      top3 = posicion
    ) |>
    mutate(across(everything(), as.character))
  columnas_generadas <- names(bitacora)

  # Fecha de entrada y anotaciones manuales de los archivos que ya estaban
  anotado <- previa |> select(archivo, !any_of(columnas_generadas))
  bitacora <- bitacora |>
    left_join(anotado, by = "archivo") |>
    mutate(fecha = coalesce(if ("fecha" %in% names(anotado)) fecha else NA_character_,
                            as.character(Sys.Date())))
  if (!"f1_kaggle" %in% names(bitacora)) bitacora$f1_kaggle <- NA_character_

  bitacora <- bind_rows(bitacora, previa |> filter(!archivo %in% bitacora$archivo)) |>
    relocate(fecha) |>
    relocate(any_of(names(previa)))

  write_csv(bitacora, ruta_bitacora, na = "")
  invisible(bitacora)
}
