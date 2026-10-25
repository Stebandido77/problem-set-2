# ==============================================================================
# funciones/modelado.R
# Receta común, validación cruzada, tuning y predicción. Todos los algoritmos
# pasan por aquí para garantizar una cancha nivelada (mismas variables y mismo
# preprocesamiento).
# ==============================================================================

# Receta de preprocesamiento compartida por todos los algoritmos.
# Argumentos:
#   datos    = base de entrenamiento
#   balanceo = "ninguno", "smote" o "submuestreo" (manejo del desbalance de clases)
# Devuelve: objeto recipe sin estimar.
crear_receta <- function(datos, balanceo = c("ninguno", "smote", "submuestreo")) {
  balanceo <- match.arg(balanceo)

  receta <- recipe(pobre ~ ., data = datos) |>
    update_role(id, new_role = "id") |>
    step_impute_median(all_numeric_predictors()) |>
    step_impute_mode(all_nominal_predictors()) |>
    step_novel(all_nominal_predictors()) |>
    step_dummy(all_nominal_predictors()) |>
    step_zv(all_predictors()) |>
    step_normalize(all_numeric_predictors())

  # El balanceo solo se aplica en entrenamiento (skip = TRUE por defecto en themis)
  receta <- switch(
    balanceo,
    ninguno     = receta,
    smote       = receta |> step_smote(pobre, seed = SEMILLA),
    submuestreo = receta |> step_downsample(pobre, seed = SEMILLA)
  )

  receta
}

# Particiones de validación cruzada estratificadas por la variable objetivo.
crear_folds <- function(datos, v = N_FOLDS) {
  set.seed(SEMILLA)
  vfold_cv(datos, v = v, strata = pobre)
}

# Conjunto de métricas que se calcula durante el tuning.
metricas_tuning <- metric_set(f_meas, precision, recall, roc_auc)

# Corre el flujo completo de un algoritmo: tuning por CV, umbral óptimo de F1,
# ajuste final, predicción en test y submission a Kaggle.
# Argumentos:
#   algoritmo      = nombre del algoritmo (ver guardar_submission())
#   modelo         = especificación parsnip (con tune() donde aplique)
#   receta         = salida de crear_receta()
#   folds          = salida de crear_folds()
#   grilla         = grilla de hiperparámetros (NULL si no hay nada que tunear)
#   datos_train    = base de entrenamiento
#   datos_test     = base de test
#   especificacion = descripción corta para la bitácora
# Devuelve: lista con mejores hiperparámetros, umbral, métricas CV y modelo final.
correr_algoritmo <- function(algoritmo, modelo, receta, folds, grilla,
                             datos_train, datos_test, especificacion) {
  flujo <- workflow() |> add_recipe(receta) |> add_model(modelo)
  control <- control_grid(save_pred = TRUE, parallel_over = "everything")

  set.seed(SEMILLA)
  resultados_cv <- if (is.null(grilla)) {
    fit_resamples(flujo, folds, metrics = metricas_tuning,
                  control = control_resamples(save_pred = TRUE))
  } else {
    tune_grid(flujo, folds, grid = grilla, metrics = metricas_tuning, control = control)
  }

  mejores <- if (is.null(grilla)) tibble() else select_best(resultados_cv, metric = "f_meas")

  # Predicciones fuera de muestra del mejor modelo -> umbral que maximiza F1
  predicciones_oof <- collect_predictions(resultados_cv, parameters = if (nrow(mejores)) mejores)
  busqueda <- buscar_umbral_f1(
    verdad = as.integer(predicciones_oof$pobre == "pobre"),
    probabilidad = predicciones_oof$.pred_pobre
  )
  metricas_cv <- busqueda$curva |> filter(umbral == busqueda$umbral)

  # Ajuste final con toda la base de entrenamiento
  flujo_final <- if (nrow(mejores)) finalize_workflow(flujo, mejores) else flujo
  modelo_final <- fit(flujo_final, datos_train)

  probabilidad_test <- predict(modelo_final, datos_test, type = "prob")$.pred_pobre
  guardar_submission(
    ids = datos_test$id,
    clases = as.integer(probabilidad_test >= busqueda$umbral),
    algoritmo = algoritmo,
    especificacion = especificacion,
    f1_cv = metricas_cv$f1
  )

  resultado <- list(
    algoritmo = algoritmo,
    especificacion = especificacion,
    mejores_hiperparametros = mejores,
    umbral = busqueda$umbral,
    metricas_cv = metricas_cv,
    curva_umbral = busqueda$curva,
    modelo_final = modelo_final
  )

  guardar_resultado(resultado)
  resultado
}

# Modelo lineal de probabilidad. Va aparte porque parsnip no lo trata como
# clasificador: se estima con lm() sobre la receta horneada.
# Mismos argumentos que correr_algoritmo(), sin modelo ni grilla.
correr_lpm <- function(receta, folds, datos_train, datos_test, especificacion) {
  ajustar_y_predecir <- function(entrenamiento, evaluacion) {
    receta_estimada <- prep(receta, training = entrenamiento)
    base_train <- bake(receta_estimada, new_data = NULL) |>
      mutate(pobre = as.integer(pobre == "pobre"))
    base_eval <- bake(receta_estimada, new_data = evaluacion)

    modelo <- lm(pobre ~ ., data = select(base_train, -id))
    preds <- predict(modelo, newdata = select(base_eval, -id))
    preds <- if_else(is.na(preds), 0, preds)

    list(
      modelo = modelo,
      probabilidad = pmin(pmax(preds, 0), 1)
    )
  }

  predicciones_oof <- map_dfr(folds$splits, function(particion) {
    evaluacion <- assessment(particion)
    tibble(
      verdad = as.integer(evaluacion$pobre == "pobre"),
      probabilidad = ajustar_y_predecir(analysis(particion), evaluacion)$probabilidad
    )
  })

  busqueda <- buscar_umbral_f1(predicciones_oof$verdad, predicciones_oof$probabilidad)
  metricas_cv <- busqueda$curva |> filter(umbral == busqueda$umbral)

  final <- ajustar_y_predecir(datos_train, datos_test)
  guardar_submission(
    ids = datos_test$id,
    clases = as.integer(final$probabilidad >= busqueda$umbral),
    algoritmo = "LPM",
    especificacion = especificacion,
    f1_cv = metricas_cv$f1
  )

  resultado <- list(
    algoritmo = "LPM",
    especificacion = especificacion,
    mejores_hiperparametros = tibble(),
    umbral = busqueda$umbral,
    metricas_cv = metricas_cv,
    curva_umbral = busqueda$curva,
    modelo_final = final$modelo
  )

  guardar_resultado(resultado)
  resultado
}

# Guarda el objeto completo (.rds, no se sube) y una fila de métricas (.csv, sí se sube).
guardar_resultado <- function(resultado) {
  marca <- format(Sys.time(), "%Y%m%d_%H%M%S")
  base_nombre <- paste0(str_replace_all(str_to_lower(resultado$algoritmo), " ", "_"), "_", marca)

  saveRDS(resultado, file.path(RUTA_RESULTADOS, paste0(base_nombre, ".rds")))

  ruta_metricas <- file.path(RUTA_RESULTADOS, "metricas_cv.csv")
  fila <- resultado$metricas_cv |>
    mutate(
      algoritmo = resultado$algoritmo,
      especificacion = resultado$especificacion,
      archivo_rds = paste0(base_nombre, ".rds"),
      .before = 1
    )
  write_csv(fila, ruta_metricas, append = file.exists(ruta_metricas))
}

# Carga los datos procesados que produce 01_preparar_datos.R.
cargar_bases_modelado <- function() {
  list(
    train = readRDS(file.path(RUTA_PROCESADOS, "train_hogar.rds")),
    test  = readRDS(file.path(RUTA_PROCESADOS, "test_hogar.rds"))
  )
}
