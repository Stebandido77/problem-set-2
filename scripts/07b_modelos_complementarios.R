# ============================================================================== 
# 07b_modelos_complementarios.R
# Ejecución de candidatos de alta capacidad predictiva utilizando la infraestructura
# nativa de la escalera de modelos (correr_candidato_escalera).
# ============================================================================== 

source(here::here("scripts", "00_configuracion.R"))

# Desactivar workers paralelos de 'future' para evitar duplicación masiva de RAM
plan(sequential)

# 1. Cargar partición y folds oficiales del proyecto
entrada   <- cargar_particion()
particion <- entrada$particion
folds     <- entrada$folds

# 2. Definir las variables del Nivel 4+
variables <- definir_niveles()$nivel_4v

# Verificar que todas las variables existan en el dataset de entrenamiento
faltantes <- setdiff(variables, names(particion$entrenamiento))
if (length(faltantes) > 0) {
  stop("Faltan variables del nivel 4+: ", paste(faltantes, collapse = ", "))
}

# 3. Lista de algoritmos a evaluar con configuración económica de RAM
algoritmos <- c("elastic_net", "random_forest", "xgboost")

message(">>> Iniciando ajuste de modelos balanceados en Nivel 4+...")

for (id_alg in algoritmos) { # Corregido a 'in'
  message("\n--------------------------------------------------")
  message("Procesando candidato: ", id_alg)
  message("--------------------------------------------------")
  
  tryCatch({
    correr_candidato_escalera(
      id_algoritmo = id_alg,
      nombre       = paste0("n4v_", id_alg),
      nivel        = "4+",
      variables    = variables,
      particion    = particion,
      folds        = folds,
      economico    = TRUE
    )
  }, error = function(e) {
    message("   [Error en ", id_alg, "]: ", e$message)
  })
  
  # Liberar memoria RAM tras finalizar cada modelo
  gc(verbose = FALSE)
}

message("\n>>> ¡Modelos procesados exitosamente! Archivos generados en stores/submissions/candidatos/")