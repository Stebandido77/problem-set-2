# ==============================================================================
# 05_nivel_4_algoritmos.R
# Escalera, nivel 4: cinco algoritmos con las variables del nivel 3 y la misma
# receta (cancha nivelada). Cada uno con búsqueda sistemática de hiperparámetros
# por CV; los rangos están documentados en especificar_algoritmo().
# ==============================================================================

source(here::here("scripts", "00_configuracion.R"))

entrada <- cargar_particion()
particion <- entrada$particion
folds <- entrada$folds
niveles <- definir_niveles()

algoritmos_nivel_4 <- c("elastic_net", "cart", "naive_bayes", "xgboost", "random_forest")

walk(algoritmos_nivel_4, function(id_algoritmo) {
  correr_candidato_escalera(
    id_algoritmo, paste0("n4_", id_algoritmo), "4", niveles$nivel_3, particion, folds
  )
})

plan(sequential)
