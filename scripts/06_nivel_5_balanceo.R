# ==============================================================================
# 06_nivel_5_balanceo.R
# Escalera, nivel 5: ¿manejar el desbalance de clases cambia el resultado?
# Toma el mejor algoritmo del nivel 4 (por F1 de CV: el holdout no se usa para
# decidir) y lo re-estima con SMOTE, submuestreo y pesos de clase, con la misma
# grilla. "Sin balanceo" es el propio candidato del nivel 4.
# El balanceo vive dentro de la receta (o del motor, para los pesos): solo
# afecta el entrenamiento de cada fold, nunca la validación ni el holdout.
# ==============================================================================

source(here::here("scripts", "00_configuracion.R"))

entrada <- cargar_particion()
particion <- entrada$particion
folds <- entrada$folds
niveles <- definir_niveles()

ganador_nivel_4 <- mejor_por_cv(cargar_resultados_escalera("^n4_"))
id_ganador <- str_remove(ganador_nivel_4$nombre, "^n4_")
message("Mejor algoritmo del nivel 4 (F1 CV): ", id_ganador)

walk(c("smote", "submuestreo", "pesos"), function(estrategia) {
  correr_candidato_escalera(
    id_ganador, paste0("n5_", id_ganador, "_", estrategia), "5", niveles$nivel_3,
    particion, folds, balanceo = estrategia
  )
})

plan(sequential)
