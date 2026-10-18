# ==============================================================================
# 04_niveles_1_2_3.R
# Escalera, niveles 1 a 3:
#   1. Naive: todos pobres, todos no pobres y la mejor regla de una variable.
#   2. Base: Logit y LPM con pocos predictores teóricos.
#   3. Base + ingeniería de variables desde personas: el mismo Logit y LPM, para
#      aislar el aporte de las variables nuevas.
# ==============================================================================

source(here::here("scripts", "00_configuracion.R"))

entrada <- cargar_particion()
particion <- entrada$particion
folds <- entrada$folds
niveles <- definir_niveles()

# ---- Nivel 1: naive ----------------------------------------------------------
# Fijan el piso del F1 y muestran lo engañoso del accuracy con 20% de pobres
correr_regla_naive("n1_todos_pobres", "todos_pobres", particion, folds)
correr_regla_naive("n1_todos_no_pobres", "todos_no_pobres", particion, folds)
correr_regla_naive("n1_una_variable", niveles$nivel_1, particion, folds)

# ---- Nivel 2: base teórica ---------------------------------------------------
correr_candidato_escalera("logit", "n2_logit", "2", niveles$nivel_2, particion, folds)
correr_candidato_escalera("lpm", "n2_lpm", "2", niveles$nivel_2, particion, folds)

# ---- Nivel 3: base + ingeniería de variables ---------------------------------
correr_candidato_escalera("logit", "n3_logit", "3", niveles$nivel_3, particion, folds)
correr_candidato_escalera("lpm", "n3_lpm", "3", niveles$nivel_3, particion, folds)

plan(sequential)
