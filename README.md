# Problem Set 2: Predicting Poverty

> *"Wars of nations are fought to change maps. But wars of poverty are fought to map change."* — M. Ali

**Big Data and Machine Learning para Economía Aplicada (MECA 4107)** · Universidad de los Andes · 2026-20 · **Equipo 06**

Predecimos la pobreza de los hogares colombianos con datos de la MESE (DANE) como un problema de
clasificación, y nos preguntamos qué modelo le serviría de verdad a un hacedor de política:
uno que prediga bien **y** que se pueda aplicar con una encuesta corta y barata.

## Integrantes

| Nombre | GitHub |
|---|---|
| Esteban | [@Stebandido77](https://github.com/Stebandido77) |
| _Por completar_ | |
| _Por completar_ | |

## Estructura del repositorio

```
.
├── document/
│   ├── enunciado_problem_set_2.pdf   # Enunciado del profesor
│   └── slides/                       # best_equipo_06.pdf y compare_equipo_06.pdf
├── scripts/
│   ├── 00_configuracion.R            # Paquetes, rutas, semilla, MODO_RAPIDO
│   ├── 01_preparar_datos.R           # Limpieza y unión hogares-personas
│   ├── 02_exploracion.R              # Desbalance y relaciones con predictores
│   ├── 03_particion.R                # Holdout estratificado 20% + folds de CV
│   ├── 04_niveles_1_2_3.R            # Escalera: naive, base teórica, base + variables
│   ├── 05_nivel_4_algoritmos.R       # Elastic Net, CART, Naive Bayes, XGBoost, RF
│   ├── 06_nivel_5_balanceo.R         # SMOTE, submuestreo y pesos sobre el mejor
│   ├── 07_nivel_6_complejo.R         # LightGBM, stacking y modelo de encuesta corta
│   ├── 08_leaderboard.R              # Leaderboard interno, escalera, top 3, bitácora
│   ├── 09_comparacion.R              # Deck compare: algoritmos en cancha nivelada
│   ├── 10_mejor_modelo.R             # Deck best: importancia, trade-off de política
│   ├── 11_exportar_python.R          # Nivel 7: bases, partición y folds a parquet
│   ├── 12_diagnostico_nivel7.R       # Nivel 7: variables disponibles y errores de n6
│   ├── main.R                        # Corre todo en orden
│   └── funciones/                    # Funciones reutilizables
│       ├── limpieza.R                # Variables de hogar/personas y niveles
│       ├── modelado.R                # Receta, algoritmos, evaluar_candidato()
│       ├── escalera.R                # Reglas naive, stacking, leaderboard, top 3
│       └── utilidades.R              # Métricas, figuras, validación de submissions
├── python/                           # Nivel 7 (Python 3.12, ver sección propia)
│   ├── requirements.txt              # Versiones exactas del entorno
│   ├── configuracion.py              # Semilla, rutas, parámetros de n6_lightgbm
│   ├── etapa1_ablacion.py ... etapa4_leaderboard.py
│   ├── funciones/                    # Variables v2, OOF anidada, auxiliares, métricas
│   └── tests/                        # Partición igual a R y auxiliares sin fuga
├── stores/
│   ├── raw/                          # CSV de Kaggle (no se suben)
│   ├── procesados/                   # Bases limpias y partición (no se suben)
│   ├── modelos/                      # Un .rds por candidato (no se suben)
│   ├── logs/corrida.log              # Hora y RAM por candidato y etapa (no se sube)
│   ├── resultados/                   # leaderboard_interno.csv y leaderboard_nivel7.csv
│   └── submissions/
│       ├── candidatos/               # Un CSV por candidato de la escalera
│       ├── top3/                     # Los 3 elegidos para subir primero
│       ├── nivel7/                   # CSV finales del nivel 7 (validados)
│       └── bitacora_kaggle.csv       # Llenar f1_kaggle a mano al subir
├── tests/
│   └── test_pesos_clase.R            # Convención de pesos de clase de cada motor
└── views/
    ├── figuras/                      # Figuras en PDF y PNG para las slides
    └── tablas/                       # Tablas en CSV y LaTeX para las slides
```

## Cómo reproducir

1. Clonar el repo y abrir `problem-set-2.Rproj` en RStudio.
2. Descargar de Kaggle `train_hogares.csv`, `train_personas.csv`, `test_hogares.csv` y
   `test_personas.csv`, y ponerlos en `stores/raw/`.
3. Correr `scripts/main.R`. Los paquetes se instalan solos con `pacman`.

Para validar que todo corre antes de la corrida larga (≈ 20 min en vez de horas):
`Sys.setenv(PS2_MODO_RAPIDO = "TRUE")` y luego `source("scripts/main.R")`. El modo rápido
tunea con una submuestra del 15% y grillas pequeñas, y escribe todo en `stores/rapido/`.

Corrida completa desde la terminal, en la raíz del repo y con la salida en un log:

```bash
Rscript -e 'testthat::test_file("tests/test_pesos_clase.R", stop_on_failure = TRUE)'
Rscript -e 'options(warn = 1); source("scripts/main.R")' > stores/logs/consola.log 2>&1
```

**Memoria y paralelismo.** Random Forest, XGBoost y LightGBM corren los folds en secuencia
y paralelizan con hilos del motor (`N_NUCLEOS`, por defecto todos los núcleos menos uno),
que comparten una sola copia de los datos. Logit, Elastic Net, CART y Naive Bayes usan
workers de `future`, cada uno con su propia copia, limitados a `N_WORKERS_MAX` (4). Ambos
se cambian con las variables de entorno `PS2_NUCLEOS` y `PS2_WORKERS`. Al terminar cada
candidato se cierran los workers, el modelo final se guarda adelgazado con `butcher` y se
escribe una línea en `stores/logs/corrida.log` con la hora y la RAM de la sesión, de los
workers y del sistema.

Cada candidato de la escalera escribe su CSV en `stores/submissions/candidatos/` y queda
guardado en `stores/modelos/`; si la corrida se cae, al volver a correr se cargan los ya
terminados. `08_leaderboard.R` arma `stores/resultados/leaderboard_interno.csv` (le une las
filas del nivel 7 desde `leaderboard_nivel7.csv`), copia los tres elegidos a
`stores/submissions/top3/` (los de `TOP3_MANUAL` en `00_configuracion.R`: n6_lightgbm,
n4_random_forest y n3_logit, este último por interpretabilidad; con `NULL` se usa la regla
automática de un representante por algoritmo) y actualiza `bitacora_kaggle.csv`: conserva lo anotado a mano
(`f1_kaggle`, `orden_subida`) y las filas del nivel 7. Después de subir a Kaggle, anoten el
F1 público a mano. El mejor modelo se elige siempre por F1 en holdout, nunca por el público.

## Escalera de modelos y evaluación interna

| Nivel | Variables | Candidatos |
|---|---|---|
| 1. Naive | ninguna / una | todos pobres, todos no pobres, mejor regla de una variable |
| 2. Base | tamaño del hogar, educación, ocupación y posición del jefe | Logit, LPM |
| 3. Base + variables | + ~45 variables construidas desde personas, vivienda y ubicación | Logit, LPM |
| 4. Algoritmos | las del nivel 3 | Elastic Net, CART, Naive Bayes, XGBoost, Random Forest |
| 5. Desbalance | las del nivel 3 | el mejor del nivel 4 con SMOTE, submuestreo y pesos de clase |
| 6. Complejo | las del nivel 3 | LightGBM (búsqueda amplia) y stacking |
| E. Encuesta corta | 10 variables baratas | el mejor del nivel 4 y un Logit |
| 7. Python | nivel 3 + v2 + ingreso predicho | LightGBM con auxiliares de ingreso (ver abajo) |

1. Holdout estratificado del 20% que no se usa para ninguna decisión.
2. CV de 5 folds sobre el 80% restante: hiperparámetros elegidos por F1 fuera de fold,
   cada configuración en su propio umbral óptimo.
3. Umbral elegido con predicciones fuera de fold y aplicado fijo al holdout.
4. Ranking final por F1 en holdout; se marca sobreajuste si F1 CV − F1 holdout > 0,02.
5. Cada candidato se reajusta sobre el 100% del train antes de predecir el test.

## Nivel 7 (Python): variables v2 y auxiliares de ingreso

El nivel 7 vive en `python/` y usa **exactamente los mismos datos, folds y holdout que R**:
`11_exportar_python.R` los exporta a `stores/procesados/*.parquet` y Python no rehace la
partición (`python/tests/test_particion.py` verifica conteos y tasa de pobreza por fold).
El protocolo es el mismo de la escalera: umbral elegido con OOF, F1 en holdout y reajuste
sobre el 100% del train para predecir el test.

- **Variables v2**: actividad y estabilidad laboral, régimen contributivo, tamaño de empresa
  y posición de todos los ocupados, horas, ingresos sí/no, amortización y arriendo frente a
  la línea de pobreza, perceptor principal y tipologías de hogares pequeños.
- **Auxiliares de ingreso**: un LightGBM predice el ingreso laboral y otro el ingreso total
  de cada persona con variables de personas presentes en test; la predicción se agrega al
  hogar (per cápita sobre la unidad de gasto y frente a Lp). Se calculan con **OOF
  anidada**: ningún hogar recibe una predicción de un modelo entrenado con él, y en cada
  fold ningún auxiliar vio el fold de validación (`python/tests/test_sin_fuga.py`).
- Encuesta corta: frontera F1 vs. número de variables con NSGA-II y Logit (sin auxiliares,
  porque una encuesta corta no puede usar un modelo entrenado con la encuesta larga).

| Paso | F1 holdout | Aporte |
|---|---|---|
| n6_lightgbm (nivel 6, R) | 0,749 | |
| + variables laborales y de composición | 0,760 | +0,012 |
| + auxiliar de ingreso laboral (`checkpoint_a4`, Kaggle 0,77) | 0,771 | +0,011 |
| + auxiliar de ingreso total | 0,778 | +0,006 |
| Optuna (75 trials) / seed bagging de 5 semillas / umbral sin Bogotá | 0,776 | 0 |
| **Final: `n7_final.csv`** (parámetros de n6, bagging de 5 semillas) | **0,776** | **+0,027** |

Entorno y corrida (en la raíz del repo, después de `main.R` o al menos de
`01`, `03` y `11`):

```bash
py -3.12 -m venv python/.venv
python/.venv/Scripts/python.exe -m pip install -r python/requirements.txt
python/.venv/Scripts/python.exe -m pytest python/tests
python/.venv/Scripts/python.exe python/etapa1_ablacion.py 90
python/.venv/Scripts/python.exe python/etapa2_checkpoint.py
python/.venv/Scripts/python.exe python/etapa2_variantes.py
python/.venv/Scripts/python.exe python/etapa2_optuna.py --minutos 40 --auxiliar-total
python/.venv/Scripts/python.exe python/etapa2_bagging_n6.py
python/.venv/Scripts/python.exe python/umbral_sin_bogota.py
python/.venv/Scripts/python.exe python/etapa3_encuesta_corta.py 27
python/.venv/Scripts/python.exe python/etapa4_leaderboard.py
Rscript scripts/08_leaderboard.R && Rscript scripts/10_mejor_modelo.R
```

Un solo proceso pesado a la vez; LightGBM paraleliza con hilos (`PS2_NUCLEOS`). Los
auxiliares y las predicciones quedan en caché en `stores/procesados/nivel7/`, y cada etapa
escribe hora y RAM en `stores/logs/nivel7.log`.

## Decisiones de diseño

- **Clasificación directa** de `pobre`, como pide el enunciado.
- **Cancha nivelada**: todos los algoritmos comparten la misma base y la misma receta
  (`crear_receta()`), así las diferencias de desempeño reflejan el algoritmo.
- Solo usamos variables presentes en **train y test**.
- Hiperparámetros por **validación cruzada de 5 folds** estratificada, con grilla regular o
  *latin hypercube* según el número de hiperparámetros (rangos en `especificar_algoritmo()`).
- El **umbral de clasificación** se elige maximizando el F1 fuera de fold, y se reporta la
  curva completa para discutir filtración vs. subcobertura.
- **Sin fuga de información**: imputación, normalización, dummies y balanceo viven en la
  receta y se estiman solo con la parte de entrenamiento de cada fold.

## Checklist de entregables

- [ ] ≥ 15 submissions en Kaggle
- [ ] ≥ 5 algoritmos distintos (LPM, Logit, Elastic Net, CART, RF, Boosting, Naive Bayes)
- [ ] ≥ 5 commits sustanciales por integrante en `main`
- [ ] `document/slides/best_equipo_06.pdf`
- [ ] `document/slides/compare_equipo_06.pdf`
- [ ] Ambos decks con la diapositiva *Best Model Overview* de primera

## Cómo trabajamos

- Mensajes de commit en español y descriptivos (`Agrega tasa de dependencia al hogar`).
- Una iteración del modelo = un commit.
- Si usan ramas, fusionen a `main` con *merge commit* (no *squash*) para que cada commit
  conserve a su autor.
- `.claude/` tiene la configuración compartida de Claude Code: sin firmas de coautoría y con
  las reglas de estilo del proyecto.
