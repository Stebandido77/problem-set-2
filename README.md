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
| Neislen | [@NeislenG](https://github.com/NeislenG) |
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
│   ├── 07b_modelos_complementarios.R # Cinco candidatos económicos con variables complementarias
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
│   ├── etapa1_ablacion.py ... etapa10b_submissions.py
│   ├── etapa11_estrato.py            # Estrato como auxiliar y reconciliación del arriendo
│   ├── etapa12_enfoques.py           # Cuantiles, DeepSets, GPBoost, KNN fuera de fold
│   ├── etapa12_autogluon.py          # AutoGluon (entorno aparte: requirements_autogluon.txt)
│   ├── etapa13_mezcla_final.py       # Mezcla final desde predicciones guardadas (sin reentrenar)
│   ├── etapa14_busquedas.py          # Programación genética de variables + colonia de abejas sobre RealMLP
│   ├── etapa15_modelos_politica.py   # Pseudoetiquetado, Fex_c como peso, EBM y análisis de política
│   ├── etapa16_evaluacion.py         # Evaluación definitiva: CV repetida, bootstrap pareado del holdout, MDE
│   ├── diagnostico_forense.py        # Auditoría de datos, fugas, submissions y shift
│   ├── funciones/                    # Variables v2, OOF anidada, auxiliares, métricas, repetición e inferencia
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
| 4+. Variables complementarias | nivel 3 + 8 conteos/tasas laborales y hacinamiento ya derivados | Logit, Elastic Net, CART, Random Forest (150 árboles) y XGBoost (hasta 600 árboles) con búsqueda económica |
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

El nivel 4+ compara cinco candidatos sobre el mismo bloque de variables. Sus grillas se
reducen para controlar costo (Elastic Net: 12 combinaciones; CART: 4; Random Forest:
4 configuraciones y 150 árboles; XGBoost: 4 configuraciones). Cada submission se evalúa
con el mismo holdout y umbral OOF; el top 3 manual no cambia automáticamente.

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
python/.venv/Scripts/python.exe python/etapa11_estrato.py
python/.venv/Scripts/python.exe python/diagnostico_forense.py
python/.venv/Scripts/python.exe python/etapa12_enfoques.py preparar
python/.venv_autogluon/Scripts/python.exe python/etapa12_autogluon.py --corrida A --minutos 40
python/.venv_autogluon/Scripts/python.exe python/etapa12_autogluon.py --corrida B --minutos 55
for e in distribucional deepsets gpboost knn cierre; do python/.venv/Scripts/python.exe python/etapa12_enfoques.py $e; done
python/.venv/Scripts/python.exe python/etapa13_mezcla_final.py
for e in genetico abejas cierre; do python/.venv/Scripts/python.exe python/etapa14_busquedas.py $e; done
# Si un proceso murió antes del contexto test, --sin-test cierra el candidato desde sus parciales sin entrenar
python/.venv/Scripts/python.exe python/etapa15_modelos_politica.py todo   # o, desacoplado: Start-Process (ver la etapa 15)
python/.venv/Scripts/python.exe python/etapa16_evaluacion.py fase1       # o, desacoplado: Start-Process (ver la etapa 16)
Rscript scripts/08_leaderboard.R && Rscript scripts/10_mejor_modelo.R
```

AutoGluon vive en un entorno aparte (`py -3.12 -m venv python/.venv_autogluon` e instalar
`python/requirements_autogluon.txt`) porque fija versiones distintas de numpy y pandas.

Un solo proceso pesado a la vez; LightGBM paraleliza con hilos (`PS2_NUCLEOS`). Los
auxiliares y las predicciones quedan en caché en `stores/procesados/nivel7/`, y cada etapa
escribe hora y RAM en `stores/logs/nivel7.log`.

### Etapa 11: el estrato como objetivo auxiliar y el arriendo imputado

`Estrato1` solo existe en train y es una característica del hogar (constante entre sus
personas; `11_exportar_python.R` lo comprueba y lo exporta por hogar). `etapa11_estrato.py`
lo usa como **objetivo auxiliar**, nunca como predictor: un LightGBM multiclase sobre los
280 predictores de hogar de a4 (presentes en test) lo predice con la misma OOF anidada de
los auxiliares de ingreso, y al clasificador entran el estrato esperado y P(estrato <= 2).
La comparación es de 1 semilla, con los parámetros de n6, contra `n7_v2_total_suma`
(a4 + auxiliares de ingreso laboral y total):

| Candidato (1 semilla) | F1 CV | F1 holdout | Δ CV | Δ holdout |
|---|---|---|---|---|
| base: `n7_v2_total_suma` | 0,7693 | 0,7776 | | |
| + estrato esperado y P(estrato <= 2) | 0,7686 | 0,7766 | −0,0007 | −0,0010 |
| per cápita de los auxiliares con arriendo imputado | 0,7694 | 0,7760 | +0,0000 | −0,0016 |

El auxiliar predice bien el estrato fuera de muestra (Spearman 0,89 con el observado,
AUC 0,94 para estrato <= 2), pero no agrega F1: la regla de la etapa (subir más de 0,003
en CV y en holdout) no se cumple, así que no se corre el bagging sobre la base de A ni se
genera CSV. La reconciliación del arriendo imputado (`arriendo_imputado_hogar()`: P5130 en
vivienda propia pagada y en usufructo, P5130 − P5100 en propia pagando; reproduce
`Ingtotugarr − Ingtotug` en el 98,4 % de los hogares de train) tampoco mejora, de modo que
`agregar_ingreso_hogar()` la conserva como opción (`arriendo_imputado=False` por defecto) y
`cargar_insumos()` puede aplicarla sobre las cachés existentes. Tablas en
`views/tablas/nivel7/etapa11_*.csv`.

### Diagnóstico forense

`diagnostico_forense.py` audita los datos crudos, las cachés y las entregas sin modificar
nada (tablas `views/tablas/nivel7/forense_*.csv`; `--solo seccion,...` corre una parte).
Lo que se comprobó:

- **Construcción**: cero personas sin hogar, cero hogares sin personas, `Nper` y `Npersug`
  coinciden con los conteos, un jefe por hogar, y la suma de `Ingtot` de la unidad de gasto
  reproduce `Ingtotug` en todos los hogares. `Pobre == (Ingpcug < Lp)` sin excepciones;
  `Ingpcug = Ingtotugarr / Npersug`, donde `Ingtotugarr` suma el arriendo imputado a
  propietarios (78.295 hogares; sin él cambiaría la clase de 8.994). Los auxiliares de
  ingreso no incluyen ese componente: la reconciliación se probó en la etapa 11.
- **Poda de agregados**: `podar_columnas()` compara sumas de hashes sin orden y descartó
  como duplicadas dos columnas que no lo eran (`ag_conyuge_P6870_5` y
  `ag_conyuge_P7150_2`); el efecto predictivo no está medido.
- **Submissions**: los 30 CSV tienen los 66.168 ids del test en el mismo orden, y en los
  cinco candidatos con predicciones guardadas la clase coincide con probabilidad y umbral OOF.
- **Ids y orden**: ids hexadecimales de 24 caracteres sin intersección train-test; ni los
  prefijos ni los fragmentos numéricos se asocian con la pobreza. El orden de las filas del
  CSV sí tiene estructura (pobreza por deciles entre 19,3 % y 21,5 %, chi cuadrado
  p = 0,00002) que reproduce la composición geográfica. Ni ids ni orden se usan como
  predictores.
- **Superposición**: 21 hogares de test tienen un perfil exacto en train y 438 uno casi
  exacto (casi todos unipersonales); solo un par coincide en todos los predictores. No se
  usan para predecir.
- **Validación adversarial** (train sin Bogotá vs test, LightGBM de 5 folds): AUC 0,713
  con las 1.324 columnas de A y **0,501** con las 1.306 observadas, sin los 18 auxiliares.
  La separación viene de las representaciones auxiliares (ajuste completo frente a OOF), no
  de un cambio detectable en los predictores originales.
- **Experimentos** (clasificador n6 de 1 semilla sobre la base de A): añadir auxiliares de
  indigencia y log(Ingpcug/Lp) da F1 holdout 0,7785 frente a 0,7798 de la base (IC 95 %
  pareado de −0,004 a +0,002); reponderar por los pesos adversariales sin Bogotá da 0,7808
  frente a 0,7802 (IC de −0,002 a +0,003) y con los pesos estáticos 0,7803. Ninguna
  diferencia es distinguible de cero.

### Etapa 12: enfoques no probados (AutoML, distribucional, conjuntos, efectos aleatorios, vecinos)

`etapa12_enfoques.py` y `etapa12_autogluon.py` prueban cinco familias que no habían
entrado en la escalera, todas con los mismos folds, el mismo holdout, el umbral elegido en
OOF con `buscar_umbral_f1()`, OOF anidada para cualquier variable construida con la etiqueta
y los tests de fuga del pipeline. La base es `n7_a_agregados_bag5` (LightGBM n6 sobre la
base de A, bagging de 5 semillas); los candidatos de 1 semilla se comparan primero con
`n12_a_1s` (la misma receta con 1 semilla) y solo pasan a bagging si suben +0,002 en CV y
en holdout.

- **AutoGluon** (`best_quality`, métrica F1, bagging por nuestros 5 folds vía `groups`,
  sin calibración de umbral propia). Corrida A: base casi cruda de 353 columnas (variables
  de hogar tal cual, incluidos los factores de expansión, valores del jefe y del cónyuge,
  media y máximo del resto y de todos por cada variable original de persona, conteos por
  rol), sin variables construidas ni auxiliares. Corrida B: la base de A con los
  auxiliares anidados (1.324 columnas).
- **Regresión distribucional**: 19 cuantiles (0,05 a 0,95) de log(Ingpcug/Lp) con
  LightGBM sobre a4, anidados; de ellos salen P(Ingpcug < Lp) implícita (interpolada),
  mediana y ancho intercuantil. Se usan para clasificar directo y como variables en A.
  NGBoost y LightGBMLSS no se instalaron porque fijan versiones de numpy, pandas y torch
  incompatibles con el entorno.
- **DeepSets con atención** (`funciones/conjuntos.py`): cada hogar como conjunto de sus
  personas (152 variables por persona) más 121 de hogar; codificador compartido, agregación
  por suma, media, máximo y atención, parada temprana en un 10 % interno; 3 semillas y
  mezcla con LightGBM con pesos elegidos en OOF.
- **GPBoost**: boosting con intercepto aleatorio por dominio (bernoulli-logit) sobre A.
- **KNN fuera de fold**: tasa de pobreza de los 10, 50 y 200 vecinos más cercanos en las
  60 variables de mayor ganancia estandarizadas, calculada solo con el entrenamiento de
  cada contexto (y con 5 folds internos para las filas de entrenamiento), añadida a A.

| Candidato | F1 CV (sd) | F1 holdout | Prec. | Recall | F1 banda [0,8; 1,2] Lp | Min | Δ CV | Δ holdout | Corr. OOF con LightGBM |
|---|---|---|---|---|---|---|---|---|---|
| base: `n7_a_agregados_bag5` | 0,7748 (0,005) | 0,7806 | 0,748 | 0,816 | 0,650 | | | | 1 |
| referencia: `n12_a_1s` | 0,7737 (0,005) | 0,7798 | 0,743 | 0,820 | 0,653 | 6,7 | −0,0011 | −0,0008 | 0,9996 |
| AutoGluon A (base cruda) | 0,7622 (0,005) | 0,7635 | 0,746 | 0,782 | 0,618 | 44,6 | −0,0125 | −0,0171 | 0,967 |
| AutoGluon B (base de A) | 0,7715 (0,003) | 0,7771 | 0,756 | 0,800 | 0,645 | 100,3 | −0,0033 | −0,0035 | 0,990 |
| cuantiles directo (P(<Lp) + umbral OOF) | 0,7532 (0,003) | 0,7567 | 0,723 | 0,793 | 0,631 | 6,8 | −0,0216 | −0,0239 | 0,948 |
| A + cuantiles (1 semilla) | 0,7750 (0,004) | 0,7777 | 0,747 | 0,811 | 0,647 | 4,4 | +0,0002 | −0,0029 | 0,993 |
| DeepSets (1 semilla) | 0,7505 (0,003) | 0,7529 | 0,726 | 0,782 | 0,625 | 6,0 | −0,0242 | −0,0277 | 0,949 |
| DeepSets (bagging 3 semillas) | 0,7538 (0,002) | 0,7582 | 0,728 | 0,791 | 0,631 | 17,1 | −0,0210 | −0,0224 | 0,956 |
| LightGBM 0,90 + DeepSets 0,10 (pesos OOF) | 0,7751 (0,005) | 0,7812 | 0,753 | 0,812 | 0,648 | | +0,0004 | +0,0006 | 0,9996 |
| GPBoost por dominio (1 semilla) | 0,7710 (0,005) | 0,7737 | 0,758 | 0,790 | 0,632 | 2,7 | −0,0038 | −0,0069 | 0,991 |
| A + KNN fuera de fold (1 semilla) | 0,7732 (0,004) | 0,7816 | 0,746 | 0,821 | 0,655 | 4,4 | −0,0016 | +0,0010 | 0,996 |

Los deltas son frente a `n7_a_agregados_bag5` (los minutos del cierre incluyen la OOF, el
holdout y, en AutoGluon, el leaderboard y la importancia por permutación). **Ningún
candidato cumple la regla** de +0,002 en CV y en holdout, de modo que la etapa no genera
CSV. Lo que se aprendió:

- AutoGluon sobre la base cruda (ensamble ponderado de LightGBM, XGBoost, CatBoost,
  LightGBM extra trees, LightGBM grande y una red; CatBoost con bagging consumió 29 de los
  40 minutos) queda 1,7 puntos de holdout por debajo; sus variables más importantes por
  permutación son arriendo, tenencia, número de miembros no jefe ni cónyuge, el ingreso de
  desocupados y la edad media, es decir, las mismas señales que ya usamos. Sobre nuestra
  base (cinco redes y LightGBM extra trees; el LightGBM normal no terminó en 55 minutos)
  queda 0,35 puntos por debajo, y sus variables más importantes son nuestros auxiliares de
  ingreso anidados. Lo que AutoGluon tiene y nosotros no es el ensamble automático de
  redes y árboles; lo que nos da ventaja son los auxiliares con OOF anidada, que valen
  1,4 puntos de holdout entre la corrida A y la B.
- Los cuantiles quedan bien calibrados (cobertura OOF a menos de 0,01 de cada alfa,
  `etapa12_cuantiles_calibracion.csv`), pero la P(<Lp) implícita correlaciona 0,993 en OOF
  con la base: no hay información nueva sobre los auxiliares de ingreso.
- DeepSets sola rinde como el resto de redes (etapas 8 y 10), 2,2 puntos por debajo; la
  curva de pesos de la mezcla es plana entre 0,05 y 0,20 de red
  (`etapa12_pesos_lgb_conjuntos.csv`).
- GPBoost estima una varianza del intercepto por dominio cercana a cero: las dummies de
  dominio y los auxiliares ya absorben esa heterogeneidad.
- El KNN fuera de fold es la única variable que sube en holdout (+0,0018 frente a la
  referencia de 1 semilla), pero baja en CV y no cruza el umbral.

Tablas en `views/tablas/nivel7/etapa12_*.csv` (`etapa12_resumen.csv` es la de arriba).

### Etapa 13: mezcla final desde las predicciones guardadas

`etapa13_mezcla_final.py` no reentrena nada: toma las predicciones OOF, de holdout y de
test que ya estaban en caché y mezcla la base `n7_a_agregados_bag5` con los tres
candidatos menos correlacionados con ella que dejaron las etapas 10 y 12: A + KNN fuera de
fold (`n12_a_knn_1s`), RealMLP (`n10_realmlp_bag5`) y DeepSets con atención
(`n12_conjuntos_bag3`). Los pesos se eligen solo en la OOF (todo el simplex de paso 0,05:
1.771 vectores con cuatro candidatos y 231 con tres, F1 con el umbral óptimo) y el umbral
también sale de la OOF; holdout y test reciben la misma combinación. La grilla incluye a la
base sola, así que la mezcla nunca pierde contra ella en OOF.

`n12_a_knn_1s` no tiene predicción de test (la etapa 12 no reajustó sobre el 100 % porque
no pasó la regla), de modo que la mezcla de los cuatro se reporta solo en CV y holdout y el
CSV sale de los tres candidatos con test.

| Candidato | Pesos (elegidos en OOF) | F1 CV (sd) | F1 holdout | Prec. | Recall | Δ CV | Δ holdout | Corr. OOF con la base |
|---|---|---|---|---|---|---|---|---|
| base: `n7_a_agregados_bag5` | | 0,7748 (0,005) | 0,7806 | 0,748 | 0,816 | | | 1 |
| `n13_mezcla_con_knn` (sin test) | 0,60 base + 0,20 KNN + 0,10 RealMLP + 0,10 DeepSets | 0,7756 (0,004) | 0,7819 | 0,748 | 0,818 | +0,0009 | +0,0013 | 0,9989 |
| `n13_mezcla_final` | 0,80 base + 0,15 RealMLP + 0,05 DeepSets | 0,7753 (0,005) | 0,7820 | 0,749 | 0,818 | +0,0006 | +0,0014 | 0,9993 |

Ninguna cumple la regla de +0,002 en CV y en holdout. Aun así `n13_mezcla_final.csv` se
genera como submission extra (validada: 66.168 filas, tasa de pobres 0,232, difiere de
`n7_a_agregados_bag5.csv` en 345 hogares, el 0,5 %), queda en la bitácora con `f1_kaggle`
vacía y en el leaderboard del nivel 7. La curva de pesos es plana (las seis mejores
combinaciones de cada grilla están a menos de 0,0003 de F1 OOF entre sí) y el KNN aporta
+0,0003 en CV y −0,0001 en holdout frente a la mezcla de tres, así que no justifica
reajustarlo sobre el 100 % para meterlo en el CSV. Tablas en
`views/tablas/nivel7/etapa13_*.csv`.

### Etapa 14: programación genética de variables y colonia de abejas sobre RealMLP

`etapa14_busquedas.py` corre tres experimentos desacoplados, cada uno con su tope de
minutos y corte duro (genético 120, abejas 90, cierre 15), con los mismos folds, holdout y
OOF anidada del pipeline y con el F1 de la clase pobre con umbral OOF como aptitud en las
dos búsquedas. Las bases son `n13_mezcla_final` (CV 0,7753 / holdout 0,7820; regla del CSV)
y `n7_a_agregados_bag5` (correlación OOF y mezclas). Todo lo que se construye con la
etiqueta se ajusta solo con el entrenamiento de cada contexto.

**Programación genética de variables.** `funciones/formulas.py` envuelve el
`SymbolicTransformer` de gplearn sobre las 60 variables de más gain de la etapa 7: población
1.000, 25 generaciones, torneo 20, cruce 0,7, mutación de subárbol 0,1, de elevación 0,05 y de
punto 0,1, parsimonia 0,0005, funciones +, −, ×, ÷, log, máx y mín. La aptitud es el F1 con
umbral óptimo del pipeline sobre una submuestra del 25 % del entrenamiento del contexto (la
salida del programa se pasa a rangos y se toma la mejor de las dos orientaciones), y de las 50
mejores del salón de la fama quedan las 15 menos correlacionadas. Se ajusta por contexto (7
ajustes de ~3 min) en un subproceso liviano (`python -m funciones.formulas`), porque los
workers de gplearn en Windows reimportan el módulo principal y mueren si este carga torch.
Las 15 fórmulas se agregan a la base de A y se ajustan LightGBM n6 y XGBoost con n6 traducido
(`hist`, `lossguide`, `max_leaves` 31, `min_child_weight` 27 ≈ 171 × 0,16 de hessiana media,
`gamma`, `subsample` y `colsample_bynode` de n6): primero una semilla contra la referencia
de una semilla `n12_a_1s`, y bagging de 5 solo si mejora +0,002 en CV y en holdout.

| Candidato (1 semilla) | F1 CV (sd) | F1 holdout | Δ CV vs `n12_a_1s` | Δ holdout | Minutos |
|---|---|---|---|---|---|
| `n14_a_gp_lgb_s0` (A + 15 fórmulas, LightGBM n6) | 0,7741 (0,0035) | 0,7785 | +0,0004 | −0,0013 | 8,5 |
| `n14_a_gp_xgb_s0` (A + 15 fórmulas, XGBoost n6) | 0,7740 (0,0044) | 0,7788 | +0,0003 | −0,0009 | 21,2 |

Ninguno pasa la regla, así que no hubo bagging. Las fórmulas ganadoras
(`views/tablas/nivel7/etapa14_gp_formulas.csv`, 105 fórmulas de longitud 1 a 25, mediana 5)
dicen por qué: las 105 usan `log_ingreso_pred_pc_lp_v3`, 73 `arriendo_por_persona_lp`, 30
`tenencia_propia_pagada`, 30 `log_arriendo_relativo`, 14 `tenencia_usufructo` y 14
`cuartos_por_persona`, casi siempre sumadas (312 sumas, 61 máximos, 17 mínimos, 1 resta y 1
división). Como la aptitud es invariante a transformaciones monótonas, el genético suma el
ingreso predicho consigo mismo y le agrega términos de arriendo y tenencia. Las mejores por
contexto, simplificadas: `cv_1` (F1 0,7597) `3·ingreso_pred + arriendo_por_persona +
tenencia_propia_pagada + tenencia_usufructo`; `cv_2` (0,7480) `ingreso_pred +
arriendo_por_persona`; `cv_3` (0,7397) y `cv_4` (0,7430) `2·ingreso_pred +
arriendo_por_persona`; `cv_5` (0,7523) `3·ingreso_pred + 2·arriendo_por_persona +
mín(tenencia_propia_pagada, cuartos_por_persona)`; holdout (0,7387) y test (0,7413)
`máx(log_arriendo_relativo, ingreso_pred)`. Son combinaciones de variables que los árboles ya
tienen por separado, de ahí la correlación OOF de 0,998 con LightGBM y la ausencia de mejora.

**Colonia de abejas sobre RealMLP.** `funciones/abejas.py` (3 fuentes, límite de abandono
2) se mueve en seis posiciones continuas en [0, 1] decodificadas a capas {2, 3, 4}, ancho
{128, 256, 384, 512}, dropout [0, 0,4], lr [0,005, 0,08] (log), weight decay [1e-4, 5e-2]
(log) y embeddings numéricos PBLD sí/no; la configuración de la etapa 10 entra en la
población inicial. La aptitud es el F1 OOF (umbral óptimo) de los folds 1 a 3, cada uno
ajustado con una submuestra estratificada del 25 % de su entrenamiento (26.393 hogares) y
evaluado completo. Una evaluación sin embeddings tarda 1,3 min y una con PBLD 8,5 min, así
que un ciclo (7 evaluaciones) necesita ~60 min: con 90 min de presupuesto y 40 reservados
para la evaluación completa la colonia se quedó en el ciclo 0 con las 3 evaluaciones
iniciales (`etapa14_abc_historia.csv`): etapa 10 (3 × 256, dropout 0,15, lr 0,02, wd 0,01,
sin embeddings) 0,7566; 3 × 256 con dropout 0,39, lr 0,064, wd 0,0052 y PBLD 0,7580; 4 × 256
con PBLD 0,7550. La mejor se evaluó completa (`n14_realmlp_abc_s0`): 5 folds y holdout en
67 min (~11 min por contexto con PBLD). El proceso murió por falta de memoria del sistema
antes del contexto test, y la opción nueva `--sin-test` lo cerró desde los parciales de los
5 folds y el holdout sin entrenar nada (sin test no hay CSV posible; al volver a correr sin
la opción solo se ajusta el test). Frente a `n10_realmlp_s0`: CV −0,0003, holdout +0,0040.

**Cierre.** Tabla con las bases, los candidatos y la mezcla de cada uno con
`n7_a_agregados_bag5` (pesos en {0, 0,05, …, 0,5} elegidos solo en OOF, umbral OOF). Banda
de Lp = F1 en los hogares del holdout cuyo ingreso per cápita observado está entre 0,8 y 1,2
veces la línea de pobreza.

| Candidato | F1 CV (sd) | F1 holdout | Prec. | Recall | Banda Lp | Corr. OOF con LightGBM | Δ CV vs n13 | Δ holdout vs n13 |
|---|---|---|---|---|---|---|---|---|
| base: `n13_mezcla_final` | 0,7753 (0,0049) | 0,7820 | 0,749 | 0,818 | 0,653 | 0,9993 | | |
| base: `n7_a_agregados_bag5` | 0,7748 (0,0050) | 0,7806 | 0,748 | 0,816 | 0,650 | 1 | −0,0006 | −0,0014 |
| `n14_a_gp_lgb_s0` | 0,7741 (0,0035) | 0,7785 | 0,755 | 0,803 | 0,643 | 0,9982 | −0,0012 | −0,0035 |
| mezcla 0,85 base + 0,15 `n14_a_gp_lgb_s0` | 0,7750 (0,0046) | 0,7799 | 0,751 | 0,811 | 0,648 | 1,0000 | −0,0003 | −0,0021 |
| `n14_a_gp_xgb_s0` | 0,7740 (0,0044) | 0,7788 | 0,751 | 0,809 | 0,647 | 0,9975 | −0,0014 | −0,0032 |
| mezcla 0,90 base + 0,10 `n14_a_gp_xgb_s0` | 0,7750 (0,0046) | 0,7805 | 0,752 | 0,812 | 0,648 | 1,0000 | −0,0004 | −0,0015 |
| `n14_realmlp_abc_s0` (sin test) | 0,7660 (0,0041) | 0,7757 | 0,739 | 0,816 | 0,654 | 0,9847 | −0,0093 | −0,0063 |
| mezcla 0,95 base + 0,05 `n14_realmlp_abc_s0` (sin test) | 0,7748 (0,0047) | 0,7804 | 0,752 | 0,811 | 0,648 | 1,0000 | −0,0005 | −0,0016 |

Ningún candidato ni mezcla supera a `n13_mezcla_final` en CV y en holdout, así que la etapa
no genera CSV ni toca la bitácora; los seis candidatos quedan en el leaderboard del nivel 7
sin archivo. Las curvas de pesos son planas (`etapa14_pesos_*.csv`: la base sola está a menos
de 0,0003 de F1 OOF de la mejor mezcla). Tiempo usado: genético 53 min de 120; abejas 94 min
de 90 (el reloj compartido entre relanzamientos contó los 9 min entre la muerte del proceso y
el cierre con `--sin-test`, que se lanzó con `--minutos 120` para que el corte duro no lo
matara); cierre 0,1 min. Tablas en `views/tablas/nivel7/etapa14_*.csv`.

### Etapa 15: pseudoetiquetado, factores de expansión, EBM y análisis de política

`etapa15_modelos_politica.py` corre seis experimentos desacoplados, cada uno con su tope
de minutos, corte duro, vigilante de memoria y guardado reanudable por contexto. `todo` los
encadena como subprocesos (relanzando el que salga por memoria) y se lanza con
`Start-Process` para que no dependa de la terminal. Reutiliza la infraestructura de la
etapa 14: una semilla contra `n12_a_1s`, bagging solo si mejora +0,002 en CV y holdout,
cierre con mezclas con `n7_a_agregados_bag5` (pesos solo en OOF) y CSV solo si supera a
`n13_mezcla_final` en CV y en holdout. Toda la corrida tomó 27 minutos de los 180 previstos
porque ningún candidato pasó el gate.

**Parte A, experimentos de modelo.**

- *Ensamble por fold.* No hay predicciones de test de los cinco modelos de la CV de
  `n7_a_agregados_bag5`: `evaluar_modelo()` predice en cada `cv_k` solo su fold de
  validación y borra los parciales al terminar, así que se informa y se sigue
  (`etapa15_ensamble_fold.csv`). `predicciones_test_por_fold()` queda lista para leerlas de
  `stores/procesados/nivel7/test_por_fold/` si alguna etapa las guarda.
- *Pseudoetiquetado* (`n15_pseudo_s0`). En cada contexto, el LightGBM n6 ajustado solo con su
  entrenamiento predice el test de Kaggle; los hogares con P(pobre) ≥ 0,95 o ≤ 0,05 (unos
  42.250 por contexto, el 64 % del test: ~2.260 pobres y ~40.000 no pobres) entran al
  entrenamiento con su etiqueta predicha y peso 0,5 (columna `peso_fila`, que
  `crear_ajuste_lightgbm()` convierte en `sample_weight`). Las etiquetas nunca ven el holdout
  ni el fold de validación; la única salvedad es que las variables de esos hogares (los
  auxiliares de ingreso) son las del contexto test, porque las matrices de A no tienen
  auxiliares por fold para el test. Resultado: CV 0,7744 (sd 0,0037) / holdout 0,7805,
  +0,0007 y +0,0008 frente a `n12_a_1s`. No pasa el gate, aunque con una sola semilla iguala
  al bagging de cinco (0,7748 / 0,7806) y baja la desviación entre folds.
- *Factores de expansión como pesos* (`n15_fex_s0`). LightGBM n6 sobre A con
  `sample_weight = Fex_c / media(Fex_c)` del entrenamiento de cada contexto (el factor no
  entra como variable). CV 0,7715 / holdout 0,7755, −0,0022 y −0,0043 frente a `n12_a_1s`.
  Tampoco gana en la métrica que optimiza: el F1 de holdout ponderado por Fex_c es 0,7778
  contra 0,7776 de `n12_a_1s`, 0,7790 de `n7_a_agregados_bag5` y 0,7820 de
  `n13_mezcla_final` (`etapa15_fex.csv`), mientras el F1 sin ponderar, el de Kaggle, cae
  0,004.
- *EBM* (`n15_ebm_top60_s0`). `ExplainableBoostingClassifier` de interpret sobre las 60
  variables de más gain de A, 20 interacciones elegidas por el propio EBM, 8 bolsas y early
  stopping: 6,4 minutos los siete contextos. CV 0,7706 (sd 0,0051) / holdout 0,7768, −0,0031
  y −0,0030 frente a `n12_a_1s`, que usa las 1.323 variables. Su mezcla 0,85 + 0,15 con
  `n7_a_agregados_bag5` da el mejor CV de la etapa (0,7754, +0,0001 sobre n13) pero
  −0,0017 en holdout. Las curvas de las 10 variables más importantes están en
  `views/figuras/nivel7/ebm_01_*.png` a `ebm_10_*.png` (aporte al log-odds por tramo, con la
  banda entre bolsas y la densidad de hogares) y la importancia de los 15 términos
  principales en `ebm_importancia.png` (`etapa15_ebm_importancia.csv`). Mandan el ingreso
  total predicho (0,567; aporte positivo hasta ~2 millones y un escalón a −2,3 pasados los
  5 millones), el arriendo por persona relativo a Lp (0,496), la tenencia propia pagada
  (0,434), el log del ingreso per cápita predicho sobre Lp (0,408) y tres proporciones de
  perceptores de ingresos no laborales de la GEIH; la interacción más fuerte es arriendo por
  persona × tenencia propia pagada (0,394).

| Candidato | F1 CV (sd) | F1 holdout | Prec. | Recall | Banda Lp | Corr. OOF con LightGBM | Δ CV vs n13 | Δ holdout vs n13 |
|---|---|---|---|---|---|---|---|---|
| base: `n13_mezcla_final` | 0,7753 (0,0049) | 0,7820 | 0,749 | 0,818 | 0,653 | 0,9993 | | |
| base: `n7_a_agregados_bag5` | 0,7748 (0,0050) | 0,7806 | 0,748 | 0,816 | 0,650 | 1 | −0,0006 | −0,0014 |
| `n15_pseudo_s0` | 0,7744 (0,0037) | 0,7805 | 0,752 | 0,812 | 0,648 | 0,9992 | −0,0009 | −0,0015 |
| mezcla 0,80 base + 0,20 `n15_pseudo_s0` | 0,7751 (0,0045) | 0,7799 | 0,752 | 0,810 | 0,647 | 1,0000 | −0,0002 | −0,0021 |
| `n15_fex_s0` | 0,7715 (0,0048) | 0,7755 | 0,741 | 0,813 | 0,650 | 0,9931 | −0,0038 | −0,0065 |
| mezcla 0,85 base + 0,15 `n15_fex_s0` | 0,7748 (0,0049) | 0,7790 | 0,750 | 0,811 | 0,646 | 0,9998 | −0,0005 | −0,0030 |
| `n15_ebm_top60_s0` | 0,7706 (0,0051) | 0,7768 | 0,745 | 0,811 | 0,648 | 0,9886 | −0,0047 | −0,0052 |
| mezcla 0,85 base + 0,15 `n15_ebm_top60_s0` | 0,7754 (0,0050) | 0,7803 | 0,748 | 0,816 | 0,650 | 0,9998 | +0,0001 | −0,0017 |

Ninguno supera a `n13_mezcla_final` en CV y en holdout a la vez: no hay CSV nuevo ni cambios
en la bitácora; los seis candidatos quedan en el leaderboard del nivel 7 sin archivo.

**Parte B, análisis de política** (`funciones/politica.py`, con `n13_mezcla_final` en el
holdout: 32.993 hogares, 6.605 pobres, umbral OOF 0,36; nada se reentrena). Subcobertura =
pobres no incluidos / pobres; filtración = no pobres incluidos / incluidos.

- *Errores por subgrupo* (`etapa15_errores_subgrupo.csv`, figura
  `errores_subgrupo_n13_mezcla_final.png`). En total el modelo deja fuera al 18,2 % de los
  pobres (1.201) e incluye 1.812 no pobres (25,1 % de los incluidos). Cuatro grupos superan
  1,5 veces la subcobertura total (27,3 %): los hogares de una persona (33,6 %: 153 de 455
  pobres) y de dos (28,5 %: 241 de 847), Manizales (35,7 %) y Pereira (28,6 %). Los hogares
  de 5 o más personas, donde está el 36 % de los pobres, tienen la subcobertura más baja
  (10,8 %). Zona (rural 17,8 % vs. urbano 18,2 %), jefatura (femenina 16,6 % vs. masculina
  19,5 %) y adultos mayores sin pensión (17,3 % vs. 18,4 %) no muestran brechas. La
  filtración también se concentra en los hogares de una y dos personas (32 %) y en Tunja,
  Manizales y Bucaramanga (33 % a 36 %).
- *Pesos de bienestar* (`etapa15_pesos_bienestar.csv`). Cada pobre excluido pesa su brecha
  (1 − Ingpcug/Lp) y cada no pobre incluido su distancia a la línea (Ingpcug/Lp − 1).

| Modelo | Subcobertura | Ponderada por brecha | Brecha media: excluidos / todos los pobres | Excluidos con ingreso ≥ 0,8 Lp | Excluidos con ingreso < 0,5 Lp | Filtración | Ponderada por distancia | Distancia media: incluidos / todos los no pobres |
|---|---|---|---|---|---|---|---|---|
| `n13_mezcla_final` | 18,2 % | 11,2 % | 0,224 / 0,365 | 58,8 % | 11,2 % | 25,1 % | 1,2 % | 0,51 / 2,83 |
| `n3_logit` (48 variables) | 23,2 % | 16,6 % | 0,261 / 0,365 | 53,2 % | 14,8 % | 32,7 % | 1,7 % | 0,51 / 2,83 |
| `nE_logit` (10 variables) | 31,6 % | 27,7 % | 0,320 / 0,365 | 43,3 % | 22,0 % | 41,4 % | 3,0 % | 0,71 / 2,83 |
| encuesta corta, Logit de 8 variables | 26,0 % | 19,9 % | 0,280 / 0,365 | 47,9 % | 16,1 % | 39,8 % | 2,7 % | 0,63 / 2,83 |

  Los pobres que excluye `n13_mezcla_final` son los que están cerca de la línea, no los más
  pobres: su brecha media es 0,22 contra 0,37 del pobre promedio y 0,40 del pobre cubierto,
  el 59 % tiene ingreso de al menos 0,8 Lp y la subcobertura ponderada por brecha baja a
  11,2 %. Los no pobres que incluye están pegados a la línea (distancia media 0,51 contra
  2,83 del no pobre promedio; el 45 % a menos del 20 % de Lp), así que la filtración
  ponderada es del 1,2 %. Con modelos más simples la exclusión se corre hacia los más
  pobres: el 22 % de los excluidos de `nE_logit` y el 16 % de los de la encuesta corta
  están en pobreza extrema, contra el 11 % con n13. La encuesta corta de 8 variables (Logit
  reajustado con los mismos folds, F1 OOF 0,6588, umbral 0,3212) queda entre `n3_logit` y
  `nE_logit` en todas las medidas.
- *Curva de cobertura* (`etapa15_curva_cobertura.csv`, `etapa15_curva_cobertura_puntos.csv`,
  figura `curva_cobertura.png`): porcentaje de pobres cubiertos al atender el x % de hogares
  de mayor probabilidad, con la tasa de pobreza del holdout en 20,0 %. En su umbral OOF,
  `n13_mezcla_final` atiende al 21,9 % de los hogares y cubre al 81,8 % de los pobres;
  `n3_logit` 22,9 % y 76,8 %; la encuesta corta 24,6 % y 74,0 %. Atendiendo al 25 % de los
  hogares cubren el 86,8 %, 80,5 % y 74,7 %; al 30 %, el 92,9 %, 87,1 % y 81,5 %; al 40 %, el
  98,2 %, 94,7 % y 90,6 %.

Tiempo usado: fold 0 min, pseudoetiquetado 12,8, Fex_c 7,3, EBM 6,5, política 0,3 y cierre
0,1 (27,4 min de 180). Tablas en `views/tablas/nivel7/etapa15_*.csv` y figuras en
`views/figuras/nivel7/`.

### Etapa 16: evaluación definitiva de las hipótesis

Las etapas 7 a 15 compararon candidatos con una sola partición de 5 folds y un gate de
+0,002; las diferencias entre los mejores (0,7748 a 0,7753 en CV, 0,7806 a 0,7820 en
holdout) son del orden del ruido de muestreo, y los candidatos se eligieron mirando esa
misma partición y ese mismo holdout. `etapa16_evaluacion.py` mide si alguna mejora es
real con un diseño pensado para eso, en fases con reporte al final de cada una.

**Fase 1, precisión.** Validación cruzada repetida sobre el 80 %: tres particiones nuevas
de 5 folds estratificados por `pobre` (`funciones/repeticion.py`, semillas 2126, 2226 y
2326; la de R queda como referencia y no entra en las 15 evaluaciones porque sobre ella se
eligieron los candidatos). En cada partición se recalculan los auxiliares de ingreso
(laboral, total y fuentes v3) con la OOF anidada de siempre y `verificar_sin_fuga()`, y se
escriben las matrices de A de sus cinco contextos `cv_k`
(`stores/procesados/nivel7/etapa16/particion_r<r>/`). Los candidatos se reentrenan tal
como fueron definidos: `n7_a_agregados_bag5`, `n7_v3_fuentes_bag5` y
`n7_n6param_a4_total_bag5` con 5 semillas, CatBoost B3 con 1, RealMLP (configuración de
la etapa 10) con 5 y DeepSets con 3; las mezclas usan sus pesos fijos (`n13_mezcla_final`
0,80/0,15/0,05; LGB + RealMLP 0,85/0,15); A + KNN fuera de fold se corre con 5 semillas
para que la comparación con la base aísle las variables KNN (también en la partición
original, `n16_a_knn_bag5`, para tener su holdout). La métrica es la del pipeline: umbral
elegido en la OOF completa de cada partición y F1 de la clase pobre por fold con ese
umbral. El holdout es el de siempre: predicciones guardadas del pipeline con su umbral
OOF original y bootstrap pareado de 2.000 réplicas (la misma remuestra de hogares para
todos los candidatos). Comparaciones contra `n7_a_agregados_bag5` (`funciones/inferencia.py`):
diferencia por fold (misma partición y fold) con IC t al 95 % (14 gl) y diferencia por
réplica en el holdout con IC percentil; una mejora cuenta como real solo si el IC pareado
excluye cero en la CV repetida **y** en el holdout. El efecto mínimo detectable (MDE) es
(t₀,₉₇₅ + t₀,₈₀) × EE en CV y 2,80 × EE bootstrap en holdout (α = 0,05 bilateral,
potencia 0,80).

Corre desacoplado (`Start-Process`), con hasta dos subprocesos a la vez (8 hilos cada uno),
vigilante de memoria que sale limpio si el sistema baja de 2 GB libres (código 3; el
orquestador relanza ese trabajo solo, hasta 8 veces), guardado reanudable por fold y por
ajuste auxiliar, estado en `etapa16/fase1/estado.json`, `fin.txt` al terminar y log propio
en `stores/logs/etapa16_fase1.log`. `--prueba` corre la miniatura completa (una partición,
20 iteraciones, 2 semillas, 50 réplicas) en `etapa16_prueba/` sin tocar nada. Tablas en
`views/tablas/nivel7/etapa16_fase1_*.csv` (folds, pareadas, bootstrap, resumen y tiempos).

**Resultados de la fase 1** (corrida del 22 al 23 de octubre de 2026: 6,2 horas de reloj,
628 minutos de trabajo en 12 subprocesos, una sola salida limpia por memoria al arrancar las
redes de la partición 2, relanzada en solitario). F1 CV repetida = media de las 15
evaluaciones (error estándar entre paréntesis); Δ = candidato − `n7_a_agregados_bag5`, con
el IC 95 % pareado naive (15 pares) y, en CV, el corregido por dependencia entre folds
(Nadeau y Bengio, 2003; EE × √(1/15 + 1/4)).

| Candidato | F1 CV repetida (EE) | F1 OOF original | F1 holdout [IC 95 %] | Δ CV [IC naive] | Δ CV IC corregido | Δ holdout [IC bootstrap] | MDE CV / corr. / holdout | Min |
|---|---|---|---|---|---|---|---|---|
| `n7_a_agregados_bag5` (base) | 0,7752 (0,0008) | 0,7748 | 0,7806 [0,7731; 0,7885] | | | | | 79 |
| `n13_mezcla_final` | 0,7758 (0,0008) | 0,7753 | 0,7820 [0,7743; 0,7897] | +0,0006 [−0,0001; +0,0013] | [−0,0009; +0,0021] | +0,0014 [−0,0003; +0,0032] | 0,0010 / 0,0021 / 0,0025 | 292 |
| mezcla LGB + RealMLP | 0,7754 (0,0008) | 0,7750 | 0,7815 [0,7739; 0,7892] | +0,0003 [−0,0003; +0,0008] | [−0,0010; +0,0015] | +0,0008 [−0,0007; +0,0025] | 0,0008 / 0,0017 / 0,0023 | 241 |
| `n7_v3_fuentes_bag5` | 0,7749 (0,0007) | 0,7738 | 0,7792 [0,7715; 0,7872] | −0,0003 [−0,0008; +0,0002] | [−0,0014; +0,0008] | −0,0014 [−0,0038; +0,0011] | 0,0007 / 0,0015 / 0,0035 | 19 |
| A + KNN (5 semillas) | 0,7741 (0,0007) | 0,7741 | 0,7796 [0,7718; 0,7874] | −0,0011 [−0,0021; −0,0001] | [−0,0034; +0,0011] | −0,0010 [−0,0035; +0,0015] | 0,0014 / 0,0031 / 0,0035 | 77 |
| CatBoost B3 | 0,7722 (0,0007) | 0,7723 | 0,7786 [0,7707; 0,7864] | −0,0030 [−0,0042; −0,0018] | [−0,0056; −0,0004] | −0,0021 [−0,0054; +0,0011] | 0,0017 / 0,0037 / 0,0046 | 70 |
| `n7_n6param_a4_total_bag5` | 0,7697 (0,0008) | 0,7692 | 0,7760 [0,7683; 0,7838] | −0,0055 [−0,0067; −0,0043] | [−0,0081; −0,0030] | −0,0046 [−0,0088; −0,0003] | 0,0016 / 0,0036 / 0,0060 | 18 |

**Ninguna mejora es real** según la regla (IC pareado que excluya cero en CV repetida y en
holdout). Lo que sí se sostiene:

- La CV repetida reproduce la OOF original casi al milésimo (0,7752 frente a 0,7748 en la
  base): la partición de R no era especialmente favorable, y el orden de los candidatos es el
  mismo en las tres particiones nuevas.
- `n13_mezcla_final` le gana a la base en 9 de los 15 folds (diferencias por fold entre
  −0,0014 y +0,0034), por +0,0006 de media (IC naive [−0,0001; +0,0013]) y +0,0014 en
  holdout (IC [−0,0003; +0,0032]): ambos rozan el cero sin excluirlo. Es, con todo, el único candidato que no pierde en ninguna de las
  dos medidas, y sigue siendo el CSV recomendado.
- El KNN fuera de fold deja de parecer prometedor con 5 semillas: −0,0011 en CV (IC naive
  excluye cero por debajo) y −0,0010 en holdout; su holdout positivo de la etapa 12 era
  ruido de 1 semilla. CatBoost B3 y la base sin agregados ni fuentes pierden con claridad.
- **Efecto mínimo detectable**: con este diseño, una mejora tendría que ser de ~0,001 en CV
  (0,002 con la corrección por dependencia) y de 0,0025 a 0,006 en holdout, según cuánto
  difiera el candidato de la base, para detectarse con potencia 0,80. Todas las diferencias
  observadas entre los mejores candidatos están por debajo del MDE del holdout: el holdout
  de 32.993 hogares no distingue mejoras de un par de milésimas, y la CV repetida solo las
  distingue si son sistemáticas en los 15 folds.

Tablas en `views/tablas/nivel7/etapa16_fase1_*.csv`; la OOF de cada candidato en cada
partición queda en `stores/procesados/nivel7/etapa16/particion_r<r>/predicciones/` para
las fases siguientes.

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
