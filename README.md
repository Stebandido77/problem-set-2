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
│   ├── etapa1_ablacion.py ... etapa10b_submissions.py
│   ├── etapa11_estrato.py            # Estrato como auxiliar y reconciliación del arriendo
│   ├── etapa12_enfoques.py           # Cuantiles, DeepSets, GPBoost, KNN fuera de fold
│   ├── etapa12_autogluon.py          # AutoGluon (entorno aparte: requirements_autogluon.txt)
│   ├── etapa13_mezcla_final.py       # Mezcla final desde predicciones guardadas (sin reentrenar)
│   ├── diagnostico_forense.py        # Auditoría de datos, fugas, submissions y shift
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
python/.venv/Scripts/python.exe python/etapa11_estrato.py
python/.venv/Scripts/python.exe python/diagnostico_forense.py
python/.venv/Scripts/python.exe python/etapa12_enfoques.py preparar
python/.venv_autogluon/Scripts/python.exe python/etapa12_autogluon.py --corrida A --minutos 40
python/.venv_autogluon/Scripts/python.exe python/etapa12_autogluon.py --corrida B --minutos 55
for e in distribucional deepsets gpboost knn cierre; do python/.venv/Scripts/python.exe python/etapa12_enfoques.py $e; done
python/.venv/Scripts/python.exe python/etapa13_mezcla_final.py
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
