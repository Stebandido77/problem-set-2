"""Diagnóstico forense del nivel 7: solo lee datos, cachés y entregas existentes y
escribe tablas en views/tablas/nivel7/forense_*.csv. No modifica modelos,
cachés de auxiliares, submissions ni la bitácora.

Es el único script de python/ que lee stores/raw/ (solo lectura): audita los
archivos crudos y su paso a las bases procesadas. Ni los ids ni las
coincidencias entre train y test se usan nunca como predictores.

Secciones (cada una es una función y una tabla):
  estructura    join hogares-personas, unidad de gasto, NA, identidad de la
                etiqueta (Pobre == Ingpcug < Lp) y de sus componentes
  arriendo      Ingtotugarr - Ingtotug frente a la regla de arriendo_imputado_hogar()
  ids           estructura de los ids y del orden de las filas del CSV
  superposicion hogares de test con un perfil exacto o casi exacto en train
  poda          error de podar_columnas() (hash sin orden) en los agregados
  submissions   alineación de cada CSV con el test y con las predicciones guardadas
  adversarial   LightGBM train sin Bogotá vs test con las 1.324 columnas de A y
                el control con solo las 1.306 observadas (sin auxiliares)
  experimentos  clasificador n6 de 1 semilla sobre la base de A: base, + auxiliares
                de indigencia y log(Ingpcug/Lp), sin Bogotá y reponderado por
                los pesos adversariales (completos y estáticos)
Las partes caras (adversarial, auxiliares y experimentos) guardan lo terminado
en stores/procesados/nivel7/forense/ y se retoman desde ahí.

Uso (desde la raíz del repo):
    python/.venv/Scripts/python.exe python/diagnostico_forense.py [--solo seccion,seccion]
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import chi2_contingency, spearmanr
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import funciones.agregados as agregados
from configuracion import N_HILOS, PARAMETROS_N6, RAIZ, RUTA_NIVEL7, RUTA_PROCESADOS, RUTA_TABLAS, SEMILLA
from etapa11_estrato import reconciliacion_arriendo
from funciones.anidado import asignar_grupos, definir_contextos, predecir_anidado, verificar_sin_fuga
from funciones.auxiliares import FUERA_UNIDAD_GASTO
from funciones.bases import BLOQUES_A4
from funciones.bitacora import registrar_log
from funciones.datos import cargar_hogares, cargar_particion, cargar_personas, imputar_mediana, preparar_matriz
from funciones.metricas import buscar_umbral_f1, metricas_por_fold
from funciones.modelado import resultado_desde_predicciones
from funciones.variables import construir_grupos

RUTA_CRUDOS = RAIZ / "stores" / "raw"
RUTA_FORENSE = RUTA_NIVEL7 / "forense"
RUTA_MATRICES_A = RUTA_NIVEL7 / "etapa10" / "matrices"
RUTA_SUBMISSIONS = RAIZ / "stores" / "submissions"
SECCIONES = ["estructura", "arriendo", "ids", "superposicion", "poda", "submissions", "adversarial", "experimentos"]
PARAMETROS_ADVERSARIAL = {"objective": "binary", "num_iterations": 400, "learning_rate": 0.05, "num_leaves": 15,
                          "min_data_in_leaf": 200, "feature_fraction": 0.8, "bagging_fraction": 0.8,
                          "bagging_freq": 1, "verbose": -1}
# Pesos de reponderación: densidad test/train estimada con el clasificador adversarial
RECORTE_PROBABILIDAD = (0.01, 0.99)
RECORTE_PESOS = (0.05, 20)
# Submissions del nivel 7 con predicciones guardadas: candidato -> archivo CSV
SUBMISSIONS_CON_PREDICCIONES = {"n7_a4_checkpoint": "checkpoint_a4", "n7_n6param_a4_total_bag5": "n7_final",
                                "n7_v3_fuentes_bag5": "n7_v3_final", "n7_a_agregados_bag5": "n7_a_agregados_bag5",
                                "n10_mezcla_lgb_realmlp": "n10_mezcla_lgb_realmlp"}


# ---- Utilidades ---------------------------------------------------------------

def cargar_crudos():
    """Las cuatro bases de Kaggle, solo lectura, con id como texto."""
    def leer(nombre):
        return pd.read_csv(RUTA_CRUDOS / f"{nombre}.csv", dtype={"id": str, "Depto": str}, low_memory=False)
    return {n: leer(n) for n in ("train_hogares", "test_hogares", "train_personas", "test_personas")}


def guardar_tabla(nombre, tabla):
    """Escribe views/tablas/nivel7/forense_<nombre>.csv y la devuelve."""
    tabla = pd.DataFrame(tabla)
    tabla.to_csv(RUTA_TABLAS / f"forense_{nombre}.csv", index=False)
    registrar_log("forense", f"tabla forense_{nombre}.csv ({len(tabla)} filas)")
    return tabla


def guardar_json(nombre, objeto):
    """Guarda un resultado intermedio en la caché del diagnóstico."""
    RUTA_FORENSE.mkdir(parents=True, exist_ok=True)
    (RUTA_FORENSE / f"{nombre}.json").write_text(
        json.dumps(objeto, indent=2, ensure_ascii=False,
                   default=lambda v: v.item() if isinstance(v, np.generic) else str(v)), encoding="utf-8")


# ---- 1. Estructura y etiqueta -------------------------------------------------

def auditar_estructura(crudos):
    """Join hogares-personas, unidad de gasto, NA e identidad de la etiqueta.

    Argumentos: crudos = cargar_crudos().
    Devuelve: lista de filas (comprobacion, train, test).
    """
    filas = []

    def anotar(comprobacion, train, test=None):
        filas.append({"comprobacion": comprobacion, "train": train, "test": test})

    por_base = {}
    for base in ("train", "test"):
        h, p = crudos[f"{base}_hogares"], crudos[f"{base}_personas"]
        conteo = p.groupby("id").size().reindex(h["id"], fill_value=0).to_numpy()
        en_ug = p[~p["P6050"].isin(FUERA_UNIDAD_GASTO)].groupby("id").size().reindex(h["id"], fill_value=0).to_numpy()
        jefes = p[p["P6050"] == 1].groupby("id").size().reindex(h["id"], fill_value=0)
        arriendo_estimado = h["P5130"].mask(h["P5130"] < 1000)
        arriendo_pagado = h["P5140"].mask(h["P5140"] < 1000)
        por_base[base] = {
            "hogares": len(h), "personas": len(p),
            "ids de hogar repetidos": int(h["id"].duplicated().sum()),
            "llaves (id, Orden) repetidas": int(p.duplicated(["id", "Orden"]).sum()),
            "personas sin hogar": int((~p["id"].isin(h["id"])).sum()),
            "hogares sin personas": int((conteo == 0).sum()),
            "conteo de personas distinto de Nper": int((conteo != h["Nper"]).sum()),
            "conteo de la unidad de gasto distinto de Npersug": int((en_ug != h["Npersug"]).sum()),
            "hogares sin jefe": int((jefes == 0).sum()),
            "hogares con varios jefes": int((jefes > 1).sum()),
            "personas fuera de la unidad de gasto (P6050 6, 7, 8)": int(p["P6050"].isin(FUERA_UNIDAD_GASTO).sum()),
            "hogares con personas fuera de la unidad de gasto": int(p.loc[p["P6050"].isin(FUERA_UNIDAD_GASTO), "id"].nunique()),
            "geografía de la persona distinta de la del hogar": int(sum(
                p.merge(h[["id", c]], on="id", suffixes=("_p", "_h")).eval(f"{c}_p != {c}_h").sum()
                for c in ("Dominio", "Depto", "Clase"))),
            "hogares sin arriendo estimado ni pagado (>= 1.000 pesos)": int((arriendo_estimado.isna() & arriendo_pagado.isna()).sum()),
            "ocupados sin horas (P6800)": int(((p["Oc"] == 1) & p["P6800"].isna()).sum()),
            "P6210 faltante (tasa)": float(p["P6210"].isna().mean()),
            "P6210s1 faltante (tasa)": float(p["P6210s1"].isna().mean()),
            "P6920 faltante (tasa)": float(p["P6920"].isna().mean()),
            "tasa de pobreza": float(h["Pobre"].mean()) if "Pobre" in h else None,
        }
    for comprobacion in por_base["train"]:
        anotar(comprobacion, por_base["train"][comprobacion], por_base["test"][comprobacion])

    h, p = crudos["train_hogares"], crudos["train_personas"]
    ingreso_ug = p["Ingtot"].fillna(0).where(~p["P6050"].isin(FUERA_UNIDAD_GASTO), 0)
    suma_ug = ingreso_ug.groupby(p["id"]).sum().reindex(h["id"]).to_numpy()
    anotar("Pobre distinto de (Ingpcug < Lp)", int(h["Pobre"].ne((h["Ingpcug"] < h["Lp"]).astype(int)).sum()))
    anotar("Indigente distinto de (Ingpcug < Li)", int(h["Indigente"].ne((h["Ingpcug"] < h["Li"]).astype(int)).sum()))
    anotar("Ingpcug faltante", int(h["Ingpcug"].isna().sum()))
    anotar("Ingpcug igual a cero", int(h["Ingpcug"].eq(0).sum()))
    anotar("Ingpcug exactamente igual a Lp", int(h["Ingpcug"].eq(h["Lp"]).sum()))
    anotar("Ingpcug distinto de Ingtotugarr / Npersug", int((~np.isclose(h["Ingpcug"], h["Ingtotugarr"] / h["Npersug"], atol=0.01)).sum()))
    anotar("suma de Ingtot de la unidad de gasto distinta de Ingtotug", int((~np.isclose(suma_ug, h["Ingtotug"], atol=0.01)).sum()))
    anotar("Ingtotugarr mayor que Ingtotug (hogares con arriendo imputado)", int((h["Ingtotugarr"] - h["Ingtotug"] > 0.01).sum()))
    anotar("Npobres distinto de Pobre x Nper", int(h["Npobres"].ne(h["Pobre"] * h["Nper"]).sum()))
    anotar("Npobres distinto de Pobre x Npersug", int(h["Npobres"].ne(h["Pobre"] * h["Npersug"]).sum()))
    anotar("Nindigentes distinto de Indigente x Nper", int(h["Nindigentes"].ne(h["Indigente"] * h["Nper"]).sum()))
    anotar("columnas de hogares solo en train", ", ".join(sorted(set(h.columns) - set(crudos["test_hogares"].columns))))
    anotar("columnas de personas solo en train (número)", len(set(p.columns) - set(crudos["test_personas"].columns)))
    return filas


# ---- 2. Ids y orden del CSV ----------------------------------------------------

def auditar_ids(crudos):
    """Asociación entre los ids (prefijos, fragmentos numéricos, orden) y la pobreza.

    Argumentos: crudos = cargar_crudos().
    Devuelve: (filas de resumen, tabla de asociaciones).
    """
    h, t = crudos["train_hogares"], crudos["test_hogares"]
    resumen = [
        {"comprobacion": "longitud de los ids", "train": int(h["id"].str.len().nunique() == 1 and h["id"].str.len().iloc[0]),
         "test": int(t["id"].str.len().nunique() == 1 and t["id"].str.len().iloc[0])},
        {"comprobacion": "ids hexadecimales", "train": int(h["id"].str.fullmatch("[0-9a-f]+").sum()),
         "test": int(t["id"].str.fullmatch("[0-9a-f]+").sum())},
        {"comprobacion": "ids comunes a train y test", "train": len(set(h["id"]) & set(t["id"])), "test": None},
        {"comprobacion": "cambios de dominio entre filas consecutivas", "train": int(h["Dominio"].ne(h["Dominio"].shift()).sum()),
         "test": int(t["Dominio"].ne(t["Dominio"].shift()).sum())},
    ]
    asociaciones = []
    for k in (1, 2):
        tabla = pd.crosstab(h["id"].str[:k], h["Pobre"])
        chi, p_valor, _, _ = chi2_contingency(tabla)
        tasas = h.groupby(h["id"].str[:k])["Pobre"].mean()
        asociaciones.append({"tipo": f"prefijo de {k} caracteres", "estadistico": "V de Cramér",
                             "valor": float(np.sqrt(chi / len(h))), "p": float(p_valor),
                             "min_tasa": float(tasas.min()), "max_tasa": float(tasas.max())})
    series = {"orden de las filas del CSV": pd.Series(np.arange(len(h))), "orden lexicográfico del id": h["id"].rank(),
              "id: 8 primeros hexadecimales": h["id"].str[:8].map(lambda s: int(s, 16)),
              "id: 8 hexadecimales del medio": h["id"].str[8:16].map(lambda s: int(s, 16)),
              "id: 8 últimos hexadecimales": h["id"].str[16:].map(lambda s: int(s, 16))}
    for nombre, serie in series.items():
        rho, p_valor = spearmanr(serie, h["Pobre"])
        deciles = pd.qcut(serie, 10, labels=False, duplicates="drop")
        _, p_chi2, _, _ = chi2_contingency(pd.crosstab(deciles, h["Pobre"]))
        tasas = h.groupby(deciles)["Pobre"].mean()
        asociaciones.append({"tipo": nombre, "estadistico": "Spearman", "valor": float(rho), "p": float(p_valor),
                             "p_chi2_deciles": float(p_chi2), "min_tasa": float(tasas.min()), "max_tasa": float(tasas.max())})
    return resumen, asociaciones


# ---- 3. Superposición train-test -----------------------------------------------

def firma_personas(personas, campos):
    """Multiconjunto de personas de cada hogar (tuplas ordenadas de los campos)."""
    q = personas[["id"] + campos].fillna(-999).sort_values(["id"] + campos)
    q["firma"] = list(map(tuple, q[campos].to_numpy()))
    return q.groupby("id", sort=False)["firma"].agg(tuple)


def auditar_superposicion(crudos):
    """Hogares de test con un perfil exacto o casi exacto en train.

    Exacto: geografía, cuartos, dormitorios, tenencia, ambos arriendos, Nper,
    Npersug y Lp del hogar, más el multiconjunto de personas por sexo,
    parentesco, nivel y grado educativo y edad. Casi exacto: mismo bloque sin
    los arriendos ni la edad exactos y con el patrón de NA de los arriendos;
    se permite +/- 1 año por persona, razón máxima de 1,10 en cada arriendo y
    de 1,01 en Lp. Las tasas son de pobreza de las contrapartes de TRAIN.
    Argumentos: crudos = cargar_crudos().
    Devuelve: lista de filas por criterio.
    """
    hogares = pd.concat([crudos["train_hogares"].assign(base="train"), crudos["test_hogares"].assign(base="test")],
                        ignore_index=True)
    campos_persona = ["P6020", "P6050", "P6210", "P6210s1", "P6040"]
    personas = pd.concat([crudos["train_personas"][["id"] + campos_persona],
                          crudos["test_personas"][["id"] + campos_persona]], ignore_index=True)
    campos_hogar = ["Dominio", "Depto", "Clase", "P5000", "P5010", "P5090", "P5130", "P5140", "Nper", "Npersug", "Lp"]
    q = hogares.set_index("id")
    q["composicion"] = firma_personas(personas, campos_persona).reindex(q.index)
    q["firma"] = list(map(tuple, q[campos_hogar + ["composicion"]].fillna(-999).to_numpy()))
    train, test = q[q["base"] == "train"], q[q["base"] == "test"]
    exactos = (test.reset_index()[["id", "firma"]]
               .merge(train.reset_index()[["id", "firma", "Pobre"]], on="firma", suffixes=("_test", "_train")))

    # Casi exactos: bloque sin arriendos ni edad, y vecinos en edad / log arriendos / log Lp
    q["composicion"] = firma_personas(personas, campos_persona[:-1]).reindex(q.index)
    orden = personas.fillna(-999).sort_values(["id"] + campos_persona)
    q["edades"] = orden.groupby("id", sort=False)["P6040"].agg(tuple).reindex(q.index)
    bloque = ["Dominio", "Depto", "Clase", "P5000", "P5010", "P5090", "Nper", "Npersug", "composicion"]
    for arriendo in ("P5130", "P5140"):
        q[f"{arriendo}_na"] = q[arriendo].isna()
        bloque.append(f"{arriendo}_na")
    q["bloque"] = list(map(tuple, q[bloque].fillna(-999).to_numpy()))
    compartidos = set(q.loc[q["base"] == "train", "bloque"]) & set(q.loc[q["base"] == "test", "bloque"])
    q = q[q["bloque"].isin(compartidos)]

    def vectores(v):
        edades = np.array(v["edades"].to_list(), dtype=float)
        extras = np.column_stack([np.log(v[c].fillna(0).clip(lower=1)) / np.log(tolerancia)
                                  for c, tolerancia in (("P5130", 1.1), ("P5140", 1.1), ("Lp", 1.01))])
        return np.column_stack([edades, extras])

    pares = []
    for _, g in q.groupby("bloque", sort=False):
        tr, te = g[g["base"] == "train"], g[g["base"] == "test"]
        if not len(tr) or not len(te):
            continue
        vecinos = cKDTree(vectores(tr)).query_ball_point(vectores(te), r=1.000001, p=np.inf)
        for id_test, indices in zip(te.index, vecinos):
            pares += [(id_test, tr.index[j], tr["Pobre"].iloc[j]) for j in indices]
    cercanos = pd.DataFrame(pares, columns=["id_test", "id_train", "Pobre"])

    def resumir(criterio, v):
        if not len(v):
            return {"criterio": criterio, "hogares_test": 0, "hogares_train": 0, "pares": 0}
        por_test = v.groupby("id_test")["Pobre"].agg(["mean", "nunique"])
        return {"criterio": criterio, "hogares_test": int(v["id_test"].nunique()), "hogares_train": int(v["id_train"].nunique()),
                "pares": len(v), "pobreza_contrapartes_train": float(v.drop_duplicates("id_train")["Pobre"].mean()),
                "pobreza_promedio_por_hogar_test": float(por_test["mean"].mean()),
                "hogares_test_con_contrapartes_discrepantes": int((por_test["nunique"] > 1).sum())}

    ids_exactos = set(exactos["id_test"])
    return [resumir("exacto", exactos),
            resumir("casi exacto, sin los exactos", cercanos[~cercanos["id_test"].isin(ids_exactos)]),
            resumir("casi exacto, con los exactos", cercanos),
            {"criterio": "referencia", "hogares_test": len(test), "hogares_train": len(train),
             "pobreza_contrapartes_train": float(train["Pobre"].mean())}]


# ---- 4. Poda de agregados -----------------------------------------------------

def auditar_poda(personas, hogar):
    """Reproduce la poda de duplicados de construir_agregados() comparando columna
    por columna lo que la huella (suma de hashes, sin orden) declara idéntico.

    Argumentos: personas = cargar_personas(); hogar = cargar_hogares().
    Devuelve: (resumen, tabla de columnas descartadas que no eran duplicadas).
    """
    ruta = RUTA_FORENSE / "poda_agregados.csv"
    if ruta.exists():
        casos = pd.read_csv(ruta)
    else:
        falsos = []

        def inspeccionar(tabla):
            moda = tabla.apply(lambda c: c.value_counts(dropna=False, normalize=True).iloc[0])
            tabla = tabla.loc[:, moda <= agregados.PROPORCION_MAXIMA_MODA]
            huellas = tabla.apply(lambda c: pd.util.hash_pandas_object(c, index=False).sum())
            primera = {}
            for columna, huella in huellas.items():
                if huella not in primera:
                    primera[huella] = columna
                    continue
                conservada = primera[huella]
                distintas = ~(tabla[columna].eq(tabla[conservada]) | (tabla[columna].isna() & tabla[conservada].isna()))
                falsos.append({"descartada": columna, "conservada": conservada, "hogares_distintos": int(distintas.sum())})
            return tabla.loc[:, ~huellas.duplicated()]

        original = agregados.podar_columnas
        agregados.podar_columnas = inspeccionar
        try:
            agregados.construir_agregados(personas, hogar.index)
        finally:
            agregados.podar_columnas = original
        casos = pd.DataFrame(falsos)
        RUTA_FORENSE.mkdir(parents=True, exist_ok=True)
        casos.to_csv(ruta, index=False)
    falsos = casos[casos["hogares_distintos"] > 0]
    resumen = {"columnas descartadas como duplicadas": len(casos), "descartadas que no eran duplicadas": len(falsos)}
    return resumen, falsos


# ---- 5. Submissions -----------------------------------------------------------

def auditar_submissions(particion):
    """Cada CSV de stores/submissions/ frente al test y, cuando hay predicciones
    guardadas, frente a la clase que sale de su probabilidad y umbral OOF.

    Argumentos: particion = cargar_particion() (el test está en el orden del CSV crudo).
    Devuelve: lista de filas por archivo.
    """
    ids_test = particion.loc[particion["base"] == "test", "id"].reset_index(drop=True)
    verdad = particion.set_index("id")["pobre"]
    con_predicciones = {v: k for k, v in SUBMISSIONS_CON_PREDICCIONES.items()}
    filas = []
    for ruta in sorted(RUTA_SUBMISSIONS.rglob("*.csv")):
        if ruta.name == "bitacora_kaggle.csv":
            continue
        sub = pd.read_csv(ruta, dtype={"id": str})
        if "id" not in sub:
            continue
        fila = {"archivo": str(ruta.relative_to(RUTA_SUBMISSIONS)).replace("\\", "/"), "filas": len(sub),
                "ids_repetidos": int(sub["id"].duplicated().sum()), "mismo_conjunto_que_test": set(sub["id"]) == set(ids_test),
                "mismo_orden_que_test": bool(sub["id"].equals(ids_test)), "tasa_pobres": float(sub.iloc[:, 1].mean())}
        candidato = con_predicciones.get(ruta.stem) if ruta.parent.name == "nivel7" else None
        if candidato and (RUTA_NIVEL7 / f"predicciones_{candidato}.parquet").exists():
            resultado = resultado_desde_predicciones(candidato, verdad)
            clase = (resultado["prediccion_test"] >= resultado["umbral"]).astype(int)
            fila.update({"candidato": candidato, "umbral_oof": resultado["umbral"],
                         "clases_distintas_de_las_predicciones": int(sub.set_index("id")["pobre"].ne(clase.reindex(sub["id"]).to_numpy()).sum())})
        filas.append(fila)
    return filas


# ---- 6. Validación adversarial ------------------------------------------------

def cargar_contexto_a(contexto, particion):
    """Matrices de la base de A de un contexto (etapa 10), con comprobaciones de
    que no traen ids, etiqueta ni ingresos y de que ajuste y evaluación no se cruzan."""
    x_ent = pd.read_parquet(RUTA_MATRICES_A / f"{contexto}_ent.parquet")
    y_ent = x_ent.pop("pobre").astype(int)
    x_eval = pd.read_parquet(RUTA_MATRICES_A / f"{contexto}_eval.parquet")
    assert x_ent.index.is_unique and x_eval.index.is_unique and not x_ent.index.isin(x_eval.index).any()
    assert np.array_equal(y_ent.to_numpy(), particion.loc[x_ent.index, "pobre"].astype(int).to_numpy())
    assert not any(c in x_ent for c in ("id", "Pobre", "Indigente", "Ingpcug", "Ingtotug", "Npobres"))
    return x_ent, y_ent, x_eval


def columnas_auxiliares(columnas):
    """Las columnas que salen de un auxiliar de ingreso (predichas, no observadas)."""
    return [c for c in columnas if c.startswith(("ingreso_pred", "log_ingreso_pred")) or c.endswith("_v3")]


def validacion_adversarial(particion, dominio, solo_estaticas=False):
    """LightGBM de 5 folds que distingue train (sin Bogotá) de test con las
    columnas de la base de A; con solo_estaticas se excluyen los 18 auxiliares.

    Argumentos: particion = cargar_particion() indexada por id; dominio = Serie
        id -> dominio; solo_estaticas = control sin auxiliares.
    Devuelve: (Serie p_test por id, resumen) y deja la OOF y las importancias en caché.
    """
    nombre = "adversarial_estaticas" if solo_estaticas else "adversarial"
    ruta = RUTA_FORENSE / f"{nombre}_oof.parquet"
    if ruta.exists():
        return pd.read_parquet(ruta)["p_test"], json.loads((RUTA_FORENSE / f"{nombre}.json").read_text(encoding="utf-8"))
    a, _, b = cargar_contexto_a("test", particion)
    if solo_estaticas:
        columnas = [c for c in a.columns if c not in columnas_auxiliares(a.columns)]
        a, b = a[columnas], b[columnas]
    a = a[dominio.reindex(a.index).ne("BOGOTA")]
    x = pd.concat([a, b])
    y = np.r_[np.zeros(len(a), int), np.ones(len(b), int)]
    del a, b
    gc.collect()
    p, ganancia, aucs = np.zeros(len(x)), np.zeros(x.shape[1]), []
    for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEMILLA).split(x, y), 1):
        inicio = time.time()
        modelo = lgb.train({**PARAMETROS_ADVERSARIAL, "num_threads": N_HILOS, "seed": SEMILLA},
                           lgb.Dataset(x.iloc[tr], label=y[tr]))
        p[va] = modelo.predict(x.iloc[va])
        ganancia += modelo.feature_importance("gain") / 5
        aucs.append(float(roc_auc_score(y[va], p[va])))
        registrar_log("forense", f"{nombre} fold {k} AUC {aucs[-1]:.4f} ({time.time() - inicio:.0f} s)")
        del modelo
        gc.collect()
    importancia = pd.Series(ganancia, index=x.columns).sort_values(ascending=False)
    RUTA_FORENSE.mkdir(parents=True, exist_ok=True)
    importancia.rename("gain").to_csv(RUTA_FORENSE / f"{nombre}_importancias.csv")
    resumen = {"auc_oof": float(roc_auc_score(y, p)), "auc_folds": aucs, "n_train": int((y == 0).sum()),
               "n_test": int((y == 1).sum()), "variables": int(len(importancia))}
    pd.DataFrame({"p_test": p, "es_test": y}, index=x.index).to_parquet(ruta)
    guardar_json(nombre, resumen)
    return pd.Series(p, index=x.index), resumen


def tabla_adversarial(resumenes):
    """Filas de AUC por fold y tabla de las 15 variables con más gain de cada corrida."""
    filas, importancias = [], []
    for nombre, resumen in resumenes.items():
        filas.append({"corrida": nombre, "variables": resumen["variables"], "auc_oof": resumen["auc_oof"],
                      **{f"auc_fold_{k}": v for k, v in enumerate(resumen["auc_folds"], 1)},
                      "n_train_sin_bogota": resumen["n_train"], "n_test": resumen["n_test"]})
        imp = pd.read_csv(RUTA_FORENSE / f"{nombre}_importancias.csv", index_col=0)["gain"]
        top = (imp.head(15) / imp.sum()).reset_index()
        top.columns = ["variable", "proporcion_gain"]
        importancias.append(top.assign(corrida=nombre, orden=np.arange(1, len(top) + 1)))
    return filas, pd.concat(importancias, ignore_index=True)


# ---- 7. Experimentos sobre la base de A --------------------------------------

def auxiliares_forenses(particion, grupos):
    """Auxiliares de indigencia (P(Indigente)) y de log(Ingpcug / Lp) sobre los
    predictores de hogar de a4, con la OOF anidada (sin el contexto test) y los
    parámetros de n6; igual que el auxiliar directo de hogar de la etapa 1.

    Argumentos: particion = cargar_particion() indexada por id; grupos = asignar_grupos().
    Devuelve: {contexto: DataFrame con p_indigente y log_ingpcug_lp_pred}.
    """
    contextos = {k: v for k, v in definir_contextos().items() if k != "test"}
    ruta = RUTA_FORENSE / "auxiliares_hogar.parquet"
    if ruta.exists():
        tabla = pd.read_parquet(ruta)
        return {c: g.drop(columns="contexto") for c, g in tabla.groupby("contexto")}
    hogar, personas = cargar_hogares(), cargar_personas()
    bloques = construir_grupos(personas, hogar)
    x = preparar_matriz(pd.concat([bloques[b] for b in BLOQUES_A4], axis=1))
    del hogar, personas, bloques
    gc.collect()
    objetivos = pd.read_parquet(RUTA_PROCESADOS / "ingreso_hogar_train.parquet").set_index("id")
    objetivos["log_rel"] = np.log(objetivos["Ingpcug"].clip(lower=1) / objetivos["Lp"])
    parametros = {**PARAMETROS_N6, "num_threads": N_HILOS, "seed": SEMILLA, "verbose": -1}

    def ajustar(ids_entrenamiento, ids_evaluacion):
        inicio = time.time()
        a, b = imputar_mediana(x.loc[ids_entrenamiento], x.loc[ids_evaluacion])
        salida = pd.DataFrame(index=ids_evaluacion)
        for columna, objetivo, etiqueta in (("p_indigente", "binary", "Indigente"), ("log_ingpcug_lp_pred", "regression", "log_rel")):
            modelo = lgb.train({**parametros, "objective": objetivo},
                               lgb.Dataset(a, label=objetivos.loc[ids_entrenamiento, etiqueta]))
            salida[columna] = modelo.predict(b)
        registrar_log("forense", f"auxiliar de hogar n={len(a)} ({time.time() - inicio:.0f} s)")
        return salida

    variables, registro = predecir_anidado(ajustar, grupos, "forense", contextos, ruta_parcial=RUTA_FORENSE / "aux_parciales")
    verificar_sin_fuga(registro, grupos, contextos)
    pd.concat([v.assign(contexto=k) for k, v in variables.items()]).to_parquet(ruta)
    registro.to_parquet(RUTA_FORENSE / "registro_auxiliares_hogar.parquet", index=False)
    return variables


def evaluar_experimento(nombre, particion, dominio, auxiliares=None, columnas=(), pesos=None, sin_bogota=False):
    """Clasificador n6 de 1 semilla sobre la base de A en los 6 contextos
    (cv_1..cv_5 y holdout), con variantes: columnas auxiliares extra, pesos por
    hogar y exclusión de Bogotá del ajuste y la evaluación.

    Devuelve: DataFrame de predicciones (probabilidad, tipo, fold) indexado por id.
    """
    ruta = RUTA_FORENSE / f"pred_{nombre}.parquet"
    if ruta.exists():
        return pd.read_parquet(ruta)
    partes = []
    for contexto in [f"cv_{k}" for k in range(1, 6)] + ["holdout"]:
        parcial = RUTA_FORENSE / f"{nombre}_{contexto}.parquet"
        if parcial.exists():
            partes.append(pd.read_parquet(parcial))
            continue
        inicio = time.time()
        a, y, b = cargar_contexto_a(contexto, particion)
        if sin_bogota:
            dentro = dominio.reindex(a.index).ne("BOGOTA")
            a, y = a[dentro], y[dentro]
            b = b[dominio.reindex(b.index).ne("BOGOTA")]
        if columnas:
            a = a.join(auxiliares[contexto][list(columnas)])
            b = b.join(auxiliares[contexto][list(columnas)])
            assert a[list(columnas)].notna().all().all() and b[list(columnas)].notna().all().all()
        peso = None
        if pesos is not None:
            peso = pesos.reindex(a.index).to_numpy()
            assert np.isfinite(peso).all() and (peso > 0).all()
            peso = peso / peso.mean()
        modelo = lgb.train({**PARAMETROS_N6, "num_threads": N_HILOS, "seed": SEMILLA, "verbose": -1},
                           lgb.Dataset(a, label=y, weight=peso))
        pred = pd.DataFrame({"probabilidad": modelo.predict(b), "tipo": "holdout" if contexto == "holdout" else "oof",
                             "fold": np.nan if contexto == "holdout" else int(contexto[-1])}, index=b.index)
        RUTA_FORENSE.mkdir(parents=True, exist_ok=True)
        pred.to_parquet(parcial)
        partes.append(pred)
        registrar_log("forense", f"{nombre} {contexto} n={len(a)}/{len(b)} p={a.shape[1]} ({time.time() - inicio:.0f} s)")
        del a, b, modelo
        gc.collect()
    pred = pd.concat(partes)
    pred.to_parquet(ruta)
    return pred


def resumir_experimento(nombre, pred, particion, dominio):
    """F1 CV (umbral OOF) y holdout del experimento, completo y sin Bogotá."""
    y = particion.loc[pred.index, "pobre"].astype(int)
    oof, holdout = pred["tipo"].eq("oof"), pred["tipo"].eq("holdout")
    fuera_bogota = dominio.reindex(pred.index).ne("BOGOTA")
    umbral, _ = buscar_umbral_f1(y[oof], pred.loc[oof, "probabilidad"])
    umbral_sb, _ = buscar_umbral_f1(y[oof & fuera_bogota], pred.loc[oof & fuera_bogota, "probabilidad"])
    return {"experimento": nombre, "umbral": umbral,
            "f1_cv": metricas_por_fold(y[oof], pred.loc[oof, "probabilidad"], pred.loc[oof, "fold"], umbral)["f1_cv"],
            "f1_holdout": float(f1_score(y[holdout], pred.loc[holdout, "probabilidad"] >= umbral)),
            "umbral_sin_bogota": umbral_sb,
            "f1_cv_sin_bogota": metricas_por_fold(y[oof & fuera_bogota], pred.loc[oof & fuera_bogota, "probabilidad"],
                                                  pred.loc[oof & fuera_bogota, "fold"], umbral_sb)["f1_cv"],
            "f1_holdout_sin_bogota": float(f1_score(y[holdout & fuera_bogota],
                                                    pred.loc[holdout & fuera_bogota, "probabilidad"] >= umbral_sb)),
            "n_oof": int(oof.sum()), "n_holdout": int(holdout.sum())}


def diferencia_pareada(base, otro, particion, dominio, sin_bogota, semilla=SEMILLA, remuestras=5000):
    """Diferencia de F1 en holdout entre dos experimentos sobre los mismos hogares,
    con intervalo bootstrap multinomial de las 8 celdas (verdad x clase_a x clase_b)."""
    a = base["pred"][base["pred"]["tipo"].eq("holdout")]
    if sin_bogota:
        a = a[dominio.reindex(a.index).ne("BOGOTA")]
    b = otro["pred"].loc[a.index]
    llave = "umbral_sin_bogota" if sin_bogota else "umbral"
    y = particion.loc[a.index, "pobre"].astype(int).to_numpy()
    clase_a = (a["probabilidad"] >= base["resumen"][llave]).astype(int).to_numpy()
    clase_b = (b["probabilidad"] >= otro["resumen"][llave]).astype(int).to_numpy()
    frecuencias = np.bincount(4 * y + 2 * clase_a + clase_b, minlength=8)
    m = np.random.default_rng(semilla).multinomial(len(y), frecuencias / len(y), size=remuestras)
    f1_a = 2 * (m[:, 6] + m[:, 7]) / (2 * (m[:, 6] + m[:, 7]) + m[:, 2] + m[:, 3] + m[:, 4] + m[:, 5])
    f1_b = 2 * (m[:, 5] + m[:, 7]) / (2 * (m[:, 5] + m[:, 7]) + m[:, 1] + m[:, 3] + m[:, 4] + m[:, 6])
    inferior, superior = np.quantile(f1_b - f1_a, [0.025, 0.975])
    return {"base": base["resumen"]["experimento"], "comparacion": otro["resumen"]["experimento"], "sin_bogota": sin_bogota,
            "hogares": len(y), "f1_base": float(f1_score(y, clase_a)), "f1_comparacion": float(f1_score(y, clase_b)),
            "delta": float(f1_score(y, clase_b) - f1_score(y, clase_a)), "ic95_inferior": float(inferior),
            "ic95_superior": float(superior), "predicciones_distintas": int((clase_a != clase_b).sum())}


def pesos_adversariales(p_test, resumen, ids):
    """Pesos densidad test/train: odds del clasificador adversarial por el cociente
    de tamaños, recortados y con media uno sobre los hogares de ajuste."""
    prob = p_test.clip(*RECORTE_PROBABILIDAD)
    pesos = (prob / (1 - prob) * (resumen["n_train"] / resumen["n_test"])).clip(*RECORTE_PESOS)
    pesos = pesos.reindex(ids)
    return pesos / pesos.mean()


def correr_experimentos(particion, dominio, grupos, adversarial, adversarial_estatico):
    """Los cinco experimentos sobre la base de A y sus diferencias pareadas.

    Devuelve: (tabla de experimentos, tabla de diferencias pareadas, resumen de los pesos).
    """
    auxiliares = auxiliares_forenses(particion, grupos)
    ids_ajuste = particion.index[(particion["base"] == "train") & dominio.reindex(particion.index).ne("BOGOTA")]
    pesos = pesos_adversariales(*adversarial, ids_ajuste)
    pesos_estaticos = pesos_adversariales(*adversarial_estatico, ids_ajuste)
    disenos = {
        "base": {},
        "mas_indigente_y_log_ingpcug": {"auxiliares": auxiliares, "columnas": ("p_indigente", "log_ingpcug_lp_pred")},
        "base_sin_bogota": {"sin_bogota": True},
        "reponderado_sin_bogota": {"sin_bogota": True, "pesos": pesos},
        "reponderado_estatico_sin_bogota": {"sin_bogota": True, "pesos": pesos_estaticos},
    }
    # Los nombres de caché de las dos primeras corridas vienen del diagnóstico original
    alias = {"mas_indigente_y_log_ingpcug": "mas_ambos"}
    resultados = {}
    for nombre, diseno in disenos.items():
        pred = evaluar_experimento(alias.get(nombre, nombre), particion, dominio, **diseno)
        resultados[nombre] = {"pred": pred, "resumen": resumir_experimento(nombre, pred, particion, dominio)}
        registrar_log("forense", f"{nombre}: F1 CV {resultados[nombre]['resumen']['f1_cv']:.4f} | "
                                 f"holdout {resultados[nombre]['resumen']['f1_holdout']:.4f}")
    pares = [("base", "mas_indigente_y_log_ingpcug", False), ("base", "mas_indigente_y_log_ingpcug", True),
             ("base_sin_bogota", "reponderado_sin_bogota", True), ("base_sin_bogota", "reponderado_estatico_sin_bogota", True)]
    diferencias = [diferencia_pareada(resultados[a], resultados[b], particion, dominio, sb) for a, b, sb in pares]
    descripcion = pesos.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99])
    resumen_pesos = [{"pesos": "completos (con auxiliares)", **descripcion.to_dict(),
                      "tamano_efectivo": float(pesos.sum() ** 2 / (pesos ** 2).sum())},
                     {"pesos": "estáticos (sin auxiliares)", **pesos_estaticos.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).to_dict(),
                      "tamano_efectivo": float(pesos_estaticos.sum() ** 2 / (pesos_estaticos ** 2).sum())}]
    return [r["resumen"] for r in resultados.values()], diferencias, resumen_pesos


# ---- Orquestación -------------------------------------------------------------

def main(secciones):
    inicio = time.time()
    registrar_log("forense", f"inicio: {', '.join(secciones)}")
    particion = cargar_particion()
    resumen = []
    crudos = None
    if {"estructura", "arriendo", "ids", "superposicion"} & set(secciones):
        crudos = cargar_crudos()

    if "estructura" in secciones:
        resumen += [{"seccion": "estructura", **f} for f in auditar_estructura(crudos)]
    if "arriendo" in secciones:
        guardar_tabla("arriendo_imputado", reconciliacion_arriendo(cargar_hogares()))
    if "ids" in secciones:
        filas, asociaciones = auditar_ids(crudos)
        resumen += [{"seccion": "ids", **f} for f in filas]
        guardar_tabla("ids", asociaciones)
    if "superposicion" in secciones:
        guardar_tabla("superposicion", auditar_superposicion(crudos))
    del crudos
    gc.collect()

    if "poda" in secciones:
        resumen_poda, falsos = auditar_poda(cargar_personas(), cargar_hogares())
        resumen += [{"seccion": "poda", "comprobacion": k, "train": v} for k, v in resumen_poda.items()]
        guardar_tabla("poda_agregados", falsos)
    if "submissions" in secciones:
        guardar_tabla("submissions", auditar_submissions(particion))

    if {"adversarial", "experimentos"} & set(secciones):
        tabla = particion.set_index("id")
        dominio = cargar_hogares()["dominio"].astype(str)
        completo = validacion_adversarial(tabla, dominio)
        estatico = validacion_adversarial(tabla, dominio, solo_estaticas=True)
        filas, importancias = tabla_adversarial({"adversarial": completo[1], "adversarial_estaticas": estatico[1]})
        guardar_tabla("adversarial", filas)
        guardar_tabla("adversarial_importancia", importancias)
        if "experimentos" in secciones:
            experimentos, diferencias, pesos = correr_experimentos(tabla, dominio, asignar_grupos(particion), completo, estatico)
            guardar_tabla("experimentos", experimentos)
            guardar_tabla("diferencias_pareadas", diferencias)
            guardar_tabla("pesos_adversariales", pesos)

    if resumen:
        anterior = RUTA_TABLAS / "forense_resumen.csv"
        if anterior.exists() and set(secciones) != set(SECCIONES):
            previo = pd.read_csv(anterior)
            resumen = pd.concat([previo[~previo["seccion"].isin({f["seccion"] for f in resumen})], pd.DataFrame(resumen)])
        guardar_tabla("resumen", resumen)
    registrar_log("forense", f"fin ({(time.time() - inicio) / 60:.1f} min)")


if __name__ == "__main__":
    analizador = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    analizador.add_argument("--solo", default=",".join(SECCIONES), help="secciones separadas por coma")
    main([s for s in analizador.parse_args().solo.split(",") if s in SECCIONES])
