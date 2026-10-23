"""Pruebas de la etapa 16: particiones repetidas estratificadas que no tocan el
holdout, contextos solo de CV, auxiliares anidados y OOF reanudables por carpeta,
inferencia (CV repetida, comparaciones pareadas, bootstrap del holdout, MDE) y el
orquestador de trabajos (DAG, dos procesos, memoria)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from funciones import bitacora
from funciones.anidado import asignar_grupos
from funciones.bitacora import registrar_log
from funciones.repeticion import definir_contextos_cv, generar_particion_repetida


@pytest.fixture(autouse=True)
def log_en_carpeta_temporal(tmp_path, monkeypatch):
    """Las pruebas no escriben en stores/logs/nivel7.log."""
    monkeypatch.setattr(bitacora, "ARCHIVO_LOG", tmp_path / "nivel7_prueba.log")
    monkeypatch.setattr(bitacora, "ARCHIVOS_LOG_EXTRA", [])


def particion_sintetica(n_por_grupo=200, semilla=0):
    """Partición de juguete como la de R: 5 folds, holdout (6) y test (0), con
    una etiqueta real 0/1 (tasa ~0,2) en train."""
    rng = np.random.default_rng(semilla)
    filas = []
    for g in range(7):
        for i in range(n_por_grupo):
            filas.append({
                "id": f"h{g}_{i}", "pobre": (float(rng.random() < 0.2) if g else np.nan),
                "fold_id": g if 1 <= g <= 5 else pd.NA, "holdout": g == 6,
                "base": "test" if g == 0 else "train",
            })
    particion = pd.DataFrame(filas)
    particion["fold_id"] = particion["fold_id"].astype("Int64")
    return particion


# ---- Tarea 1: log por fase y particiones repetidas --------------------------------

def test_particion_repetida_conserva_holdout_y_test_y_estratifica():
    particion = particion_sintetica()
    grupos = generar_particion_repetida(particion, 7)
    original = asignar_grupos(particion)
    assert grupos.index.equals(original.index)
    assert (grupos[original == 6] == 6).all() and (grupos[original == 0] == 0).all()
    en_80 = original.isin([1, 2, 3, 4, 5])
    assert set(grupos[en_80]) == {1, 2, 3, 4, 5}
    conteos = grupos[en_80].value_counts()
    assert conteos.max() - conteos.min() <= 1
    tasas = particion.set_index("id")["pobre"].groupby(grupos).mean()
    assert tasas.loc[[1, 2, 3, 4, 5]].max() - tasas.loc[[1, 2, 3, 4, 5]].min() < 0.03
    # Los folds nuevos no son los de R
    assert (grupos[en_80] != original[en_80]).mean() > 0.5


def test_particion_repetida_es_determinista_y_cambia_con_la_semilla():
    particion = particion_sintetica()
    assert generar_particion_repetida(particion, 7).equals(generar_particion_repetida(particion, 7))
    assert not generar_particion_repetida(particion, 7).equals(generar_particion_repetida(particion, 8))


def test_contextos_cv_no_incluyen_holdout_ni_test():
    contextos = definir_contextos_cv()
    assert list(contextos) == [f"cv_{k}" for k in range(1, 6)]
    for k, (entrenamiento, evaluacion) in enumerate(contextos.values(), 1):
        assert evaluacion == (k,) and k not in entrenamiento
        assert set(entrenamiento) | {k} == {1, 2, 3, 4, 5}


def test_registrar_log_escribe_tambien_en_los_archivos_extra(tmp_path, monkeypatch):
    monkeypatch.setattr(bitacora, "ARCHIVO_LOG", tmp_path / "nivel7.log")
    monkeypatch.setattr(bitacora, "ARCHIVOS_LOG_EXTRA", [tmp_path / "fase.log"])
    registrar_log("prueba", "evento de prueba")
    assert "evento de prueba" in (tmp_path / "nivel7.log").read_text(encoding="utf-8")
    assert "evento de prueba" in (tmp_path / "fase.log").read_text(encoding="utf-8")


# ---- Tarea 2: auxiliares anidados y OOF reanudable por carpeta --------------------

def crear_espia(llamadas):
    """Auxiliar 'espía': 1 si el hogar a predecir estuvo en el entrenamiento; cuenta llamadas."""
    def espia(ids_entrenamiento, ids_prediccion):
        llamadas.append(1)
        vistos = set(ids_entrenamiento)
        return pd.DataFrame({"visto": [float(i in vistos) for i in ids_prediccion]},
                            index=pd.Index(ids_prediccion, name="id"))
    return espia


def test_anidado_en_carpeta_sin_fuga_y_con_cache(tmp_path):
    from funciones.repeticion import anidado_en_carpeta
    grupos = generar_particion_repetida(particion_sintetica(), 3)
    llamadas = []
    variables = anidado_en_carpeta("espia", crear_espia(llamadas), grupos, tmp_path, definir_contextos_cv())
    assert set(variables) == {f"cv_{k}" for k in range(1, 6)}
    assert all((v["visto"] == 0).all() for v in variables.values())
    # Cada contexto cubre todo el 80 % (los 5 folds), una vez
    assert all(len(v) == grupos.isin([1, 2, 3, 4, 5]).sum() and v.index.is_unique for v in variables.values())
    assert len(llamadas) == 15  # 5 ajustes con 4 folds (evaluación) + 10 con 3 folds (OOF interna)
    assert (tmp_path / "anidado_espia.parquet").exists() and (tmp_path / "registro_espia.parquet").exists()
    assert not (tmp_path / "parciales" / "espia").exists()
    otra_vez = anidado_en_carpeta("espia", crear_espia(llamadas), grupos, tmp_path, definir_contextos_cv())
    assert len(llamadas) == 15
    pd.testing.assert_frame_equal(otra_vez["cv_3"], variables["cv_3"])


def matrices_de_juguete(grupos, semilla=0):
    """Matriz de juguete indexada por id con una variable correlacionada con pobre."""
    rng = np.random.default_rng(semilla)
    x = pd.DataFrame({"v1": rng.normal(size=len(grupos)), "v2": rng.normal(size=len(grupos))}, index=grupos.index)
    return x


def crear_preparador_juguete(grupos, x):
    contextos = definir_contextos_cv()
    verdad = pd.Series((x["v1"] > 0.3).astype(int), index=x.index)

    def preparar(contexto):
        entrenamiento, evaluacion = contextos[contexto]
        ids_ent, ids_eval = grupos.index[grupos.isin(entrenamiento)], grupos.index[grupos.isin(evaluacion)]
        return x.loc[ids_ent], verdad.loc[ids_ent].to_numpy(), x.loc[ids_eval]

    return preparar, verdad


def ajustar_juguete(x_ent, y_ent, x_eval):
    return (x_eval["v1"] > 0.3).astype(float).to_numpy() * 0.8 + 0.1, None


def test_evaluar_oof_es_reanudable_y_cubre_el_80_una_vez(tmp_path):
    from funciones.repeticion import cargar_oof, evaluar_oof
    grupos = generar_particion_repetida(particion_sintetica(), 5)
    preparar, verdad = crear_preparador_juguete(grupos, matrices_de_juguete(grupos))
    llamadas = []

    def ajustar_que_falla(x_ent, y_ent, x_eval):
        llamadas.append(1)
        if len(llamadas) == 3:
            raise RuntimeError("corte simulado")
        return ajustar_juguete(x_ent, y_ent, x_eval)

    with pytest.raises(RuntimeError):
        evaluar_oof("cand", preparar, ajustar_que_falla, verdad, tmp_path)
    assert len(list((tmp_path / "parciales" / "cand").glob("cv_*.parquet"))) == 2
    assert cargar_oof("cand", tmp_path) is None
    oof = evaluar_oof("cand", preparar, ajustar_juguete, verdad, tmp_path)
    assert list(oof.columns) == ["probabilidad", "fold", "verdad"]
    assert sorted(oof["fold"].unique()) == [1, 2, 3, 4, 5] and oof.index.is_unique
    assert len(oof) == grupos.isin([1, 2, 3, 4, 5]).sum()
    assert (oof["verdad"] == verdad.loc[oof.index]).all()
    assert (oof["fold"] == grupos.loc[oof.index]).all()
    pd.testing.assert_frame_equal(cargar_oof("cand", tmp_path), oof)
    assert not (tmp_path / "parciales" / "cand").exists()


def test_promediar_oof_pondera_y_alinea_por_id(tmp_path):
    from funciones.repeticion import cargar_oof, evaluar_oof, promediar_oof
    grupos = generar_particion_repetida(particion_sintetica(), 5)
    preparar, verdad = crear_preparador_juguete(grupos, matrices_de_juguete(grupos))
    a = evaluar_oof("a", preparar, ajustar_juguete, verdad, tmp_path)
    b = evaluar_oof("b", preparar, lambda x_ent, y_ent, x_eval: (np.full(len(x_eval), 0.5), None), verdad, tmp_path)
    # Se guarda b con las filas en otro orden para comprobar la alineación por id
    (tmp_path / "predicciones" / "b.parquet").unlink()
    b.iloc[::-1].to_parquet(tmp_path / "predicciones" / "b.parquet")
    mezcla = promediar_oof("m", ["a", "b"], [0.8, 0.2], tmp_path)
    esperado = 0.8 * a["probabilidad"] + 0.2 * 0.5
    assert np.allclose(mezcla["probabilidad"].loc[a.index], esperado)
    assert (mezcla["fold"].loc[a.index] == a["fold"]).all() and (mezcla["verdad"].loc[a.index] == a["verdad"]).all()
    pd.testing.assert_frame_equal(cargar_oof("m", tmp_path), mezcla)


# ---- Tarea 3: inferencia -----------------------------------------------------------

def oof_de_juguete(semilla, n=1000, ruido=0.3):
    """OOF sintética de 5 folds con una probabilidad informativa."""
    rng = np.random.default_rng(semilla)
    verdad = (rng.random(n) < 0.2).astype(np.int64)
    probabilidad = np.clip(0.5 * verdad + rng.normal(scale=ruido, size=n) + 0.2, 0, 1)
    return pd.DataFrame({"probabilidad": probabilidad, "fold": np.tile(np.arange(1, 6), n // 5), "verdad": verdad},
                        index=pd.Index([f"h{i}" for i in range(n)], name="id"))


def test_f1_por_fold_usa_un_solo_umbral_para_toda_la_oof():
    from funciones.inferencia import f1_por_fold_con_umbral_oof
    from funciones.metricas import buscar_umbral_f1, calcular_metricas
    oof = oof_de_juguete(0)
    folds = f1_por_fold_con_umbral_oof(oof)
    assert list(folds.columns) == ["fold", "umbral", "f1", "precision", "recall"]
    assert list(folds["fold"]) == [1, 2, 3, 4, 5] and folds["umbral"].nunique() == 1
    umbral, _ = buscar_umbral_f1(oof["verdad"], oof["probabilidad"])
    fold_2 = oof[oof["fold"] == 2]
    esperado = calcular_metricas(fold_2["verdad"], (fold_2["probabilidad"] >= umbral).astype(int))["f1"]
    assert folds.set_index("fold").loc[2, "f1"] == pytest.approx(esperado)


def test_resumir_cv_repetida_media_sd_y_error_estandar():
    from funciones.inferencia import resumir_cv_repetida
    folds = pd.DataFrame({"particion": [1] * 5 + [2] * 5 + [3] * 5, "fold": list(range(1, 6)) * 3,
                          "f1": np.linspace(0.70, 0.84, 15)})
    r = resumir_cv_repetida(folds)
    assert r["n_evaluaciones"] == 15 and r["f1_medio"] == pytest.approx(0.77)
    assert r["f1_sd"] == pytest.approx(folds["f1"].std(ddof=1))
    assert r["f1_ee"] == pytest.approx(folds["f1"].std(ddof=1) / np.sqrt(15))


def test_comparar_pareado_intervalo_t_y_mde():
    from scipy.stats import t
    from funciones.inferencia import comparar_pareado
    rng = np.random.default_rng(1)
    base = pd.DataFrame({"particion": [1] * 5 + [2] * 5 + [3] * 5, "fold": list(range(1, 6)) * 3, "f1": 0.75})
    candidato = base.assign(f1=base["f1"] + 0.004 + rng.normal(scale=0.002, size=15))
    r = comparar_pareado(candidato.sample(frac=1, random_state=2), base)  # en otro orden: se une por partición y fold
    assert r["n"] == 15 and r["ic_inferior"] < r["delta_medio"] < r["ic_superior"]
    assert r["delta_medio"] == pytest.approx((candidato["f1"] - base["f1"]).mean())
    assert r["ic_superior"] - r["delta_medio"] == pytest.approx(t.ppf(0.975, 14) * r["delta_ee"])
    assert r["mde"] == pytest.approx((t.ppf(0.975, 14) + t.ppf(0.80, 14)) * r["delta_ee"])
    assert r["excluye_cero"] is True
    identico = comparar_pareado(base, base)
    assert identico["delta_medio"] == 0 and identico["excluye_cero"] is False
    incompleto = comparar_pareado(candidato.iloc[:12], base)
    assert incompleto["n"] == 12


def test_bootstrap_holdout_es_determinista_y_el_candidato_identico_tiene_delta_cero():
    from funciones.inferencia import MDE_Z, bootstrap_holdout
    rng = np.random.default_rng(3)
    ids = pd.Index([f"h{i}" for i in range(500)], name="id")
    verdad = pd.Series((rng.random(500) < 0.2).astype(int), index=ids)
    base = pd.Series(np.where(rng.random(500) < 0.8, verdad, 1 - verdad), index=ids)
    candidato = base.copy()
    candidato.iloc[:20] = verdad.iloc[:20]  # corrige 20 hogares
    clases = {"base": base, "cand": candidato, "igual": base.iloc[::-1]}
    a = bootstrap_holdout(verdad, clases, "base", n_replicas=200, semilla=1)
    b = bootstrap_holdout(verdad, clases, "base", n_replicas=200, semilla=1)
    pd.testing.assert_frame_equal(a, b)
    assert list(a["candidato"]) == ["base", "cand", "igual"]
    igual = a.set_index("candidato").loc["igual"]
    assert igual["delta_medio"] == 0 and igual["delta_ic_inferior"] == 0 == igual["delta_ic_superior"]
    assert igual["delta_excluye_cero"] == False  # noqa: E712 (valor numpy)
    assert (a["ic_inferior"] <= a["f1"]).all() and (a["f1"] <= a["ic_superior"]).all()
    cand = a.set_index("candidato").loc["cand"]
    assert cand["delta_medio"] > 0 and cand["mde_holdout"] == pytest.approx(MDE_Z * cand["delta_sd"])
    perfecto = bootstrap_holdout(verdad, {"base": verdad, "p": verdad}, "base", n_replicas=50, semilla=1)
    assert (perfecto["f1"] == 1).all() and (perfecto["ic_inferior"] == 1).all()


def test_columnas_v3_y_a4_total_quitan_agregados_y_v3():
    from funciones.repeticion import columnas_a4_total, columnas_v3
    columnas = ["n_personas", "ag_jefe_P6040_valor", "ingreso_pred_total", "laboral_pc_lp_v3", "n_perceptores_v3",
                "ingreso_pred_total_total", "ag_todos_P6040_media"]
    assert columnas_v3(columnas) == ["n_personas", "ingreso_pred_total", "laboral_pc_lp_v3", "n_perceptores_v3",
                                     "ingreso_pred_total_total"]
    assert columnas_a4_total(columnas) == ["n_personas", "ingreso_pred_total", "ingreso_pred_total_total"]


# ---- Tareas 4 a 6: etapa16_evaluacion (nombres, mezclas fijas, KNN por carpeta, orquestador) ----

def test_carpetas_nombres_y_miniatura(monkeypatch):
    import etapa16_evaluacion as etapa16
    from funciones import auxiliares
    for atributo in ("PRUEBA", "PREFIJO", "CARPETA", "TABLAS", "PARTICIONES"):
        monkeypatch.setattr(etapa16, atributo, getattr(etapa16, atributo))
    monkeypatch.setattr(etapa16, "PARAMETROS", dict(etapa16.PARAMETROS))
    monkeypatch.setattr(auxiliares, "PARAMETROS_AUXILIAR", auxiliares.PARAMETROS_AUXILIAR)
    assert etapa16.carpeta_particion(2).name == "particion_r2" and etapa16.carpeta_particion(2).parent == etapa16.CARPETA
    assert etapa16.ruta_matrices(1) == etapa16.carpeta_particion(1) / "matrices"
    assert etapa16.nombre_candidato(2, "a_bag5") == "n16_r2_a_bag5"
    assert etapa16.PARTICIONES == (1, 2, 3) and etapa16.PARAMETROS["replicas"] == 2000
    etapa16.configurar_prueba()
    assert etapa16.PRUEBA and etapa16.CARPETA.name == "etapa16_prueba" and etapa16.PARTICIONES == (1,)
    assert etapa16.nombre_candidato(1, "a_bag5") == "n16p_r1_a_bag5" and etapa16.TABLAS == "etapa16p"
    assert etapa16.PARAMETROS["lgb"]["num_iterations"] == 20 and etapa16.PARAMETROS["catboost"]["iterations"] == 20
    assert auxiliares.PARAMETROS_AUXILIAR["num_iterations"] == 20 and etapa16.PARAMETROS["replicas"] == 50
    assert len(etapa16.PARAMETROS["semillas_bagging"]) == 2


def test_candidatos_y_pesos_de_las_mezclas_fijas_siguen_la_spec():
    from etapa16_evaluacion import BASE, CANDIDATOS, MEZCLAS
    assert MEZCLAS["mezcla_final"] == (("a_bag5", 0.80), ("realmlp_bag5", 0.15), ("conjuntos_bag3", 0.05))
    assert MEZCLAS["mezcla_lgb_realmlp"] == (("a_bag5", 0.85), ("realmlp_bag5", 0.15))
    for componentes in MEZCLAS.values():
        assert sum(w for _, w in componentes) == pytest.approx(1.0)
    assert BASE == "n7_a_agregados_bag5" and CANDIDATOS[BASE] == ("a_bag5", "n7_a_agregados_bag5")
    assert len(CANDIDATOS) == 7 and CANDIDATOS["CatBoost B3"] == ("catboost_b3", "n7_b3_catboost_agr")
    assert CANDIDATOS["n13_mezcla_final"] == ("mezcla_final", "n13_mezcla_final")


def test_variables_knn_contexto_acepta_otra_carpeta_de_matrices(tmp_path, monkeypatch):
    import etapa12_enfoques as etapa12
    monkeypatch.setattr(etapa12, "K_VECINOS", (2, 3, 4))
    monkeypatch.setattr(etapa12, "FOLDS_INTERNOS_KNN", 2)
    rng = np.random.default_rng(0)
    x_ent = pd.DataFrame(rng.normal(size=(60, 3)), columns=["a", "b", "c"],
                         index=pd.Index([f"e{i}" for i in range(60)], name="id"))
    x_ent["pobre"] = (x_ent["a"] > 0.5).astype(int)
    x_eval = pd.DataFrame(rng.normal(size=(20, 3)), columns=["a", "b", "c"],
                          index=pd.Index([f"v{i}" for i in range(20)], name="id"))
    x_ent.to_parquet(tmp_path / "cv_1_ent.parquet")
    x_eval.to_parquet(tmp_path / "cv_1_eval.parquet")
    tabla = etapa12.variables_knn_contexto("cv_1", ["a", "b", "c"], ruta_matrices=tmp_path)
    assert list(tabla.columns) == ["tasa_pobre_knn_2", "tasa_pobre_knn_3", "tasa_pobre_knn_4"]
    assert len(tabla) == 80 and tabla.notna().all().all() and ((tabla >= 0) & (tabla <= 1)).all().all()
    assert set(tabla.index) == set(x_ent.index) | set(x_eval.index)


def test_plan_fase1_tiene_dependencias_correctas():
    from etapa16_evaluacion import planificar_fase1
    plan = {t["nombre"]: t for t in planificar_fase1((1, 2))}
    assert plan["arboles_r1"]["depende_de"] == ["matrices_r1"] and plan["catboost_r1"]["depende_de"] == ["matrices_r1"]
    assert plan["redes_r2"]["depende_de"] == ["matrices_r2"] and plan["redes_r2"]["particion"] == 2
    assert plan["knn_original"]["depende_de"] == [] and plan["matrices_r2"]["depende_de"] == []
    assert set(plan["reporte"]["depende_de"]) == set(plan) - {"reporte"}
    assert list(plan)[:3] == ["matrices_r1", "knn_original", "arboles_r1"]


def test_elegir_lanzables_respeta_maximo_memoria_y_modo_solo():
    from etapa16_evaluacion import elegir_lanzables, planificar_fase1
    trabajos = planificar_fase1((1,))
    assert elegir_lanzables(trabajos, {}, activos=[], libre_gb=9) == ["matrices_r1", "knn_original"]
    estado = {"matrices_r1": "terminado", "knn_original": "terminado"}
    assert elegir_lanzables(trabajos, estado, activos=[], libre_gb=9) == ["arboles_r1", "redes_r1"]
    assert elegir_lanzables(trabajos, estado, activos=[], libre_gb=3) == ["arboles_r1"]
    corriendo = {**estado, "arboles_r1": "corriendo"}
    assert elegir_lanzables(trabajos, corriendo, activos=["arboles_r1"], libre_gb=3) == []
    assert elegir_lanzables(trabajos, corriendo, activos=["arboles_r1"], libre_gb=9) == ["redes_r1"]
    assert elegir_lanzables(trabajos, {**estado, "arboles_r1": "solo"}, activos=["arboles_r1"], libre_gb=9) == []
    dos = {**estado, "arboles_r1": "corriendo", "redes_r1": "corriendo"}
    assert elegir_lanzables(trabajos, dos, activos=["arboles_r1", "redes_r1"], libre_gb=9) == []
    casi = {t["nombre"]: "terminado" for t in trabajos if t["nombre"] != "reporte"}
    assert elegir_lanzables(trabajos, casi, activos=[], libre_gb=9) == ["reporte"]
    assert elegir_lanzables(trabajos, {**casi, "catboost_r1": "fallido"}, activos=[], libre_gb=9) == []


def test_columnas_sin_indice_omite_la_columna_id_del_parquet(tmp_path):
    from etapa16_evaluacion import columnas_sin_indice
    tabla = pd.DataFrame({"b": [1.0, 2.0], "a": [3.0, 4.0]}, index=pd.Index(["h1", "h2"], name="id"))
    tabla.to_parquet(tmp_path / "cv_1_eval.parquet")
    assert columnas_sin_indice(tmp_path / "cv_1_eval.parquet") == ["b", "a"]


# ---- Ajustes de la revisión: reanudación, reporte, bootstrap, orquestador ----------

def test_comparar_pareado_reporta_el_intervalo_corregido_por_dependencia():
    from scipy.stats import t
    from funciones.inferencia import comparar_pareado
    rng = np.random.default_rng(4)
    base = pd.DataFrame({"particion": [1] * 5 + [2] * 5 + [3] * 5, "fold": list(range(1, 6)) * 3, "f1": 0.75})
    candidato = base.assign(f1=base["f1"] + 0.004 + rng.normal(scale=0.002, size=15))
    r = comparar_pareado(candidato, base)
    # Nadeau y Bengio (2003): varianza x (1/n + n_prueba/n_entrenamiento), con 1/4 en 5 folds
    assert r["delta_ee_corregido"] == pytest.approx(r["delta_sd"] * np.sqrt(1 / 15 + 0.25))
    assert r["ic_superior_corregido"] - r["delta_medio"] == pytest.approx(t.ppf(0.975, 14) * r["delta_ee_corregido"])
    assert r["mde_corregido"] == pytest.approx((t.ppf(0.975, 14) + t.ppf(0.80, 14)) * r["delta_ee_corregido"])
    assert r["ic_superior_corregido"] > r["ic_superior"] and isinstance(r["excluye_cero_corregido"], bool)


def test_bootstrap_holdout_reporta_la_diferencia_observada():
    from funciones.inferencia import bootstrap_holdout
    rng = np.random.default_rng(5)
    ids = pd.Index([f"h{i}" for i in range(300)], name="id")
    verdad = pd.Series((rng.random(300) < 0.2).astype(int), index=ids)
    base = pd.Series(np.where(rng.random(300) < 0.8, verdad, 1 - verdad), index=ids)
    candidato = base.copy()
    candidato.iloc[:15] = verdad.iloc[:15]
    tabla = bootstrap_holdout(verdad, {"base": base, "cand": candidato}, "base", n_replicas=100, semilla=2)
    tabla = tabla.set_index("candidato")
    assert tabla.loc["cand", "delta_observado"] == pytest.approx(tabla.loc["cand", "f1"] - tabla.loc["base", "f1"])
    assert tabla.loc["base", "delta_observado"] == 0


def test_estado_para_reanudar_vuelve_a_intentar_lo_que_no_termino():
    from etapa16_evaluacion import estado_para_reanudar
    estado = {"a": "terminado", "b": "corriendo", "c": "solo", "d": "fallido", "e": "pendiente_solo"}
    assert estado_para_reanudar(estado) == {"a": "terminado", "b": "pendiente", "c": "pendiente", "d": "pendiente",
                                            "e": "pendiente_solo"}


def test_es_mejora_real_exige_las_evaluaciones_completas_y_los_dos_intervalos():
    from etapa16_evaluacion import es_mejora_real
    fila = {"candidato": "x", "n_evaluaciones": 15, "delta_cv": 0.003, "delta_cv_excluye_cero": True,
            "delta_holdout": 0.002, "delta_holdout_excluye_cero": True}
    assert es_mejora_real(fila, 15, "base") is True
    assert es_mejora_real({**fila, "n_evaluaciones": 10}, 15, "base") is False
    assert es_mejora_real({**fila, "delta_cv": -0.003}, 15, "base") is False
    assert es_mejora_real({**fila, "delta_holdout_excluye_cero": False}, 15, "base") is False
    assert es_mejora_real({**fila, "candidato": "base"}, 15, "base") is False


def test_clases_holdout_exige_los_mismos_hogares():
    from etapa16_evaluacion import clases_holdout
    ids = pd.Index(["h1", "h2", "h3"], name="id")
    bien = {"a": {"prediccion_holdout": pd.Series([0.9, 0.1, 0.5], index=ids), "umbral": 0.4},
            "b": {"prediccion_holdout": pd.Series([0.2, 0.8, 0.3], index=ids[::-1]), "umbral": 0.5}}
    clases = clases_holdout(bien, ids)
    assert list(clases["a"]) == [1, 0, 1] and list(clases["b"].loc[ids]) == [0, 1, 0]
    mal = {**bien, "c": {"prediccion_holdout": pd.Series([0.9, 0.1], index=ids[:2]), "umbral": 0.4}}
    with pytest.raises(AssertionError):
        clases_holdout(mal, ids)


def test_tiempos_se_escriben_atomicamente_y_se_leen_tolerando_archivos_rotos(tmp_path, monkeypatch):
    import etapa16_evaluacion as e16
    monkeypatch.setattr(e16, "CARPETA", tmp_path)
    e16.anotar_tiempo("n16_r1_a_bag5", 12.5)
    (tmp_path / "tiempos" / "roto.json").write_text("{\"minutos\": 1", encoding="utf-8")
    assert not list((tmp_path / "tiempos").glob("*.tmp"))
    assert e16.leer_tiempos() == {"n16_r1_a_bag5": 12.5}


def test_evaluar_o_cargar_oof_limpia_los_parciales_huerfanos(tmp_path, monkeypatch):
    import etapa16_evaluacion as e16
    from funciones.repeticion import guardar_oof
    oof = pd.DataFrame({"probabilidad": [0.1, 0.9], "fold": [1, 2], "verdad": [0, 1]},
                       index=pd.Index(["h1", "h2"], name="id"))
    guardar_oof("c", oof, tmp_path)
    (tmp_path / "parciales" / "c").mkdir(parents=True)
    (tmp_path / "parciales" / "c" / "cv_1.parquet").write_bytes(b"x")
    llamadas = []
    resultado = e16.evaluar_o_cargar_oof("c", tmp_path, lambda: llamadas.append(1))
    assert llamadas == [] and not (tmp_path / "parciales" / "c").exists()
    pd.testing.assert_frame_equal(resultado, oof)


class ProcesoFalso:
    """Proceso simulado: vivo en la primera consulta, termina con su código en la segunda."""

    def __init__(self, codigo, vivos):
        self.codigo, self.vivos, self.consultas = codigo, vivos, 0
        vivos.add(id(self))

    def poll(self):
        self.consultas += 1
        if self.consultas < 2:
            return None
        self.vivos.discard(id(self))
        return self.codigo

    def wait(self):
        self.vivos.discard(id(self))
        return self.codigo


def preparar_fase1_falsa(tmp_path, monkeypatch, codigos):
    import io
    import etapa16_evaluacion as e16
    monkeypatch.setattr(e16, "CARPETA", tmp_path)
    monkeypatch.setattr(e16, "PARTICIONES", (1,))
    monkeypatch.setattr(e16, "RELANZAMIENTOS", 2)
    monkeypatch.setattr(e16, "INTERVALO_SONDEO_S", 0)
    monkeypatch.setattr(e16, "ESPERA_RELANZAMIENTO_S", 0)
    monkeypatch.setattr(e16, "memoria_libre_gb", lambda: 9.0)
    vivos, lanzados, maximo = set(), [], [0]

    def lanzar_falso(trabajo):
        lanzados.append(trabajo["nombre"])
        proceso = ProcesoFalso(codigos[trabajo["nombre"]].pop(0), vivos)
        maximo[0] = max(maximo[0], len(vivos))
        return proceso, (io.StringIO(), io.StringIO())

    monkeypatch.setattr(e16, "lanzar_trabajo", lanzar_falso)
    return e16, lanzados, maximo


def test_fase1_relanza_por_memoria_marca_fallidos_y_corre_el_reporte(tmp_path, monkeypatch):
    import json
    codigos = {"matrices_r1": [0], "knn_original": [3, 0], "arboles_r1": [0], "redes_r1": [3, 3, 3],
               "catboost_r1": [1], "reporte": [0]}
    e16, lanzados, maximo = preparar_fase1_falsa(tmp_path, monkeypatch, codigos)
    e16.fase1()
    estado = json.loads((tmp_path / "fase1" / "estado.json").read_text(encoding="utf-8"))
    assert estado["knn_original"] == "terminado" and estado["arboles_r1"] == "terminado"
    assert estado["redes_r1"] == "fallido" and lanzados.count("redes_r1") == 3  # 1 + 2 relanzamientos
    assert estado["catboost_r1"] == "fallido" and lanzados.count("catboost_r1") == 1
    assert estado["reporte"] == "terminado" and (tmp_path / "fase1" / "fin.txt").exists()
    assert maximo[0] <= 2 and lanzados[-1] == "reporte"


def test_fase1_al_reanudar_reintenta_los_fallidos(tmp_path, monkeypatch):
    import json
    codigos = {"matrices_r1": [0], "knn_original": [0], "arboles_r1": [0], "redes_r1": [0], "catboost_r1": [0],
               "reporte": [0]}
    e16, lanzados, _ = preparar_fase1_falsa(tmp_path, monkeypatch, codigos)
    (tmp_path / "fase1").mkdir()
    (tmp_path / "fase1" / "estado.json").write_text(json.dumps({"matrices_r1": "terminado", "knn_original": "fallido"}),
                                                     encoding="utf-8")
    e16.fase1()
    assert "matrices_r1" not in lanzados and lanzados.count("knn_original") == 1
    estado = json.loads((tmp_path / "fase1" / "estado.json").read_text(encoding="utf-8"))
    assert all(v == "terminado" for v in estado.values())
