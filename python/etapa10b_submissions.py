"""Etapa 10b: submissions de candidatos de las etapas 9 y 10 que no generaron CSV
(ninguno cumplió la regla de mejora sobre n7_a_agregados_bag5 en CV y holdout).

Usa las predicciones de test que ya guardaron las etapas 9 y 10 y el umbral
elegido en OOF de cada candidato; no reentrena nada. La especificación y el
algoritmo se toman del leaderboard del nivel 7. Escribe cada CSV en
stores/submissions/nivel7/ y lo agrega a la bitácora con f1_kaggle vacía.

Uso (desde la raíz del repo):
    python/.venv/Scripts/python.exe python/etapa10b_submissions.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from configuracion import RUTA_RESULTADOS
from funciones.bitacora import registrar_log
from funciones.datos import cargar_particion
from funciones.modelado import resultado_desde_predicciones
from funciones.submission import guardar_submission

# Candidato -> nombre del CSV en stores/submissions/nivel7/
SUBMISSIONS = {
    "n10_mezcla_lgb_realmlp": "n10_mezcla_lgb_realmlp.csv",
    "n10_realmlp_bag5": "n10_realmlp.csv",
    "n10_tabm_bag3": "n10_tabm.csv",
    "n9_ga_bagging": "n9_ga_bagging.csv",
}


def main():
    particion = cargar_particion()
    verdad = particion.set_index("id")["pobre"]
    ids_test = particion.loc[particion["base"] == "test", "id"]
    leaderboard = pd.read_csv(RUTA_RESULTADOS / "leaderboard_nivel7.csv", dtype=str).set_index("candidato")

    for candidato, archivo in SUBMISSIONS.items():
        resultado = resultado_desde_predicciones(candidato, verdad)
        assert len(resultado["prediccion_test"]) == len(ids_test), f"{candidato}: faltan predicciones de test"
        fila = leaderboard.loc[candidato]
        ruta, tasa = guardar_submission(
            resultado, archivo, ids_test,
            f"{fila['especificacion']} | umbral {resultado['umbral']:.4f} (OOF)",
            algoritmo=fila["algoritmo"])
        registrar_log("etapa10b", f"{ruta.name} umbral {resultado['umbral']:.4f} tasa {tasa:.4f}")
        print(f"{archivo}: umbral {resultado['umbral']:.4f} | F1 CV {resultado['f1_cv']:.4f} | "
              f"holdout {resultado['f1_holdout']:.4f} | tasa de pobres en test {tasa:.4f}")


if __name__ == "__main__":
    main()
