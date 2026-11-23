import os
import numpy as np
import pandas as pd

# 1. Cargar el panel generado con la regresión logística
ruta_panel = "score_riesgo_corredor_2021_2025_final.csv"
panel = pd.read_csv(ruta_panel, low_memory=False)

# 2. Integrar variables locales y semáforo consistente con Google Earth (si existe el archivo calibrado)
ruta_cal = "RESULTADO_SCORE_RIESGO_CALIBRADO.csv"

if os.path.exists(ruta_cal):
    df_cal = pd.read_csv(ruta_cal, low_memory=False)
    cols_locales = [
        "id_tramo",
        "lluvia_72h_interpolada",
        "fugas_60m",
        "inund_cdmx_conteo",
        "esals_corregidos",
        "SCORE_COLAPSO_100",
        "NIVEL_RIESGO",
    ]
    cols_disp = [c for c in cols_locales if c in df_cal.columns]
    panel = panel.drop(columns=["NIVEL_RIESGO", "SCORE_COLAPSO_100"], errors="ignore").merge(
        df_cal[cols_disp].drop_duplicates(subset=["id_tramo"]),
        on="id_tramo",
        how="left",
    )
    print("[OK] Integrado NIVEL_RIESGO y variables locales desde RESULTADO_SCORE_RIESGO_CALIBRADO.csv")
else:
    # Respaldo por cuantiles de score_p_colapso si no está el archivo calibrado
    q75 = panel["score_p_colapso"].quantile(0.75)
    q90 = panel["score_p_colapso"].quantile(0.90)
    q98 = panel["score_p_colapso"].quantile(0.98)

    condiciones = [
        panel["score_p_colapso"] <= q75,
        (panel["score_p_colapso"] > q75) & (panel["score_p_colapso"] <= q90),
        (panel["score_p_colapso"] > q90) & (panel["score_p_colapso"] <= q98),
        panel["score_p_colapso"] > q98,
    ]
    etiquetas = ["1_Bajo", "2_Moderado", "3_Alto", "4_Critico"]
    panel["NIVEL_RIESGO"] = np.select(condiciones, etiquetas, default="1_Bajo")
    print("[OK] NIVEL_RIESGO calculado por cuantiles sobre score_p_colapso")

# 3. Asegurar orden y formato limpio de la columna NIVEL_RIESGO
panel["NIVEL_RIESGO"] = panel["NIVEL_RIESGO"].fillna("1_Bajo")

# Riesgo relativo respecto al promedio del corredor (útil para interpretar probabilidades pequeñas)
media_p = panel["score_p_colapso"].mean()
panel["riesgo_relativo"] = (panel["score_p_colapso"] / media_p).round(2) if media_p > 0 else 1.0

print("\n--- CONTEO POR NIVEL_RIESGO EN EL PANEL (2021-2025) ---")
print(panel["NIVEL_RIESGO"].value_counts().sort_index())

# 4. Exportar archivo definitivo completo y corte operativo 2025
panel.to_csv("DEFINITIVO_RIESGO_CORREDOR_2021_2025.csv", index=False)

corte_2025 = panel[panel["anio"] == 2025].copy()
corte_2025.to_csv("DEFINITIVO_OPERATIVO_2025_CUADRILLAS.csv", index=False)

print("\n[LISTO] Archivos exportados:")
print(" - DEFINITIVO_RIESGO_CORREDOR_2021_2025.csv (277,400 filas)")
print(" - DEFINITIVO_OPERATIVO_2025_CUADRILLAS.csv (55,480 filas)")