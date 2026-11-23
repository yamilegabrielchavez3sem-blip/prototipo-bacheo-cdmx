import glob
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import statsmodels.api as sm
from scipy.spatial import cKDTree
from google.colab import drive

# =========================================================
# 0. MONTAR DRIVE Y UBICAR CARPETA DEL PROYECTO
# =========================================================
if not os.path.exists("/content/drive/MyDrive"):
    drive.mount("/content/drive")

CARPETA_PROYECTO = "/content/drive/MyDrive/Prototípico/DB_PP_Datos_limpios"
if os.path.exists(CARPETA_PROYECTO):
    os.chdir(CARPETA_PROYECTO)
print("Carpeta de trabajo activa:", os.getcwd())

RAICES_BUSQUEDA = [
    os.getcwd(),
    "/content/drive/MyDrive/Prototípico",
    "/content",
]

def localizar_archivo(patrones):
    """Busca recursivamente en las carpetas del proyecto el primer archivo coincidente."""
    for raiz in RAICES_BUSQUEDA:
        if not os.path.exists(raiz):
            continue
        for patron in patrones:
            coincidencias = sorted(
                glob.glob(os.path.join(raiz, "**", patron), recursive=True)
            )
            if coincidencias:
                return coincidencias[0]
    return None

def leer_tabla(patrones, etiqueta):
    ruta = localizar_archivo(patrones)
    if ruta is None:
        print(f"[AVISO] No se encontró {etiqueta} ({patrones})")
        return None, None
    ext = os.path.splitext(ruta)[1].lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(ruta)
    else:
        df = pd.read_csv(ruta, low_memory=False)
    df.columns = [str(c).strip().lower() for c in df.columns]
    print(f"[OK] {etiqueta}: {ruta} ({len(df)} filas)")
    return df, ruta

def norm_01(serie):
    s = pd.to_numeric(serie, errors="coerce")
    rango = s.max() - s.min()
    if pd.isna(rango) or rango == 0:
        return pd.Series(0.0, index=s.index)
    return (s - s.min()) / rango

# =========================================================
# 1. CARGAR MALLA BASE (55,480 TRAMOS) Y CREAR PANEL (2021-2025)
# =========================================================
ruta_gpkg = localizar_archivo([
    "tramos_100m_base.gpkg",
    "BASE_MAESTRA_POLIGONO_100M.gpkg",
    "*100m*.gpkg",
    "*.gpkg",
])

if ruta_gpkg is None:
    raise FileNotFoundError(
        "No se encontró ningún archivo .gpkg. Verifica que esté en tu Drive o súbelo a Colab."
    )

gdf_tramos = gpd.read_file(ruta_gpkg)
print(f"[OK] Capa base: {ruta_gpkg} ({len(gdf_tramos)} tramos)")

if "id_tramo" not in gdf_tramos.columns:
    gdf_tramos["id_tramo"] = gdf_tramos.index.astype(str)

gdf_4326 = (
    gdf_tramos.to_crs(epsg=4326)
    if gdf_tramos.crs and gdf_tramos.crs.to_epsg() != 4326
    else gdf_tramos.copy()
)
centroides = gdf_4326.geometry.centroid

base_df = gdf_tramos.drop(columns="geometry").copy()
base_df["latitud"] = centroides.y
base_df["longitud"] = centroides.x

anios_panel = [2021, 2022, 2023, 2024, 2025]
panel = (
    base_df.assign(_k=1)
    .merge(pd.DataFrame({"anio": anios_panel, "_k": 1}), on="_k")
    .drop(columns="_k")
)
panel["A_norm"] = (panel["anio"] - 2021) / (2025 - 2021)
print(f"[OK] Panel generado: {len(panel)} filas")

# =========================================================
# 2. ASIGNAR MUNICIPIO Y K_SUELO (1.5, 1.2, 1.0)
# =========================================================
centros_mun = pd.DataFrame([
    {"MUN_KEY": "NAUCALPAN DE JUAREZ", "lat": 19.475, "lon": -99.238},
    {"MUN_KEY": "TLALNEPANTLA DE BAZ", "lat": 19.539, "lon": -99.195},
    {"MUN_KEY": "CUAUTITLAN IZCALLI", "lat": 19.647, "lon": -99.216},
    {"MUN_KEY": "TEPOTZOTLAN", "lat": 19.718, "lon": -99.222},
    {"MUN_KEY": "HUEHUETOCA", "lat": 19.835, "lon": -99.205},
    {"MUN_KEY": "TULTITLAN", "lat": 19.645, "lon": -99.168},
    {"MUN_KEY": "COACALCO DE BERRIOZABAL", "lat": 19.632, "lon": -99.104},
    {"MUN_KEY": "MIGUEL HIDALGO", "lat": 19.435, "lon": -99.192},
    {"MUN_KEY": "AZCAPOTZALCO", "lat": 19.487, "lon": -99.184},
    {"MUN_KEY": "GUSTAVO A. MADERO", "lat": 19.485, "lon": -99.115},
])

arbol_mun = cKDTree(centros_mun[["lat", "lon"]].to_numpy())
_, idx_mun = arbol_mun.query(panel[["latitud", "longitud"]].to_numpy(), k=1)
panel["MUN_KEY"] = centros_mun.iloc[idx_mun]["MUN_KEY"].values

mapa_k_suelo_mun = {
    "MIGUEL HIDALGO": 1.0,            # Zona I (Lomas)
    "AZCAPOTZALCO": 1.2,              # Zona II (Transición)
    "GUSTAVO A. MADERO": 1.5,         # Zona III (Lacustre)
    "NAUCALPAN DE JUAREZ": 1.2,       # Zona I-II (Lomas / Transición)
    "TLALNEPANTLA DE BAZ": 1.2,       # Zona I-II (Lomas / Transición)
    "TULTITLAN": 1.5,                 # Depósitos lacustres/blandos
    "COACALCO DE BERRIOZABAL": 1.5,   # Depósitos lacustres/blandos
    "CUAUTITLAN IZCALLI": 1.2,        # Transición / valle norte
    "TEPOTZOTLAN": 1.0,               # Lomeríos / roca / tepetate
    "HUEHUETOCA": 1.0,                # Lomeríos / terreno firme
}

panel["K_suelo"] = panel["MUN_KEY"].map(mapa_k_suelo_mun).fillna(1.2)
panel["zona_geotecnica"] = np.where(
    panel["K_suelo"] >= 1.4, "III. Lacustre (Vertisol)",
    np.where(panel["K_suelo"] <= 1.05, "I. Lomas (Leptosol/Andosol)", "II. Transición (Phaeozem)")
)

# =========================================================
# 3. INTEGRAR AFORO SICT (E_norm) Y LLUVIA / EDOMÉX (H_norm)
# =========================================================
df_sict, _ = leer_tabla([
    "aforo_sict_limpio_2021_2025.csv",
    "aforo_sict_limpio_2021_2025.xlsx",
    "*sict*.csv",
    "*sict*.xlsx",
], "Aforo SICT")

if df_sict is not None and {"anio", "esals_diarios"}.issubset(df_sict.columns):
    df_sict["anio"] = pd.to_numeric(df_sict["anio"], errors="coerce")
    df_sict["esals_diarios"] = pd.to_numeric(df_sict["esals_diarios"], errors="coerce")
    sict_anual = (
        df_sict.dropna(subset=["anio", "esals_diarios"])
        .groupby("anio", as_index=False)
        .agg(esals_diarios=("esals_diarios", "mean"))
    )
    sict_anual["anio"] = sict_anual["anio"].astype(int)
    panel = panel.merge(sict_anual, on="anio", how="left")
    panel["E_norm"] = norm_01(panel["esals_diarios"].fillna(0.0))
else:
    panel["E_norm"] = 0.0

componentes_h = []
df_edomex, _ = leer_tabla([
    "*afectaciones*lluvia*edo*mex*.csv",
    "*afectaciones*lluvia*edo*mex*.xlsx",
    "*edomex*.csv",
    "*edomex*.xlsx",
], "Afectaciones lluvia EdoMéx")

if df_edomex is not None and {"temp_lluvia", "mun"}.issubset(df_edomex.columns):
    df_edomex["anio_ed"] = pd.to_numeric(
        df_edomex["temp_lluvia"].astype(str).str.extract(r"(\d{4})")[0], errors="coerce"
    )
    df_edomex["mun_limpio"] = (
        df_edomex["mun"].astype(str).str.upper().str.strip()
        .str.replace("Á", "A", regex=False).str.replace("É", "E", regex=False)
        .str.replace("Í", "I", regex=False).str.replace("Ó", "O", regex=False)
        .str.replace("Ú", "U", regex=False)
    )
    ed_agg = (
        df_edomex.dropna(subset=["anio_ed"])
        .groupby(["anio_ed", "mun_limpio"], as_index=False)
        .size()
        .rename(columns={"size": "eventos_edomex", "anio_ed": "anio"})
    )
    ed_agg["anio"] = ed_agg["anio"].astype(int)
    panel = panel.merge(
        ed_agg, left_on=["anio", "MUN_KEY"], right_on=["anio", "mun_limpio"], how="left"
    ).drop(columns=["mun_limpio"], errors="ignore")
    componentes_h.append(norm_01(panel["eventos_edomex"].fillna(0.0)))

if len(componentes_h) > 0:
    panel["H_norm"] = norm_01(sum(componentes_h) / len(componentes_h))
else:
    panel["H_norm"] = norm_01(0.6 * panel["A_norm"] + 0.4 * (panel["K_suelo"] - 1.0))

panel["H_suelo"] = panel["H_norm"] * panel["K_suelo"]
panel["E_x_H_suelo"] = panel["E_norm"] * panel["H_suelo"]

# =========================================================
# 4. INTEGRAR LOCATEL (2021-2024) Y BACHEO NAUCALPAN (2026)
# =========================================================
eventos_bache = []
patron_bache = r"BACHE|PAVIMENT|REPAVIMENT|SOCAV|HUNDIM|ASFALT"

for anio_loc in [2021, 2022, 2023, 2024]:
    df_loc, _ = leer_tabla([
        f"locatel0311-{anio_loc}.csv",
        f"locatel0311-{anio_loc}.xlsx",
        f"*locatel*{anio_loc}*.csv",
        f"*locatel*{anio_loc}*.xlsx",
    ], f"Locatel {anio_loc}")
    if df_loc is not None and {"latitud", "longitud"}.issubset(df_loc.columns):
        if "tema_solicitud" in df_loc.columns:
            mask = (
                df_loc["tema_solicitud"]
                .astype(str).str.upper()
                .str.contains(patron_bache, regex=True, na=False)
            )
            df_loc = df_loc[mask]
        df_loc = df_loc.copy()
        df_loc["latitud"] = pd.to_numeric(df_loc["latitud"], errors="coerce")
        df_loc["longitud"] = pd.to_numeric(df_loc["longitud"], errors="coerce")
        df_loc = df_loc.dropna(subset=["latitud", "longitud"])
        df_loc = df_loc[
            df_loc["latitud"].between(19.0, 20.1) & df_loc["longitud"].between(-99.5, -98.8)
        ]
        df_loc["anio_ev"] = anio_loc
        eventos_bache.append(df_loc[["anio_ev", "latitud", "longitud"]])

df_nauc, _ = leer_tabla([
    "Acciones_Bacheo_2026_Naucalpan.csv",
    "Acciones_Bacheo_2026_Naucalpan.xlsx",
    "*Bacheo*Naucalpan*.csv",
    "*Bacheo*Naucalpan*.xlsx",
], "Bacheo Naucalpan")

if df_nauc is not None and {"latitud", "longitud"}.issubset(df_nauc.columns):
    df_nauc = df_nauc.copy()
    df_nauc["latitud"] = pd.to_numeric(df_nauc["latitud"], errors="coerce")
    df_nauc["longitud"] = pd.to_numeric(df_nauc["longitud"], errors="coerce")
    df_nauc = df_nauc.dropna(subset=["latitud", "longitud"])
    df_nauc = df_nauc[
        df_nauc["latitud"].between(19.0, 20.1) & df_nauc["longitud"].between(-99.5, -98.8)
    ]
    df_nauc["anio_ev"] = 2025
    eventos_bache.append(df_nauc[["anio_ev", "latitud", "longitud"]])

panel["n_baches_obs"] = 0
if len(eventos_bache) > 0:
    df_ev = pd.concat(eventos_bache, ignore_index=True)
    tramos_unicos = (
        panel[["id_tramo", "latitud", "longitud"]]
        .drop_duplicates(subset=["id_tramo"])
        .reset_index(drop=True)
    )
    lat_ref = np.deg2rad(19.6)
    m_lat = 111132.0
    m_lon = 111320.0 * np.cos(lat_ref)

    arbol = cKDTree(np.column_stack([
        tramos_unicos["latitud"].to_numpy() * m_lat,
        tramos_unicos["longitud"].to_numpy() * m_lon,
    ]))
    dist_m, idx_t = arbol.query(np.column_stack([
        df_ev["latitud"].to_numpy() * m_lat,
        df_ev["longitud"].to_numpy() * m_lon,
    ]), k=1)

    dentro = dist_m <= 150.0
    df_ev_ok = df_ev[dentro].copy()
    df_ev_ok["id_tramo"] = tramos_unicos.iloc[idx_t[dentro]]["id_tramo"].values
    print(f"[OK] Eventos de bacheo/reporte en corredor (<=150m): {len(df_ev_ok)}")

    if len(df_ev_ok) > 0:
        agg_ev = (
            df_ev_ok.groupby(["id_tramo", "anio_ev"])
            .size()
            .reset_index(name="conteo_ev")
        )
        panel = panel.merge(
            agg_ev, left_on=["id_tramo", "anio"], right_on=["id_tramo", "anio_ev"], how="left"
        ).drop(columns=["anio_ev"], errors="ignore")
        panel["n_baches_obs"] = panel["conteo_ev"].fillna(0).astype(int)
        panel.drop(columns=["conteo_ev"], errors="ignore", inplace=True)

if panel["n_baches_obs"].sum() > 0:
    panel["y_colapso"] = (panel["n_baches_obs"] > 0).astype(int)
else:
    indice_respaldo = 0.40 * panel["E_norm"] + 0.40 * panel["H_suelo"] + 0.20 * panel["A_norm"]
    panel["y_colapso"] = (indice_respaldo >= indice_respaldo.quantile(0.80)).astype(int)

print("Conteo y_colapso:\n", panel["y_colapso"].value_counts())

# =========================================================
# 5. AJUSTE LOGÍSTICO Y EXPORTACIÓN FINAL
# =========================================================
cols_x = ["E_norm", "H_suelo", "A_norm", "K_suelo", "E_x_H_suelo"]
for c in cols_x:
    panel[c] = pd.to_numeric(panel[c], errors="coerce").fillna(0.0)

cols_activas = [c for c in cols_x if panel[c].std(ddof=0) > 1e-12]
X = sm.add_constant(panel[cols_activas], has_constant="add")
y = panel["y_colapso"].astype(int)

try:
    modelo = sm.Logit(y, X).fit(method="bfgs", maxiter=500, disp=False)
    panel["score_p_colapso"] = modelo.predict(X)
    print("\n--- RESUMEN LOGÍSTICO ---")
    print(modelo.summary())
except Exception as e:
    print("[AVISO] MLE estándar con dificultad; usando regularización L2:", e)
    reg = sm.Logit(y, X).fit_regularized(method="l1", alpha=1.0, L1_wt=0.0, disp=False)
    panel["score_p_colapso"] = reg.predict(X)
    print("Parámetros L2:\n", reg.params)

print("\n--- DISTRIBUCIÓN ANUAL DE score_p_colapso ---")
print(panel.groupby("anio")["score_p_colapso"].describe())

ruta_salida = os.path.join(os.getcwd(), "score_riesgo_corredor_2021_2025_final.csv")
panel.to_csv(ruta_salida, index=False)
print(f"\n[LISTO] Guardado en: {ruta_salida}")