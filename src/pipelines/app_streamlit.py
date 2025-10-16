# -*- coding: utf-8 -*-

"""
MELI-Boost Intelligence Platform
Dashboard narrativo del proceso de desarrollo end-to-end
"""

import sys
from pathlib import Path



ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from catboost import CatBoostRegressor, Pool
from sqlalchemy import create_engine, text, inspect

from src.features import clean_cols, NUM_COLS, CAT_COLS
from src.data_access import load_clients, upsert_predictions
from src.config import DB_URI


# -------------------------
# Configuración de página
# -------------------------

st.set_page_config(
    page_title="MELI-Boost Intelligence Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado mejorado
st.html("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hero header */
    .hero-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: pulse 15s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    
    .hero-header h1 {
        color: white;
        font-weight: 800;
        margin: 0;
        font-size: 3rem;
        position: relative;
        z-index: 1;
    }
    
    .hero-header .subtitle {
        color: rgba(255,255,255,0.95);
        margin: 0.8rem 0 0 0;
        font-size: 1.3rem;
        font-weight: 300;
        position: relative;
        z-index: 1;
    }
    
    .hero-header .model-name {
        display: inline-block;
        background: rgba(255,255,255,0.2);
        padding: 0.5rem 1.2rem;
        border-radius: 20px;
        margin-top: 1rem;
        font-weight: 600;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.3);
    }
    
    /* Section headers con timeline */
    .section-header-container {
        display: flex;
        align-items: center;
        margin: 3rem 0 1.5rem 0;
        position: relative;
    }
    
    .section-number {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        width: 50px;
        height: 50px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 1.4rem;
        margin-right: 1rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .section-title {
        color: #2d3748;
        font-weight: 700;
        font-size: 1.8rem;
        margin: 0;
    }
    
    .section-subtitle {
        color: #718096;
        font-size: 1rem;
        margin: 0.5rem 0 0 66px;
        font-style: italic;
    }
    
    /* Narrative boxes */
    .narrative-box {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #667eea;
        margin: 1.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .narrative-box h4 {
        color: #667eea;
        font-weight: 700;
        margin-top: 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .narrative-box p {
        color: #4a5568;
        line-height: 1.7;
        margin: 0.8rem 0;
    }
    
    .narrative-box ul {
        color: #4a5568;
        line-height: 1.8;
    }
    
    .narrative-box code {
        background: rgba(102, 126, 234, 0.1);
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9em;
        color: #667eea;
    }
    
    /* Metric cards mejoradas */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-top: 4px solid #667eea;
        transition: all 0.3s;
        text-align: center;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.2);
    }
    
    .metric-card .metric-label {
        color: #718096;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-card .metric-value {
        color: #2d3748;
        font-size: 2rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }
    
    /* Chart containers */
    .chart-container {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin: 1.5rem 0;
    }
    
    .chart-title {
        color: #2d3748;
        font-weight: 700;
        font-size: 1.2rem;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .chart-insight {
        background: #f7fafc;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
        font-size: 0.95rem;
        color: #4a5568;
        border-left: 3px solid #667eea;
    }
    
    /* Decision cards */
    .decision-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin: 1rem 0;
        border-left: 4px solid #48bb78;
    }
    
    .decision-card h5 {
        color: #48bb78;
        font-weight: 700;
        margin: 0 0 0.8rem 0;
    }
    
    /* Process timeline */
    .timeline-item {
        display: flex;
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .timeline-dot {
        min-width: 12px;
        height: 12px;
        background: #667eea;
        border-radius: 50%;
        margin-top: 0.5rem;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.2);
    }
    
    .timeline-content {
        flex: 1;
        padding-bottom: 1.5rem;
        border-left: 2px solid #e2e8f0;
        margin-left: 5px;
        padding-left: 2rem;
    }
    
    .timeline-content h5 {
        color: #2d3748;
        font-weight: 600;
        margin: 0 0 0.5rem 0;
    }
    
    .timeline-content p {
        color: #718096;
        line-height: 1.6;
        margin: 0;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.8rem 2.5rem;
        font-weight: 700;
        font-size: 1.05rem;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
    }
    
    /* Info callouts */
    .callout {
        padding: 1.2rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-size: 0.95rem;
    }
    
    .callout-info {
        background: #ebf8ff;
        border-left: 4px solid #3182ce;
        color: #2c5282;
    }
    
    .callout-success {
        background: #f0fff4;
        border-left: 4px solid #48bb78;
        color: #22543d;
    }
    
    .callout-warning {
        background: #fffaf0;
        border-left: 4px solid #ed8936;
        color: #7c2d12;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f7fafc 0%, #edf2f7 100%);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #f7fafc;
        border-radius: 10px;
        font-weight: 600;
        color: #2d3748;
    }
</style>
""")


# -------------------------
# Paths y constantes
# -------------------------
ARTIFACTS_DIR = Path("artifacts")
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------
# Funciones auxiliares
# -------------------------
def get_latest_model_path() -> Path:
    runs = sorted(
        ARTIFACTS_DIR.glob("*/model_catboost.cbm"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    if not runs:
        st.error("⚠️ No hay modelos en artifacts/. Entrena uno primero.")
        st.stop()
    return runs[0]


@st.cache_resource(show_spinner=False)
def load_model(model_path: Path) -> CatBoostRegressor:
    model = CatBoostRegressor()
    model.load_model(str(model_path))
    return model


def score_df(model: CatBoostRegressor, df_raw: pd.DataFrame) -> pd.DataFrame:
    df = clean_cols(df_raw)
    cat_idx = list(range(len(NUM_COLS), len(NUM_COLS) + len(CAT_COLS)))
    pool = Pool(df[NUM_COLS + CAT_COLS], cat_features=cat_idx)
    df["pred_total_compras"] = model.predict(pool)
    return df


def read_metrics_bundle(model_path: Path):
    run_dir = model_path.parent
    payload = {}
    mfile = run_dir / "metrics.json"
    if mfile.exists():
        with open(mfile, "r", encoding="utf-8") as f:
            payload["metrics"] = json.load(f)
    for name in [
        ("feature_importances.csv", "fi"),
        ("deciles_diagnostics.csv", "deciles"),
        ("segment_metrics_brand.csv", "seg_brand"),
        ("segment_metrics_gender.csv", "seg_gender"),
    ]:
        fpath = run_dir / name[0]
        if fpath.exists():
            payload[name[1]] = pd.read_csv(fpath)
    return payload


@st.cache_resource(show_spinner=False)
def get_engine():
    return create_engine(DB_URI, pool_pre_ping=True)


def run_sql_df(sql: str) -> pd.DataFrame:
    eng = get_engine()
    with eng.connect() as con:
        return pd.read_sql(text(sql), con)


plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# =========================
# APP PRINCIPAL
# =========================

# Hero Header
st.html("""
<div class="hero-header">
    <h1>🚀 MELI-Boost Intelligence Platform</h1>
    <p class="subtitle">Una narrativa completa del proceso de desarrollo end-to-end</p>
    <span class="model-name">✨ Modelo: MELI-Boost v1 — CatBoost Regressor</span>
</div>
""")


# -------------------------
# SIDEBAR
# -------------------------
with st.sidebar:
    st.image("https://http2.mlstatic.com/frontend-assets/ml-web-navigation/ui-navigation/5.21.22/mercadolibre/logo__large_plus.png", width=200)
    st.markdown("---", unsafe_allow_html=True)
    
    st.markdown("### 🎯 Navegación Rápida", unsafe_allow_html=True)
    st.markdown("""
    - [📋 Metodología](#metodolog-a-y-criterios)
    - [🗄️ Datos](#datos-y-arquitectura)
    - [🤖 Modelo](#modelo-meli-boost-v1)
    - [📊 Evaluación](#evaluaci-n-y-diagn-sticos)
    - [💡 Insights](#insights-de-negocio)
    - [🎯 Plan de Acción](#plan-de-acci-n)
    """, unsafe_allow_html=True)
    
    st.markdown("---", unsafe_allow_html=True)
    st.markdown("### ⚙️ Configuración", unsafe_allow_html=True)
    
    model_path = get_latest_model_path()
    st.success(f"**Modelo activo:** {model_path.parent.name}")
    model = load_model(model_path)
    
    st.markdown("---", unsafe_allow_html=True)
    st.markdown("### 📊 Opciones", unsafe_allow_html=True)
    show_technical = st.checkbox("Detalles técnicos", value=False)
    show_code = st.checkbox("Snippets de código", value=False)


# -------------------------
# SECCIÓN 1: METODOLOGÍA
# -------------------------
st.html("""
<div class="section-header-container">
    <div class="section-number">1</div>
    <div>
        <h2 class="section-title">Metodología y Criterios de Éxito</h2>
    </div>
</div>
<p class="section-subtitle">Traduciendo objetivos de negocio en preguntas analíticas accionables</p>
""")

st.html("""
<div class="narrative-box">
    <h4>🎯 El Problema de Negocio</h4>
    <p>Mercado Libre necesita <strong>entender profundamente el comportamiento de sus clientes</strong> para optimizar estrategias de marketing, mejorar la satisfacción y maximizar el valor del ciclo de vida del cliente.</p>
    
    <p>Partimos de tres pilares fundamentales:</p>
    <ul>
        <li><strong>Diagnóstico:</strong> ¿Qué perfiles compran más? ¿Dónde se concentran las promociones? ¿Qué marcas generan más valor?</li>
        <li><strong>Insights:</strong> Medir señales de satisfacción (proxys como <code>total_compras</code> y <code>frecuencia_de_compra</code>)</li>
        <li><strong>Acción:</strong> Lineamientos de marketing y producto respaldados por datos</li>
    </ul>
</div>
""")

col1, col2 = st.columns(2)

with col1:
    st.html("""
    <div class="decision-card">
        <h5>🔍 Decisión: Target Analítico</h5>
        <p>Elegimos <code>total_compras</code> como variable objetivo porque:</p>
        <ul>
            <li>Es continua y permite regresión</li>
            <li>Refleja directamente el valor del cliente</li>
            <li>Permite scoring y priorización</li>
            <li>Es accionable para campañas</li>
        </ul>
    </div>
    """)

with col2:
    st.html("""
    <div class="decision-card">
        <h5>🎲 Decisión: Drivers Clave</h5>
        <p>Identificamos las variables que impulsan el comportamiento:</p>
        <ul>
            <li><strong>Edad:</strong> segmentación demográfica</li>
            <li><strong>Género:</strong> preferencias diferenciadas</li>
            <li><strong>Marca preferida:</strong> lealtad de marca</li>
            <li><strong>Frecuencia:</strong> engagement</li>
            <li><strong>Promociones:</strong> sensibilidad a precio</li>
        </ul>
    </div>
    """)

st.html("""
<div class="callout callout-success">
    <strong>✅ Criterios de Éxito Definidos</strong><br>
    Un modelo explicativo y operativo con métricas de generalización razonables (R², MAE, RMSE, MAPE), 
    diagnósticos por segmentos y una interfaz 100% conectada a SQL.
</div>
""")


# -------------------------
# SECCIÓN 2: DATOS
# -------------------------
st.html("""
<div class="section-header-container">
    <div class="section-number">2</div>
    <div>
        <h2 class="section-title">Datos y Arquitectura</h2>
    </div>
</div>
<p class="section-subtitle">Ingesta, estandarización y gobernanza desde SQL</p>
""")

st.html("""
<div class="narrative-box">
    <h4>🗄️ Pipeline de Datos</h4>
    <p>Desde el inicio, diseñamos una arquitectura que garantiza <strong>reproducibilidad y trazabilidad</strong>:</p>
    
    <div class="timeline-item">
        <div class="timeline-dot"></div>
        <div class="timeline-content">
            <h5>Ingesta desde Excel → SQL</h5>
            <p>Los datos originales se cargan en una base SQL (SQLite local o PostgreSQL) como <em>single source of truth</em></p>
        </div>
    </div>
    
    <div class="timeline-item">
        <div class="timeline-dot"></div>
        <div class="timeline-content">
            <h5>Normalización de Columnas</h5>
            <p>Nombres estandarizados: <code>edad</code>, <code>genero</code>, <code>marca_preferida</code>, <code>total_compras</code>, <code>frecuencia_de_compra</code>, <code>promociones_utilizadas</code></p>
        </div>
    </div>
    
    <div class="timeline-item">
        <div class="timeline-dot"></div>
        <div class="timeline-content">
            <h5>Limpieza Conservadora</h5>
            <p>Trimming de strings, uppercase en categóricas, fillna solo donde tiene sentido operativo</p>
        </div>
    </div>
</div>
""")

with st.spinner("⏳ Cargando datos desde SQL..."):
    df = load_clients()

col1, col2, col3, col4 = st.columns(4)

edad_col = "Edad" if "Edad" in df.columns else "edad"
tc_col = "Total Compras" if "Total Compras" in df.columns else "total_compras"
freq_col = "Frecuencia de Compra" if "Frecuencia de Compra" in df.columns else "frecuencia_de_compra"

metrics_data = [
    ("📊", "Total Registros", f"{len(df):,}"),
    ("👥", "Columnas", f"{len(df.columns)}"),
    ("📆", "Edad Promedio", f"{df[edad_col].mean():.1f} años"),
    ("🛍️", "Compras Promedio", f"{df[tc_col].mean():.1f}")
]

for col, (icon, label, value) in zip([col1, col2, col3, col4], metrics_data):
    with col:
        st.html(f"""
        <div class="metric-card">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """)

with st.expander("👁️ **Explorar datos completos**"):
    st.dataframe(df.head(499), use_container_width=True)
    
    if show_technical:
        st.html("**Información Técnica del Dataset:**")
        buffer = df.dtypes.to_frame('dtype').reset_index()
        buffer.columns = ['Columna', 'Tipo']
        st.dataframe(buffer, use_container_width=True)


# -------------------------
# SECCIÓN 3: EDA
# -------------------------
st.html("""
<div class="section-header-container">
    <div class="section-number">3</div>
    <div>
        <h2 class="section-title">Análisis Exploratorio (EDA)</h2>
    </div>
</div>
<p class="section-subtitle">Detectando patrones, outliers y relaciones antes del modelado</p>
""")

st.html("""
<div class="narrative-box">
    <h4>🔬 Proceso de Exploración</h4>
    <p>El EDA no es solo visualizar; es <strong>tomar decisiones informadas</strong> sobre el modelo:</p>
    <ul>
        <li>Identificamos <strong>asimetrías</strong> en edad y compras para decidir estrategias de binning</li>
        <li>Detectamos posibles <strong>outliers</strong> que podrían sesgar el entrenamiento</li>
        <li>Evaluamos el <strong>balance</strong> en variables categóricas (género, marca)</li>
        <li>Verificamos <strong>calidad de datos</strong>: nulos, tipos inconsistentes, rangos anómalos</li>
    </ul>
</div>
""")

col1, col2 = st.columns(2)

with col1:
    st.html('<div class="chart-container">')
    st.html('<p class="chart-title">📊 Distribución de Edad</p>')
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df[edad_col], bins=30, color='#667eea', alpha=0.7, edgecolor='black')
    ax.axvline(df[edad_col].mean(), color='#764ba2', linestyle='--', linewidth=2, label=f'Media: {df[edad_col].mean():.1f}')
    ax.axvline(df[edad_col].median(), color='#f093fb', linestyle='--', linewidth=2, label=f'Mediana: {df[edad_col].median():.1f}')
    ax.set_xlabel("Edad", fontsize=12, fontweight='bold')
    ax.set_ylabel("Frecuencia", fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    st.pyplot(fig, clear_figure=True)
    
    st.html(f"""
    <div class="chart-insight">
        <strong>💡 Insight:</strong> La distribución de edad muestra {
            'una distribución aproximadamente normal' if abs(df[edad_col].skew()) < 0.5 
            else 'asimetría, con concentración en ciertos rangos etarios'
        }. Esto sugiere {'segmentación etaria clara' if abs(df[edad_col].skew()) < 0.5 else 'oportunidades de microsegmentación'}.
    </div>
    """)
    st.html('</div>')

with col2:
    st.html('<div class="chart-container">')
    st.html('<p class="chart-title">🛍️ Distribución de Total Compras</p>')
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df[tc_col], bins=30, color='#764ba2', alpha=0.7, edgecolor='black')
    ax.axvline(df[tc_col].mean(), color='#667eea', linestyle='--', linewidth=2, label=f'Media: {df[tc_col].mean():.1f}')
    ax.axvline(df[tc_col].median(), color='#f093fb', linestyle='--', linewidth=2, label=f'Mediana: {df[tc_col].median():.1f}')
    ax.set_xlabel("Total Compras", fontsize=12, fontweight='bold')
    ax.set_ylabel("Frecuencia", fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    st.pyplot(fig, clear_figure=True)
    
    percentile_95 = df[tc_col].quantile(0.95)
    st.html(f"""
    <div class="chart-insight">
        <strong>💡 Insight:</strong> El 5% de clientes con mayor compra supera {percentile_95:.1f} unidades. 
        Estos son candidatos perfectos para programas VIP y estrategias de fidelización premium.
    </div>
    """)
    st.html('</div>')

# Segmentación
col1, col2 = st.columns(2)

with col1:
    st.html('<div class="chart-container">')
    st.html('<p class="chart-title">🏷️ Top 10 Marcas Preferidas</p>')
    
    marca_col = "Marca Preferida" if "Marca Preferida" in df.columns else "marca_preferida"
    marca_counts = df[marca_col].value_counts().head(10)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(marca_counts)))
    marca_counts.plot(kind='barh', ax=ax, color=colors)
    ax.set_xlabel("Cantidad de Clientes", fontsize=12, fontweight='bold')
    ax.set_ylabel("")
    ax.grid(axis='x', alpha=0.3)
    st.pyplot(fig, clear_figure=True)
    
    top_marca = marca_counts.index[0]
    top_count = marca_counts.values[0]
    st.html(f"""
    <div class="chart-insight">
        <strong>💡 Insight:</strong> {top_marca} domina con {top_count} clientes ({top_count/len(df)*100:.1f}%). 
        Esta concentración indica fuerte lealtad de marca y oportunidades de cross-selling.
    </div>
    """)
    st.html('</div>')

with col2:
    st.html('<div class="chart-container">')
    st.html('<p class="chart-title">👥 Distribución por Género</p>')
    
    gen_col = "Género" if "Género" in df.columns else "genero"
    gen_counts = df[gen_col].value_counts()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#667eea', '#764ba2', '#f093fb']
    wedges, texts, autotexts = ax.pie(gen_counts, labels=gen_counts.index, autopct='%1.1f%%', 
                                        colors=colors[:len(gen_counts)], startangle=90,
                                        textprops={'fontsize': 11, 'fontweight': 'bold'})
    for autotext in autotexts:
        autotext.set_color('white')
    ax.axis('equal')
    st.pyplot(fig, clear_figure=True)
    
    balance_score = 1 - (gen_counts.std() / gen_counts.mean())
    st.html(f"""
    <div class="chart-insight">
        <strong>💡 Insight:</strong> Balance de género: {balance_score*100:.1f}%. 
        {'Dataset balanceado permite análisis sin sesgos significativos' if balance_score > 0.8 else 'Considerar estrategias diferenciadas por género'}.
    </div>
    """)
    st.html('</div>')

# Calidad de datos
with st.expander("🔎 **Diagnóstico de Calidad de Datos**"):
    st.html("""
    <div class="narrative-box">
        <h4>🔍 Análisis de Integridad</h4>
        <p>Revisamos sistemáticamente la calidad de cada columna para detectar problemas que podrían afectar el modelo.</p>
    </div>
    """)
    
    cols_check = [c for c in df.columns if any(x in c.lower() for x in ["edad", "frecuencia", "promocion", "compra", "genero", "marca"])]
    nulls = df[cols_check].isna().sum().reset_index()
    nulls.columns = ["Columna", "Valores Nulos"]
    nulls["% Nulos"] = (nulls["Valores Nulos"] / len(df) * 100).round(2)
    nulls["Estado"] = nulls["% Nulos"].apply(lambda x: "✅ Excelente" if x == 0 else ("⚠️ Atención" if x < 5 else "❌ Crítico"))
    
    st.dataframe(nulls, use_container_width=True)
    
    total_nulls = nulls["Valores Nulos"].sum()
    if total_nulls == 0:
        st.success("✅ **Dataset limpio:** No se detectaron valores nulos en columnas críticas")
    else:
        st.warning(f"⚠️ **Atención:** {total_nulls} valores nulos detectados. Estrategia de imputación aplicada.")


# -------------------------
# SECCIÓN 4: MODELO
# -------------------------
st.html("""
<div class="section-header-container">
    <div class="section-number">4</div>
    <div>
        <h2 class="section-title">Modelo: MELI-Boost v1</h2>
    </div>
</div>
<p class="section-subtitle">CatBoost Regressor optimizado con búsqueda de hiperparámetros</p>
""")

st.html("""
<div class="narrative-box">
    <h4>🤖 ¿Por qué CatBoost?</h4>
    <p>La elección del algoritmo no fue aleatoria. CatBoost se seleccionó por cuatro razones estratégicas:</p>
    
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;">
        <div style="background: white; padding: 1rem; border-radius: 8px; border-left: 3px solid #48bb78;">
            <strong>🎯 Categóricas Nativas</strong><br>
            Maneja <code>marca_preferida</code> y <code>genero</code> sin one-hot encoding manual, preservando información ordinal
        </div>
        <div style="background: white; padding: 1rem; border-radius: 8px; border-left: 3px solid #4299e1;">
            <strong>🧮 No Linealidades</strong><br>
            Captura interacciones complejas edad×marca×promociones sin feature engineering explícito
        </div>
        <div style="background: white; padding: 1rem; border-radius: 8px; border-left: 3px solid #9f7aea;">
            <strong>🛡️ Regularización</strong><br>
            Early stopping y penalizaciones integradas reducen overfitting automáticamente
        </div>
        <div style="background: white; padding: 1rem; border-radius: 8px; border-left: 3px solid #ed8936;">
            <strong>📊 Explicabilidad</strong><br>
            PredictionValuesChange proporciona importancias interpretables para negocio
        </div>
    </div>
</div>
""")

col1, col2 = st.columns([2, 1])

with col1:
    st.html("""
    <div class="decision-card">
        <h5>⚙️ Proceso de Optimización</h5>
        <ol style="margin: 0.5rem 0 0 0; padding-left: 1.5rem;">
            <li><strong>Random Search:</strong> Exploración del espacio de hiperparámetros (profundidad, learning_rate, regularización, n_estimators)</li>
            <li><strong>Validación Cruzada:</strong> KFold con k=5 para estabilidad de métricas</li>
            <li><strong>Selección de Mejor:</strong> Priorizando R² CV como señal de capacidad explicativa</li>
            <li><strong>Entrenamiento Final:</strong> Con mejores parámetros en train completo</li>
        </ol>
    </div>
    """)

with col2:
    st.html("""
    <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🎯</div>
        <div style="color: rgba(255,255,255,0.9); font-size: 0.9rem; font-weight: 600;">OBJETIVO</div>
        <div style="font-size: 1.5rem; font-weight: 800; margin: 0.5rem 0;">Predecir</div>
        <div style="font-size: 1.1rem; font-weight: 600;">total_compras</div>
    </div>
    """)

if show_technical:
    with st.expander("🔧 **Configuración Técnica del Modelo**"):
        st.html("""
        ```python
        # Espacio de hiperparámetros explorado
        param_space = {
            'depth': [4, 6, 8, 10],
            'learning_rate': [0.01, 0.03, 0.05, 0.1],
            'l2_leaf_reg': [1, 3, 5, 7],
            'iterations': [100, 200, 500, 1000],
            'border_count': [32, 64, 128],
            'random_strength': [0.5, 1, 2]
        }
        
        # Variables numéricas y categóricas
        NUM_COLS = ['edad', 'frecuencia_de_compra', 'promociones_utilizadas']
        CAT_COLS = ['genero', 'marca_preferida']
        ```
        """)


# -------------------------
# SECCIÓN 5: EVALUACIÓN
# -------------------------
st.html("""
<div class="section-header-container">
    <div class="section-number">5</div>
    <div>
        <h2 class="section-title">Evaluación y Diagnósticos</h2>
    </div>
</div>
<p class="section-subtitle">Validación cruzada, métricas globales y análisis por segmentos</p>
""")

st.html("""
<div class="narrative-box">
    <h4>📊 Estrategia de Evaluación Multinivel</h4>
    <p>No basta con un R² global. Implementamos un sistema de evaluación que responde a preguntas de negocio:</p>
    <ul>
        <li><strong>¿El modelo generaliza?</strong> → Validación cruzada con 5 folds</li>
        <li><strong>¿Qué tan preciso es?</strong> → MAE, RMSE, MAPE en test holdout</li>
        <li><strong>¿Sobre/subpredice?</strong> → Análisis por deciles</li>
        <li><strong>¿Es justo entre segmentos?</strong> → Métricas por marca y género</li>
    </ul>
</div>
""")

bundle = read_metrics_bundle(model_path)

if "metrics" in bundle:
    gm = bundle["metrics"].get("global_metrics", {})
    
    st.markdown("### 🎯 Métricas de Performance", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    metrics_display = [
        ("R² Score", gm.get('R2_test', float('nan')), "📈", "Varianza explicada"),
        ("MAE", gm.get('MAE_test', float('nan')), "🎯", "Error absoluto medio"),
        ("RMSE", gm.get('RMSE_test', float('nan')), "📊", "Error cuadrático medio"),
        ("MAPE", gm.get('MAPE_test', float('nan')), "💯", "Error porcentual")
    ]
    
    for col, (label, value, icon, desc) in zip([col1, col2, col3, col4], metrics_display):
        with col:
            if label == "MAPE":
                display_value = f"{value:.2f}%" if not np.isnan(value) else "N/A"
            else:
                display_value = f"{value:.3f}" if not np.isnan(value) else "N/A"
            
            # Color basado en calidad
            if label == "R² Score":
                color = "#48bb78" if value > 0.7 else ("#ed8936" if value > 0.5 else "#f56565")
            else:
                color = "#667eea"
            
            st.html(f"""
            <div class="metric-card" style="border-top-color: {color};">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
                <div class="metric-label">{label}</div>
                <div class="metric-value">{display_value}</div>
                <div style="color: #718096; font-size: 0.85rem; margin-top: 0.3rem;">{desc}</div>
            </div>
            """)
    
    # Interpretación de métricas
    r2_val = gm.get('R2_test', 0)
    if r2_val > 0.75:
        interpretation = "🌟 Excelente capacidad explicativa"
        color_class = "success"
    elif r2_val > 0.6:
        interpretation = "✅ Buena capacidad predictiva"
        color_class = "info"
    else:
        interpretation = "⚠️ Margen de mejora detectado"
        color_class = "warning"
    
    st.html(f"""
    <div class="callout callout-{color_class}">
        <strong>{interpretation}</strong><br>
        El modelo explica {r2_val*100:.1f}% de la variabilidad en total_compras. 
        {'Ideal para producción y toma de decisiones.' if r2_val > 0.7 else 'Considerar feature engineering adicional o datos complementarios.'}
    </div>
    """)

# Feature Importance
if "fi" in bundle:
    st.markdown("### 🔑 Variables Más Influyentes", unsafe_allow_html=True)
    
    st.html("""
    <div class="narrative-box">
        <h4>💡 ¿Qué impulsa las compras?</h4>
        <p>La importancia de variables nos dice <strong>dónde enfocar esfuerzos</strong>. Variables con alta importancia son palancas de acción prioritarias.</p>
    </div>
    """)
    
    fi = bundle["fi"].head(10)
    
    st.html('<div class="chart-container">')
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(fi)))
    bars = ax.barh(fi['feature'], fi['importance_pvc'], color=colors)
    
    # Añadir valores en las barras
    for i, (bar, val) in enumerate(zip(bars, fi['importance_pvc'])):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{val:.2f}%', va='center', fontweight='bold', fontsize=10)
    
    ax.set_xlabel("Importancia (%)", fontsize=12, fontweight='bold')
    ax.set_ylabel("")
    ax.set_title("Top 10 Variables por Prediction Value Change", fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    st.pyplot(fig, clear_figure=True)
    st.html('</div>')
    
    top_feature = fi.iloc[0]['feature']
    top_importance = fi.iloc[0]['importance_pvc']
    
    st.html(f"""
    <div class="chart-insight">
        <strong>💡 Insight Clave:</strong> <code>{top_feature}</code> es el factor dominante con {top_importance:.2f}% de importancia. 
        Estrategias que optimicen esta variable tendrán el mayor impacto en compras.
    </div>
    """)

# Deciles
if "deciles" in bundle:
    with st.expander("📈 **Diagnóstico por Deciles de Predicción**"):
        st.html("""
        <div class="narrative-box">
            <h4>🎯 Calibración del Modelo</h4>
            <p>Los deciles revelan si el modelo <strong>sobre o subestima sistemáticamente</strong> en diferentes rangos de predicción.</p>
        </div>
        """)
        
        deciles_df = bundle["deciles"]
        st.dataframe(deciles_df.style.background_gradient(subset=['y_pred_mean', 'y_true_mean'], cmap='RdYlGn'), 
                     use_container_width=True)
        
        st.html("""
        <div class="callout callout-info">
            <strong>📊 Cómo leer esta tabla:</strong><br>
            • <strong>Deciles bajos:</strong> clientes de bajo valor predicho<br>
            • <strong>Deciles altos:</strong> clientes de alto valor predicho<br>
            • <strong>Diferencia y_pred - y_true:</strong> si es positiva → sobrepredicción, si es negativa → subpredicción<br>
            • <strong>Objetivo:</strong> diferencias cercanas a cero indican buena calibración
        </div>
        """)

# Segmentos
if "seg_brand" in bundle or "seg_gender" in bundle:
    st.markdown("### 🎯 Análisis de Equidad por Segmentos", unsafe_allow_html=True)
    
    st.html("""
    <div class="narrative-box">
        <h4>⚖️ Fairness y Performance Diferencial</h4>
        <p>Evaluamos si el modelo funciona <strong>igualmente bien para todos los grupos</strong>. 
        Diferencias significativas en métricas pueden indicar:</p>
        <ul>
            <li>Oportunidades de mejora en features específicas de segmento</li>
            <li>Necesidad de modelos especializados</li>
            <li>Sesgos en datos de entrenamiento a corregir</li>
        </ul>
    </div>
    """)
    
    tab1, tab2 = st.tabs(["📊 Análisis por Marca", "👥 Análisis por Género"])

with tab1:
    if "seg_brand" in bundle:
        seg_brand = bundle["seg_brand"]

        # detectar nombre dinámico de la columna de segmento (ej. seg_0)
        seg_col = next((c for c in seg_brand.columns if str(c).startswith("seg_")), seg_brand.columns[0])

        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(
                seg_brand.rename(columns={seg_col: "segmento_marca"})
                .style.background_gradient(subset=["R2", "MAE"], cmap="RdYlGn"),
                use_container_width=True
            )

        with col2:
            best_brand = seg_brand.loc[seg_brand["R2"].idxmax(), seg_col]
            worst_brand = seg_brand.loc[seg_brand["R2"].idxmin(), seg_col]

            st.html(f"""
            <div class="metric-card" style="background: #f0fff4; border-top-color: #48bb78;">
                <div style="color: #22543d; font-weight: 600; margin-bottom: 0.5rem;">🏆 Mejor Performance</div>
                <div style="color: #2d3748; font-size: 1.3rem; font-weight: 700;">{best_brand}</div>
            </div>

            <div class="metric-card" style="background: #fffaf0; border-top-color: #ed8936; margin-top: 1rem;">
                <div style="color: #7c2d12; font-weight: 600; margin-bottom: 0.5rem;">⚠️ Requiere Atención</div>
                <div style="color: #2d3748; font-size: 1.3rem; font-weight: 700;">{worst_brand}</div>
            </div>
            """)

with tab2:
    if "seg_gender" in bundle:
        seg_gender = bundle["seg_gender"]

        st.dataframe(
            seg_gender.style.background_gradient(subset=["R2", "MAE"], cmap="RdYlGn"),
            use_container_width=True
        )

        r2_diff = seg_gender["R2"].max() - seg_gender["R2"].min()
        if r2_diff < 0.05:
            fairness_msg = "✅ Performance balanceada entre géneros"
            fairness_class = "success"
        elif r2_diff < 0.1:
            fairness_msg = "⚠️ Ligera diferencia de performance entre géneros"
            fairness_class = "warning"
        else:
            fairness_msg = "❌ Diferencia significativa - revisar features o datos"
            fairness_class = "warning"

        st.html(f"""
        <div class="callout callout-{fairness_class}">
            <strong>{fairness_msg}</strong><br>
            Diferencia de R² entre géneros: {r2_diff:.3f}
        </div>
        """)


# -------------------------
# SECCIÓN 6: INSIGHTS SQL
# -------------------------
st.html("""
<div class="section-header-container">
    <div class="section-number">6</div>
    <div>
        <h2 class="section-title">Insights de Negocio</h2>
    </div>
</div>
<p class="section-subtitle">KPIs por segmento: marca × género desde SQL</p>
""")

st.html("""
<div class="narrative-box">
    <h4>💼 Del Modelo a la Acción</h4>
    <p>Los insights accionables surgen de <strong>cruzar predicciones con segmentos de negocio</strong>:</p>
    <ul>
        <li><strong>Marcas con alto avg_total_compras y bajo avg_promos:</strong> Champions orgánicos, fidelidad natural</li>
        <li><strong>Marcas con bajo avg_total_compras y alto avg_promos:</strong> Dependencia de descuentos, revisar rentabilidad</li>
        <li><strong>Combinaciones edad×género×marca:</strong> Audiencias para campañas hipersegmentadas</li>
    </ul>
</div>
""")

eng = get_engine()
dialect = inspect(eng).dialect.name.lower()

if dialect == "sqlite":
    sql = """
    SELECT
        UPPER(genero) AS genero,
        UPPER(marca_preferida) AS marca,
        ROUND(AVG(total_compras), 2) AS avg_total_compras,
        ROUND(AVG(frecuencia_de_compra), 2) AS avg_freq,
        ROUND(AVG(promociones_utilizadas), 2) AS avg_promos,
        COUNT(*) AS n
    FROM clientes
    GROUP BY 1,2
    ORDER BY n DESC;
    """
elif dialect in ("postgresql", "postgres"):
    sql = """
    SELECT
        UPPER(genero) AS genero,
        UPPER(marca_preferida) AS marca,
        AVG(total_compras)::NUMERIC(12,2) AS avg_total_compras,
        AVG(frecuencia_de_compra)::NUMERIC(12,2) AS avg_freq,
        AVG(promociones_utilizadas)::NUMERIC(12,2) AS avg_promos,
        COUNT(*) AS n
    FROM clientes
    GROUP BY 1,2
    ORDER BY 6 DESC;
    """
else:
    sql = """
    SELECT
        UPPER(genero) AS genero,
        UPPER(marca_preferida) AS marca,
        CAST(AVG(total_compras) AS DECIMAL(12,2)) AS avg_total_compras,
        CAST(AVG(frecuencia_de_compra) AS DECIMAL(12,2)) AS avg_freq,
        CAST(AVG(promociones_utilizadas) AS DECIMAL(12,2)) AS avg_promos,
        COUNT(*) AS n
    FROM clientes
    GROUP BY 1,2
    ORDER BY 6 DESC;
    """

with st.spinner("⏳ Consultando KPIs por segmento..."):
    kpi_df = pd.read_sql(text(sql), eng)

with st.expander("📊 **Tabla Completa: KPIs por Marca × Género**"):
    st.dataframe(kpi_df.style.background_gradient(subset=['avg_total_compras', 'avg_promos'], cmap='RdYlGn'), 
                 use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    st.html('<div class="chart-container">')
    st.html('<p class="chart-title">📈 Promedio de Compras por Marca (Top 10)</p>')
    
    marca_avg = kpi_df.groupby("marca")["avg_total_compras"].mean().sort_values(ascending=True).tail(10)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(marca_avg)))
    bars = ax.barh(marca_avg.index, marca_avg.values, color=colors)
    
    for bar, val in zip(bars, marca_avg.values):
        ax.text(val + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{val:.2f}', va='center', fontweight='bold', fontsize=9)
    
    ax.set_xlabel("Promedio Total Compras", fontsize=12, fontweight='bold')
    ax.set_ylabel("")
    ax.grid(axis='x', alpha=0.3)
    st.pyplot(fig, clear_figure=True)
    
    top_marca = marca_avg.index[-1]
    top_valor = marca_avg.values[-1]
    
    st.html(f"""
    <div class="chart-insight">
        <strong>💡 Hallazgo:</strong> {top_marca} lidera con {top_valor:.2f} compras promedio. 
        Cliente objetivo para campañas premium y pruebas de nuevos productos.
    </div>
    """)
    st.html('</div>')

with col2:
    st.html('<div class="chart-container">')
    st.html('<p class="chart-title">🎁 Uso de Promociones por Marca (Top 10)</p>')
    
    promo_avg = kpi_df.groupby("marca")["avg_promos"].mean().sort_values(ascending=True).tail(10)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.cool(np.linspace(0.3, 0.9, len(promo_avg)))
    bars = ax.barh(promo_avg.index, promo_avg.values, color=colors)
    
    for bar, val in zip(bars, promo_avg.values):
        ax.text(val + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{val:.2f}', va='center', fontweight='bold', fontsize=9)
    
    ax.set_xlabel("Promedio Promociones Utilizadas", fontsize=12, fontweight='bold')
    ax.set_ylabel("")
    ax.grid(axis='x', alpha=0.3)
    st.pyplot(fig, clear_figure=True)
    
    top_promo = promo_avg.index[-1]
    top_promo_val = promo_avg.values[-1]
    
    st.html(f"""
    <div class="chart-insight">
        <strong>💡 Hallazgo:</strong> {top_promo} muestra mayor sensibilidad a promociones ({top_promo_val:.2f} promedio). 
        Evaluar rentabilidad y probar estrategias alternativas de valor.
    </div>
    """)
    st.html('</div>')


# -------------------------
# SECCIÓN 7: PREDICCIONES
# -------------------------
st.html("""
<div class="section-header-container">
    <div class="section-number">7</div>
    <div>
        <h2 class="section-title">Scoring y Predicciones</h2>
    </div>
</div>
<p class="section-subtitle">Aplicando MELI-Boost v1 sobre la base completa de clientes</p>
""")

st.html("""
<div class="narrative-box">
    <h4>🎯 Del Insight a la Ejecución</h4>
    <p>El scoring permite <strong>priorizar acciones</strong> basadas en valor esperado:</p>
    <ul>
        <li>Identificar clientes de alto potencial para cross-selling</li>
        <li>Segmentar audiencias por deciles de predicción</li>
        <li>Personalizar ofertas según perfil predictivo</li>
        <li>Medir uplift de campañas vs predicción baseline</li>
    </ul>
</div>
""")

col1, col2 = st.columns([3, 1])

with col1:
    if st.button("🚀 **Generar Predicciones con MELI-Boost v1**", use_container_width=True):
        with st.spinner("⏳ Ejecutando modelo sobre {} registros...".format(len(df))):
            df_pred = score_df(model, df)
            out_path = OUTPUTS_DIR / f"predicciones_meliboost_v1_{model_path.parent.name}.csv"
            df_pred.to_csv(out_path, index=False)
        
        st.success(f"✅ **Predicciones generadas exitosamente**")
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("📊 Predicción Promedio", f"{df_pred['pred_total_compras'].mean():.2f}", 
                     delta=f"{df_pred['pred_total_compras'].mean() - df[tc_col].mean():.2f} vs real")
        with col_b:
            st.metric("📈 Predicción Máxima", f"{df_pred['pred_total_compras'].max():.2f}")
        with col_c:
            st.metric("📉 Predicción Mínima", f"{df_pred['pred_total_compras'].min():.2f}")
        
        with st.expander("👁️ **Visualizar predicciones**"):
            display_cols = ['pred_total_compras'] + [col for col in df_pred.columns if col != 'pred_total_compras'][:6]
            st.dataframe(df_pred[display_cols].head(25).style.background_gradient(subset=['pred_total_compras'], cmap='RdYlGn'), 
                        use_container_width=True)
        
        st.info(f"💾 **Archivo guardado:** `{out_path}`")

with col2:
    st.html("""
    <div class="metric-card" style="background: linear-gradient(135deg, #48bb78 0%, #38a169 100%); color: white; height: 100%; display: flex; flex-direction: column; justify-content: center;">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">💾</div>
        <div style="color: rgba(255,255,255,0.95); font-size: 0.9rem; font-weight: 600; margin-bottom: 0.5rem;">OPCIÓN AVANZADA</div>
        <div style="font-size: 1rem; font-weight: 600; line-height: 1.4;">Guarda predicciones directamente en SQL para integración</div>
    </div>
    """)
    
    if st.toggle("💾 **Guardar en Base de Datos**", value=False):
        try:
            with st.spinner("⏳ Insertando predicciones en SQL..."):
                if 'df_pred' in locals():
                    upsert_predictions(df_pred, table_name="predicciones_meliboost_v1")
                    st.success("✅ **Datos persistidos en SQL**")
                else:
                    st.warning("⚠️ Primero genera las predicciones")
        except Exception as e:
            st.error(f"❌ Error al guardar: {e}")


# -------------------------
# SECCIÓN 8: PLAN DE ACCIÓN
# -------------------------
st.html("""
<div class="section-header-container">
    <div class="section-number">8</div>
    <div>
        <h2 class="section-title">Plan de Acción Estratégico</h2>
    </div>
</div>
<p class="section-subtitle">Recomendaciones tácticas vinculadas a hallazgos cuantitativos</p>
""")

st.html("""
<div class="narrative-box">
    <h4>🎯 De los Datos a las Decisiones</h4>
    <p>Cada insight debe traducirse en <strong>acciones concretas y medibles</strong>. 
    Las siguientes recomendaciones están respaldadas por el análisis y priorizadas por impacto esperado.</p>
</div>
""")

# Recomendaciones en cards
col1, col2 = st.columns(2)

with col1:
    st.html("""
    <div class="chart-container">
        <div class="chart-title">🎁 1. Promociones Inteligentes</div>
        
        <div style="margin: 1rem 0;">
            <strong style="color: #667eea;">📍 Dónde Actuar:</strong>
            <ul style="margin: 0.5rem 0; line-height: 1.8;">
                <li>Aumentar intensidad en segmentos donde el modelo <strong>subestima</strong> (deciles con bias negativo)</li>
                <li>En marcas con alto <code>avg_promos</code> y bajo <code>avg_total_compras</code></li>
            </ul>
        </div>
        
        <div style="margin: 1rem 0;">
            <strong style="color: #667eea;">🎯 Tácticas:</strong>
            <ul style="margin: 0.5rem 0; line-height: 1.8;">
                <li>Probar rebalanceo: menos descuentos, más beneficios no transaccionales</li>
                <li>Bundles inteligentes basados en marca preferida</li>
                <li>Loyalty tiers con recompensas progresivas</li>
            </ul>
        </div>
        
        <div style="margin: 1rem 0;">
            <strong style="color: #667eea;">📊 KPI de Éxito:</strong>
            <ul style="margin: 0.5rem 0; line-height: 1.8;">
                <li>↑ Ratio compras/promo utilizada</li>
                <li>↑ Margen bruto por transacción</li>
                <li>↓ Dependencia de descuentos (avg_promos)</li>
            </ul>
        </div>
    </div>
    """)
    
    st.html("""
    <div class="chart-container">
        <div class="chart-title">👥 3. Fidelización Premium</div>
        
        <div style="margin: 1rem 0;">
            <strong style="color: #667eea;">🎯 Audiencia Objetivo:</strong>
            <ul style="margin: 0.5rem 0; line-height: 1.8;">
                <li>Top decile de predicción (mayor valor esperado)</li>
                <li>Alta <code>frecuencia_de_compra</code> + bajo <code>avg_promos</code></li>
                <li>Marcas premium con fidelidad orgánica</li>
            </ul>
        </div>
        
        <div style="margin: 1rem 0;">
            <strong style="color: #667eea;">💎 Iniciativas:</strong>
            <ul style="margin: 0.5rem 0; line-height: 1.8;">
                <li>Programa VIP con beneficios exclusivos</li>
                <li>Early access a productos nuevos</li>
                <li>Reconocimiento personalizado</li>
                <li>Servicio premium (envío express, soporte prioritario)</li>
            </ul>
        </div>
        
        <div style="margin: 1rem 0;">
            <strong style="color: #667eea;">📊 KPI de Éxito:</strong>
            <ul style="margin: 0.5rem 0; line-height: 1.8;">
                <li>↑ Customer Lifetime Value (CLV)</li>
                <li>↑ Retention rate del top decile</li>
                <li>↑ NPS segmento premium</li>
            </ul>
        </div>
    </div>
    """)

with col2:
    st.html("""
    <div class="chart-container">
        <div class="chart-title">📢 2. Marketing Dirigido</div>
        
        <div style="margin: 1rem 0;">
            <strong style="color: #667eea;">🎯 Segmentación Avanzada:</strong>
            <ul style="margin: 0.5rem 0; line-height: 1.8;">
                <li>Cohortes edad × marca × género con mayor valor esperado</li>
                <li>Look-alikes del top decile de predicción</li>
                <li>Microsegmentos por importancia de features</li>
            </ul>
        </div>
        
        <div style="margin: 1rem 0;">
            <strong style="color: #667eea;">🚀 Campañas:</strong>
            <ul style="margin: 0.5rem 0; line-height: 1.8;">
                <li><strong>Email:</strong> Personalizados por marca preferida</li>
                <li><strong>Push:</strong> Timing basado en frecuencia histórica</li>
                <li><strong>Retargeting:</strong> Audiencias de predicción media-alta</li>
                <li><strong>Cross-sell:</strong> Productos complementarios por marca</li>
            </ul>
        </div>
        
        <div style="margin: 1rem 0;">
            <strong style="color: #667eea;">📊 KPI de Éxito:</strong>
            <ul style="margin: 0.5rem 0; line-height: 1.8;">
                <li>↑ Conversion rate de campañas</li>
                <li>↓ CAC (Cost per Acquisition)</li>
                <li>↑ ROAS (Return on Ad Spend)</li>
            </ul>
        </div>
    </div>
    """)
    
    st.html("""
    <div class="chart-container">
        <div class="chart-title">🔄 4. Mejora Continua</div>
        
        <div style="margin: 1rem 0;">
            <strong style="color: #667eea;">🔬 Experimentación:</strong>
            <ul style="margin: 0.5rem 0; line-height: 1.8;">
                <li>A/B tests de campañas por decil de predicción</li>
                <li>Medir uplift real vs predicción baseline</li>
                <li>Validar hipótesis de feature importance</li>
            </ul>
        </div>
        
        <div style="margin: 1rem 0;">
            <strong style="color: #667eea;">🔄 Ciclo de Reentrenamiento:</strong>
            <ul style="margin: 0.5rem 0; line-height: 1.8;">
                <li>Reentrenamiento trimestral o ante drift detectado</li>
                <li>Incorporar features adicionales (recencia, estacionalidad)</li>
                <li>Monitorear fairness por género y marca continuamente</li>
            </ul>
        </div>
        
        <div style="margin: 1rem 0;">
            <strong style="color: #667eea;">📈 Escalamiento:</strong>
            <ul style="margin: 0.5rem 0; line-height: 1.8;">
                <li>MLOps: pipeline automatizado de retraining</li>
                <li>Model registry con versionado</li>
                <li>Alertas de drift y anomalías</li>
            </ul>
        </div>
    </div>
    """)

# Roadmap visual
st.markdown("### 🗺️ Roadmap de Implementación", unsafe_allow_html=True)

st.html("""
<div style="background: white; padding: 2rem; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem;">
        <div style="text-align: center; padding: 1rem; background: #f0fff4; border-radius: 8px; border-top: 3px solid #48bb78;">
            <div style="font-size: 1.5rem; font-weight: 700; color: #22543d; margin-bottom: 0.5rem;">Semana 1-2</div>
            <div style="font-size: 0.9rem; color: #2d3748; line-height: 1.6;">
                • Identificar top/bottom deciles<br>
                • Diseñar campañas piloto<br>
                • Configurar tracking
            </div>
        </div>
        
        <div style="text-align: center; padding: 1rem; background: #ebf8ff; border-radius: 8px; border-top: 3px solid #3182ce;">
            <div style="font-size: 1.5rem; font-weight: 700; color: #2c5282; margin-bottom: 0.5rem;">Semana 3-4</div>
            <div style="font-size: 0.9rem; color: #2d3748; line-height: 1.6;">
                • Lanzar campañas segmentadas<br>
                • A/B tests por decil<br>
                • Monitorear KPIs
            </div>
        </div>
        
        <div style="text-align: center; padding: 1rem; background: #faf5ff; border-radius: 8px; border-top: 3px solid #9f7aea;">
            <div style="font-size: 1.5rem; font-weight: 700; color: #553c9a; margin-bottom: 0.5rem;">Mes 2</div>
            <div style="font-size: 0.9rem; color: #2d3748; line-height: 1.6;">
                • Analizar resultados<br>
                • Optimizar estrategias<br>
                • Escalar exitosas
            </div>
        </div>
        
        <div style="text-align: center; padding: 1rem; background: #fffaf0; border-radius: 8px; border-top: 3px solid #ed8936;">
            <div style="font-size: 1.5rem; font-weight: 700; color: #7c2d12; margin-bottom: 0.5rem;">Trimestral</div>
            <div style="font-size: 0.9rem; color: #2d3748; line-height: 1.6;">
                • Reentrenar modelo<br>
                • Incorporar learnings<br>
                • Actualizar features
            </div>
        </div>
    </div>
</div>
""")


# -------------------------
# SECCIÓN 9: LABORATORIO SQL
# -------------------------
st.html("""
<div class="section-header-container">
    <div class="section-number">9</div>
    <div>
        <h2 class="section-title">Laboratorio SQL Interactivo</h2>
    </div>
</div>
<p class="section-subtitle">Exploración ad-hoc con seguridad (solo lectura)</p>
""")

st.html("""
<div class="narrative-box">
    <h4>🔬 Empoderamiento Analítico</h4>
    <p>Este espacio permite al equipo de negocio <strong>responder preguntas no previstas</strong> sin dependencias técnicas:</p>
    <ul>
        <li>Consultas 100% desde SQL (single source of truth)</li>
        <li>Modo solo lectura (seguridad garantizada)</li>
        <li>Ideal para análisis exploratorios y validaciones rápidas</li>
    </ul>
</div>
""")

default_sql = """-- Consulta de ejemplo: Análisis de valor por segmento
SELECT 
    genero, 
    marca_preferida, 
    AVG(total_compras) AS avg_total, 
    AVG(frecuencia_de_compra) AS avg_freq,
    AVG(promociones_utilizadas) AS avg_promos,
    COUNT(*) AS n_clientes
FROM clientes
GROUP BY 1, 2
HAVING COUNT(*) >= 10
ORDER BY avg_total DESC
LIMIT 20;"""

user_sql = st.text_area("✍️ **Escribe tu consulta SQL**", value=default_sql, height=200)

col1, col2 = st.columns([3, 1])

with col1:
    if st.button("▶️ **Ejecutar Consulta**", use_container_width=False):
        if any(x in user_sql.lower() for x in ["drop", "delete", "update", "insert", "alter", "create"]):
            st.error("⚠️ **Operación bloqueada:** Solo se permiten consultas SELECT por seguridad.")
        else:
            try:
                with st.spinner("⏳ Ejecutando consulta en SQL..."):
                    dfq = run_sql_df(user_sql)
                
                st.success(f"✅ **Consulta ejecutada exitosamente:** {len(dfq)} filas retornadas")
                
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("📊 Filas", len(dfq))
                with col_b:
                    st.metric("📋 Columnas", len(dfq.columns))
                with col_c:
                    st.metric("💾 Tamaño", f"{dfq.memory_usage(deep=True).sum() / 1024:.1f} KB")
                
                st.dataframe(dfq, use_container_width=True)
                
                # Opción de descarga
                csv = dfq.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Descargar Resultados CSV",
                    data=csv,
                    file_name="consulta_sql_resultados.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"❌ **Error en la consulta:**\n```\n{str(e)}\n```")

with col2:
    st.html("""
    <div class="callout callout-info" style="height: 100%;">
        <strong>💡 Tips SQL:</strong><br>
        • Usa <code>LIMIT</code> para resultados grandes<br>
        • <code>GROUP BY</code> para agregaciones<br>
        • <code>WHERE</code> para filtros<br>
        • <code>ORDER BY</code> para ordenar
    </div>
    """)

with st.expander("📚 **Ejemplos de Consultas Útiles**"):
    ejemplos = {
        # 1) TOP clientes por compras (sin usar cliente_id/rowid)
        "Top clientes por compras": """
SELECT
  edad,
  genero,
  marca_preferida,
  total_compras
FROM clientes
WHERE total_compras IS NOT NULL
ORDER BY total_compras DESC
LIMIT 20;
""",

        # 2) (esta ya funcionaba) Análisis de promociones efectivas
        "Análisis de promociones efectivas": """
SELECT 
    marca_preferida,
    ROUND(AVG(total_compras), 2) AS avg_compras,
    ROUND(AVG(promociones_utilizadas), 2) AS avg_promos,
    ROUND(AVG( (total_compras * 1.0) / NULLIF(promociones_utilizadas, 0) ), 2) AS eficiencia_promo
FROM clientes
GROUP BY marca_preferida
ORDER BY (eficiencia_promo IS NULL), eficiencia_promo DESC;
""",

        # 3) (esta ya funcionaba) Segmentación por edad
        "Segmentación por edad": """
SELECT 
    CASE 
        WHEN edad < 25 THEN '18-24'
        WHEN edad < 35 THEN '25-34'
        WHEN edad < 45 THEN '35-44'
        WHEN edad < 55 THEN '45-54'
        ELSE '55+'
    END AS rango_edad,
    COUNT(*) AS n_clientes,
    ROUND(AVG(total_compras), 2) AS avg_compras,
    ROUND(AVG(frecuencia_de_compra), 2) AS avg_frecuencia
FROM clientes
GROUP BY rango_edad
ORDER BY CASE rango_edad
    WHEN '18-24' THEN 1
    WHEN '25-34' THEN 2
    WHEN '35-44' THEN 3
    WHEN '45-54' THEN 4
    WHEN '55+'  THEN 5
  END;
""",

        # 4) Clientes de alto valor con bajo engagement (sin usar cliente_id/rowid)
        "Clientes de alto valor bajo engagement": """
SELECT 
    edad,
    genero,
    marca_preferida, 
    total_compras,
    frecuencia_de_compra
FROM clientes
WHERE total_compras IS NOT NULL
  AND total_compras > (
        SELECT AVG(COALESCE(total_compras, 0)) * 1.5
        FROM clientes
    )
  AND frecuencia_de_compra < (
        SELECT AVG(COALESCE(frecuencia_de_compra, 0))
        FROM clientes
    )
ORDER BY total_compras DESC
LIMIT 30;
"""
    }

    for titulo, query in ejemplos.items():
        if st.button(f"📋 {titulo}", key=f"ejemplo_{titulo}"):
            st.code(query, language="sql")



# -------------------------
# SECCIÓN 10: METODOLOGÍA TÉCNICA
# -------------------------
if show_technical:
    st.html("""
    <div class="section-header-container">
        <div class="section-number">10</div>
        <div>
            <h2 class="section-title">Apéndice: Metodología Técnica</h2>
        </div>
    </div>
    <p class="section-subtitle">Detalles de implementación para el equipo técnico</p>
    """)
    
    tab1, tab2, tab3, tab4 = st.tabs(["🏗️ Arquitectura", "🔧 Pipeline", "📊 Validación", "⚠️ Limitaciones"])
    
    with tab1:
        st.html("""
        <div class="narrative-box">
            <h4>🏗️ Arquitectura de la Solución</h4>
            
            <pre style="background: #f7fafc; padding: 1rem; border-radius: 8px; overflow-x: auto;">
┌─────────────────────────────────────────────────────────────┐
│                     CAPA DE DATOS (SQL)                      │
│  • Single source of truth                                    │
│  • SQLite local / PostgreSQL                                 │
│  • Tabla: clientes                                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  CAPA DE FEATURES (Python)                   │
│  • clean_cols(): estandarización                             │
│  • NUM_COLS: edad, frecuencia, promociones                   │
│  • CAT_COLS: genero, marca_preferida                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              ENTRENAMIENTO (train_model.py)                  │
│  • Random search hiperparámetros                             │
│  • KFold CV (k=5)                                            │
│  • Entrenamiento final con best params                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  ARTEFACTOS VERSIONADOS                      │
│  • model_catboost.cbm                                        │
│  • metrics.json                                              │
│  • feature_importances.csv                                   │
│  • segment_metrics_*.csv                                     │
│  • deciles_diagnostics.csv                                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              APLICACIÓN (Streamlit Dashboard)                │
│  • Narrativa del proceso                                     │
│  • Scoring on-demand                                         │
│  • Exploración SQL segura                                    │
└─────────────────────────────────────────────────────────────┘
            </pre>
        </div>
        """)
    
    with tab2:
        st.html("""
        <div class="narrative-box">
            <h4>🔧 Pipeline de Entrenamiento</h4>
        </div>
        """)
        
        if show_code:
            st.code("""
# Pseudocódigo del pipeline
def train_model():
    # 1. Carga y preparación
    df = load_clients()  # desde SQL
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(...)
    
    # 2. Random search con CV
    param_distributions = {
        'depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'l2_leaf_reg': [1, 3, 5, 7],
        ...
    }
    
    search = RandomizedSearchCV(
        estimator=CatBoostRegressor(),
        param_distributions=param_distributions,
        cv=KFold(n_splits=5),
        scoring='r2',
        n_iter=50,
        random_state=42
    )
    
    search.fit(X_train, y_train)
    
    # 3. Entrenamiento final
    best_model = CatBoostRegressor(**search.best_params_)
    best_model.fit(X_train, y_train)
    
    # 4. Evaluación completa
    y_pred = best_model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred)
    segment_metrics = compute_segment_metrics(X_test, y_test, y_pred)
    deciles = compute_decile_diagnostics(y_test, y_pred)
    
    # 5. Guardado de artefactos
    save_artifacts(best_model, metrics, segment_metrics, deciles)
            """, language="python")
    
    with tab3:
        st.html("""
        <div class="narrative-box">
            <h4>📊 Estrategia de Validación</h4>
            
            <p><strong>Validación Cruzada (CV):</strong></p>
            <ul>
                <li>KFold con k=5 para reducir varianza</li>
                <li>Cada fold: 80% train, 20% validation</li>
                <li>Métricas promediadas: R², MAE, RMSE</li>
                <li>Guardamos todos los trials para auditoría</li>
            </ul>
            
            <p><strong>Holdout Test:</strong></p>
            <ul>
                <li>20% del dataset original</li>
                <li>Nunca visto durante búsqueda de hiperparámetros</li>
                <li>Smoke test final de generalización</li>
            </ul>
            
            <p><strong>Diagnósticos Adicionales:</strong></p>
            <ul>
                <li><strong>Deciles:</strong> Detecta sobre/subpredicción sistemática</li>
                <li><strong>Segmentos:</strong> Fairness por marca y género</li>
                <li><strong>Residuales:</strong> Patrones en errores (heteroscedasticidad)</li>
            </ul>
        </div>
        """)
    
    with tab4:
        st.html("""
        <div class="narrative-box">
            <h4>⚠️ Limitaciones y Supuestos</h4>
            
            <p><strong>Calidad de Datos:</strong></p>
            <ul>
                <li>Asumimos que <code>promociones_utilizadas</code> captura todas las redenciones</li>
                <li>Si hay promociones no registradas, puede sesgar conclusiones</li>
                <li>Missing values imputados pueden introducir bias</li>
            </ul>
            
            <p><strong>Tamaño Muestral:</strong></p>
            <ul>
                <li>Con datasets pequeños, métricas tienen alta varianza entre splits</li>
                <li>Guardamos CV completa para entender estabilidad</li>
                <li>Interpretaciones estadísticas requieren intervalos de confianza</li>
            </ul>
            
            <p><strong>Generalización:</strong></p>
            <ul>
                <li>Modelo aprende del histórico</li>
                <li>Cambios de mix de productos requieren retraining</li>
                <li>Nuevas marcas no vistas necesitan estrategia de cold-start</li>
            </ul>
            
            <p><strong>Ética y Fairness:</strong></p>
            <ul>
                <li>Monitoreamos métricas por género para detectar sesgos</li>
                <li>No usamos variables protegidas como features si no es legal</li>
                <li>Decisiones basadas en modelo deben ser auditables</li>
            </ul>
        </div>
        """)
        
        st.warning("""
        ⚠️ **Disclaimer:** Este modelo es una herramienta de apoyo a la decisión, no un oráculo. 
        Las recomendaciones deben validarse con pruebas A/B y considerando contexto de negocio completo.
        """)


# -------------------------
# FOOTER
# -------------------------
st.html("---")
st.html("""
<div style='text-align: center; padding: 3rem 0 2rem 0;'>
    <div style='margin-bottom: 1.5rem;'>
        <span style='font-size: 2rem; font-weight: 800; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                     -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            MELI-Boost Intelligence Platform
        </span>
    </div>
    
    <div style='color: #718096; font-size: 1rem; line-height: 1.8; max-width: 600px; margin: 0 auto;'>
        <p style='margin: 0.5rem 0;'><strong>Modelo:</strong> MELI-Boost v1 (CatBoost Regressor)</p>
        <p style='margin: 0.5rem 0;'><strong>Arquitectura:</strong> End-to-end desde SQL con validación multinivel</p>
        <p style='margin: 0.5rem 0;'><strong>Objetivo:</strong> Maximizar valor del cliente mediante decisiones basadas en datos</p>
    </div>
    
    <div style='margin-top: 2rem; padding-top: 1.5rem; border-top: 1px solid #e2e8f0; color: #a0aec0; font-size: 0.9rem;'>
        Desarrollado con ❤️ usando Python, CatBoost, Streamlit y SQL
    </div>
</div>
""")