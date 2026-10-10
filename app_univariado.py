"""Dashboard GAC · análisis univariado de las bases comerciales."""
import base64
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "Base_Datos_GAC"
RED, INK, GRAY = "#D9082F", "#211C1D", "#87919E"
GREEN, GOLD = "#247A61", "#A86B15"
PALETTE = [RED, "#454B57", "#B97888", "#738A9B", "#A86B15"]
STATUS = {"Sí": GREEN, "No": GRAY, "Activo": RED, "Finalizado": GRAY, "Venta": GOLD}
FILES = {
    "Base de Leads": ("Base de Leads", ["Producto", "Temperatura", "Asesor"]),
    "Plazas": ("Plazas", ["Nombre", "Producto", "Origen", "Seguimiento", "Cita efetiva ", "PDM", "SDC"]),
    "Citas Digital": ("Citas Digital", ["BDC ", "Estatus de Lead", "Potencial de compra", "Venta", "Asesor Asignado"]),
    "Bitácora de Piso": ("Bitacora de Piso", ["Inter Gerente"]),
}

st.set_page_config(page_title="GAC | Gestión comercial", page_icon="📊", layout="wide")
st.markdown("""
<style>
.stApp { background: #F5F6F8; color: #211C1D; }
.block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem; }
h1 { font-size: 2.25rem !important; letter-spacing: -.055rem; }
h2 { font-size: 1.35rem !important; letter-spacing: -.02rem; }
h3 { font-size: 1.03rem !important; }
[data-testid="stHeader"] { background: #F5F6F8; }
[data-testid="stSidebar"] { border-right: 1px solid #302A2C; }
[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
[data-testid="stMetric"] { background: white; padding: 20px 22px;
 border: 1px solid #E4E6EB; border-radius: 14px; min-height: 125px;
 border-top: 3px solid #D9082F; }
[data-testid="stMetricLabel"] { color: #586170; }
[data-testid="stMetricValue"] { font-size: 1.85rem; font-weight: 650; }
[data-testid="stVerticalBlockBorderWrapper"] > div { border-radius: 14px; }
.eyebrow { color: #B50728; font-size: .75rem; font-weight: 750;
 letter-spacing: .14rem; margin-bottom: .5rem; }
.intro { color: #586170; font-size: 1rem; max-width: 850px; margin-bottom: 1.6rem; }
.brand { display:flex; align-items:center; gap:14px; margin-bottom:24px; }
.brand img { width:84px; height:84px; border-radius:12px; }
.brand strong { color:#FFFFFF; font-size:1rem; letter-spacing:.05rem; }
.brand small { display:block; color:#B9B2B5; font-size:.75rem; margin-top:4px; }
.sidebar-note { color:#B9B2B5; font-size:.78rem; line-height:1.6; }
.footer { color:#677180; border-top:1px solid #DEE1E7; padding-top:16px;
 margin-top:30px; font-size:.78rem; }
@media (max-width: 700px) {
 .block-container { padding: 1.3rem 1rem; }
 h1 { font-size: 1.75rem !important; }
 [data-testid="stMetric"] { padding: 14px; }
}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def read_base(name, modified):
    """La fecha de modificación invalida la caché al actualizar el CSV."""
    suffix, columns = FILES[name]
    frame = pd.read_csv(DATA / f"Analisis GAC Full7agosto 2026({suffix}).csv", usecols=columns, dtype="string")
    frame.columns = frame.columns.str.strip()
    for column in frame:
        frame[column] = frame[column].str.strip().replace("", pd.NA)
    if name == "Plazas":
        frame = frame.dropna(subset=["Nombre", "Producto", "Origen"])
    elif name == "Citas Digital":
        frame = frame.dropna(subset=["BDC"])
    elif name == "Base de Leads":
        frame["Asesor"] = frame["Asesor"].replace({
            "Alan GonzÃ¡lez Belanzate": "Alan González Belanzate",
            "Pedro  GonzÃ¡les EspÃ\xadndol": "Pedro Gonzáles Espíndol",
            "CESAR DE JESUS PESTAÃ‘A": "CESAR DE JESUS PESTAÑA",
        })
    return frame


def load(name):
    path = DATA / f"Analisis GAC Full7agosto 2026({FILES[name][0]}).csv"
    return read_base(name, path.stat().st_mtime_ns)


@st.cache_data(show_spinner=False)
def read_channels(modified):
    frame = pd.read_csv(DATA / "Analisis GAC Full7agosto 2026(Leads por Canal).csv", header=3)
    frame["Canal"] = frame["Canal"].str.strip()
    frame = frame[frame["Canal"].isin(["Piso", "Plazas", "Digitales"])]
    months = frame.iloc[:, 1:31].apply(pd.to_numeric, errors="coerce")
    return pd.DataFrame({"Categoría": frame["Canal"], "Cantidad": months.sum(axis=1)}).sort_values("Cantidad", ascending=False)


def channels():
    path = DATA / "Analisis GAC Full7agosto 2026(Leads por Canal).csv"
    return read_channels(path.stat().st_mtime_ns)


def yes_no(series):
    return series.str.lower().replace({"true": "Sí", "false": "No"})


def counts(series):
    return series.value_counts().rename_axis("Categoría").reset_index(name="Cantidad")


def number(value):
    return f"{value:,.0f}"


def rate(series):
    valid = yes_no(series).dropna()
    valid = valid[valid.isin(["Sí", "No"])]
    return f"{valid.eq('Sí').mean():.1%}" if len(valid) else "Sin datos"


def metrics(items):
    for column, (label, value, help_text) in zip(st.columns(len(items)), items):
        column.metric(label, value, help=help_text)
    st.write("")


def chart(title, data, kind="bar", color_map=None, height=370, note=None):
    with st.container(border=True):
        st.subheader(title)
        if data.empty or data["Cantidad"].sum() == 0:
            st.info("No hay datos disponibles para esta gráfica.")
            return
        common = dict(color_discrete_sequence=PALETTE)
        if color_map:
            common.update(color="Categoría", color_discrete_map=color_map)
        if kind == "donut":
            fig = px.pie(data, names="Categoría", values="Cantidad", hole=.68, **common)
            fig.update_traces(textinfo="percent", textposition="inside", sort=False,
                              hovertemplate="%{label}<br>%{value:,.0f} registros · %{percent}<extra></extra>")
            fig.update_layout(annotations=[dict(text=f"<b>{number(data['Cantidad'].sum())}</b><br>registros", x=.5, y=.5, showarrow=False, font_size=19)],
                              legend=dict(orientation="h", y=-.12, x=.5, xanchor="center"))
        else:
            horizontal = kind == "bar"
            fig = px.bar(data, x="Cantidad" if horizontal else "Categoría",
                         y="Categoría" if horizontal else "Cantidad", orientation="h" if horizontal else "v",
                         text="Cantidad", **common)
            fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside", cliponaxis=False,
                              marker_line_width=0, hovertemplate=("%{y}: %{x:,.0f}<extra></extra>" if horizontal else "%{x}: %{y:,.0f}<extra></extra>"))
            fig.update_layout(showlegend=False, bargap=.32)
            if horizontal:
                fig.update_yaxes(autorange="reversed", type="category", automargin=True, showgrid=False)
                fig.update_xaxes(range=[0, data["Cantidad"].max() * 1.22], gridcolor="#EEF0F4", title="Registros", tickformat=",.0f")
            else:
                fig.update_xaxes(type="category", showgrid=False)
                fig.update_yaxes(range=[0, data["Cantidad"].max() * 1.22], gridcolor="#EEF0F4", title="Registros", tickformat=",.0f")
        fig.update_layout(template="plotly_white", height=height, paper_bgcolor="white", plot_bgcolor="white",
                          font=dict(family="Arial, sans-serif", color=INK, size=12),
                          margin=dict(l=12, r=35, t=18, b=30),
                          xaxis_title=None if kind != "bar" else "Registros", yaxis_title=None if kind != "vertical" else "Registros")
        st.plotly_chart(fig, width="stretch", theme=None, config={"displayModeBar": False, "responsive": True})
        if note:
            st.caption(note)


with st.sidebar:
    logo = ROOT / "assets" / "gac-logo.png"
    if logo.exists():
        encoded = base64.b64encode(logo.read_bytes()).decode()
        st.markdown(f'<div class="brand"><img src="data:image/png;base64,{encoded}" alt="Logo GAC"><div><strong>GAC</strong><small>INTELIGENCIA COMERCIAL</small></div></div>', unsafe_allow_html=True)
    else:
        st.header("GAC · Gestión comercial")
    stage = st.radio("Etapa del proyecto", ["I · Análisis descriptivo", "II · Modelado predictivo"], index=1)
    base = st.radio("Explorar dashboard", ["Resumen", *FILES, "Leads por Canal"]) if stage.startswith("I ·") else "Modelado"
    st.divider()
    st.markdown('<div class="sidebar-note"><strong>GESTIÓN DE PROYECTOS</strong><br>Análisis univariado<br>Bases comerciales GAC</div>', unsafe_allow_html=True)

if stage.startswith("II ·"):
    from vista_predictiva import render_predictive
    render_predictive()
    st.stop()

if not DATA.exists():
    st.title("Análisis descriptivo")
    st.info("Esta etapa utiliza las bases originales de acceso local. Coloca Base_Datos_GAC junto a la aplicación. La Etapa II funciona con los datos mensuales agregados incluidos en el proyecto.")
    st.stop()

descriptions = {
    "Resumen": ("Panorama comercial", "Una vista general de las bases disponibles. Selecciona una sección para explorar sus distribuciones y resultados."),
    "Base de Leads": ("Perfil de los leads", "Distribución de prospectos por producto, nivel de interés y asesor responsable."),
    "Plazas": ("Actividad en plazas", "Origen de los prospectos y acciones registradas durante el seguimiento comercial."),
    "Citas Digital": ("Citas digitales", "Estatus, potencial de compra y asignación de los prospectos del canal digital."),
    "Bitácora de Piso": ("Atención en piso", "Participación del gerente en los registros de la bitácora de atención."),
    "Leads por Canal": ("Desempeño por canal", "Volumen acumulado de Piso, Plazas y Digitales en los meses incluidos en la base."),
}
title, description = descriptions[base]
st.markdown('<div class="eyebrow">GAC / ANÁLISIS UNIVARIADO</div>', unsafe_allow_html=True)
st.title(title)
st.markdown(f'<div class="intro">{description}</div>', unsafe_allow_html=True)

try:
    if base == "Resumen":
        with st.spinner("Preparando el panorama comercial…"):
            frames = {name: load(name) for name in FILES}
        metrics([
            ("Leads registrados", number(len(frames["Base de Leads"])), "Filas de la Base de Leads."),
            ("Registros en plazas", number(len(frames["Plazas"])), "Registros con nombre, producto y origen."),
            ("Citas digitales", number(len(frames["Citas Digital"])), "Registros con BDC asignado; no representa citas efectivas."),
            ("Atenciones con dato", number(frames["Bitácora de Piso"]["Inter Gerente"].notna().sum()), "Registros con información sobre intervención del gerente."),
        ])
        st.caption("Cada indicador corresponde a una base independiente; no se suman porque pueden contener los mismos prospectos.")
        left, right = st.columns([1.2, 1])
        with left:
            chart("Distribución por canal", channels(), color_map={"Piso": "#454B57", "Plazas": "#B97888", "Digitales": RED},
                  note="Acumulado de las 30 columnas mensuales utilizadas en el análisis original.")
        with right:
            chart("Productos con mayor interés", counts(frames["Base de Leads"]["Producto"]), note="Fuente: Base de Leads · Se excluyen valores vacíos.")
        with st.expander("Cómo leer este dashboard"):
            st.write("El rojo identifica la marca y destaca las distribuciones principales. El verde representa respuestas afirmativas, el gris respuestas negativas o estados finalizados y el ocre las ventas. Las etiquetas permiten interpretar cada gráfica sin depender del color.")
            st.write("Las gráficas muestran frecuencias, no relaciones causales. Los valores vacíos se omiten. Las bases conservan sus reglas originales de limpieza; las filas vacías de Plazas y Citas Digital se descartan según sus campos de identificación.")
    elif base == "Base de Leads":
        frame = load(base)
        metrics([("Leads registrados", number(len(frame)), "Total de filas de la base."), ("Productos", number(frame["Producto"].nunique()), "Productos distintos con dato."), ("Asesores", number(frame["Asesor"].nunique()), "Asesores distintos con dato.")])
        left, right = st.columns(2)
        with left:
            chart("Leads por producto", counts(frame["Producto"]))
        with right:
            chart("Nivel de interés", counts(frame["Temperatura"]), note="Comparación entre categorías de interés; no representa una tendencia temporal.")
        chart("Distribución por asesor", counts(frame["Asesor"]), height=max(420, frame["Asesor"].nunique() * 29))
    elif base == "Plazas":
        frame = load(base)
        metrics([("Registros válidos", number(len(frame)), "Con nombre, producto y origen."), ("Con seguimiento", rate(frame["Seguimiento"]), "Sí / respuestas Sí o No con dato."), ("Con cita efectiva", rate(frame["Cita efetiva"]), "Sí / respuestas Sí o No con dato."), ("Prueba de manejo", rate(frame["PDM"]), "Sí / respuestas Sí o No con dato.")])
        left, right = st.columns(2)
        with left:
            chart("Origen de los leads", counts(frame["Origen"]), "donut")
        with right:
            chart("Seguimiento comercial", counts(yes_no(frame["Seguimiento"])), "vertical", STATUS)
        left, right = st.columns(2)
        with left:
            chart("Cita efectiva", counts(yes_no(frame["Cita efetiva"])), "vertical", STATUS)
        with right:
            chart("Prueba de manejo", counts(yes_no(frame["PDM"])), "vertical", STATUS)
        chart("Solicitud de crédito", counts(yes_no(frame["SDC"])), "donut", STATUS)
    elif base == "Citas Digital":
        frame = load(base)
        status = frame["Estatus de Lead"].str.capitalize().replace({"Acitvo": "Activo", "Acrivo": "Activo", "Ativo": "Activo", "Activio": "Activo", "Activo-": "Activo", "Finaliazdo": "Finalizado"})
        status = status[status.isin(["Activo", "Finalizado", "Venta"])]
        potential = frame["Potencial de compra"].replace({"80.0%": "80%", "80€": "80%"})
        order = ["5%", "20%", "30%", "40%", "50%", "60%", "65%", "70%", "75%", "80%", "85%", "90%", "95%", "100%"]
        distribution = potential.value_counts().reindex(order, fill_value=0).rename_axis("Categoría").reset_index(name="Cantidad")
        metrics([("Registros digitales", number(len(frame)), "Registros con BDC asignado."), ("Leads activos", number(status.eq("Activo").sum()), "Estatus normalizado: Activo."), ("Ventas registradas", number(yes_no(frame["Venta"]).eq("Sí").sum()), "Respuestas afirmativas en la columna Venta."), ("Proporción de ventas", rate(frame["Venta"]), "Sí / respuestas Sí o No con dato en Venta.")])
        left, right = st.columns(2)
        with left:
            chart("Estatus del lead", counts(status), color_map=STATUS, note="Estatus reconocidos: Activo, Finalizado y Venta.")
        with right:
            chart("Potencial de compra", distribution, "vertical", note="Categorías de porcentaje incluidas en el análisis original.")
        left, right = st.columns(2)
        with left:
            chart("Leads que terminaron en venta", counts(yes_no(frame["Venta"])), "donut", STATUS)
        with right:
            bdc = frame["BDC"].replace({"ana": "Ana", "Antonio/Saul": "Antonio", "Igancio": "Ignacio"})
            chart("Asignación por BDC", counts(bdc[bdc.isin(["Ana", "Antonio", "Ignacio", "Saul"])]), note="Se conservan los cuatro BDC reconocidos en el análisis original.")
        chart("10 asesores con más leads", counts(frame["Asesor Asignado"]).head(10), height=440)
    elif base == "Bitácora de Piso":
        frame = load(base)
        valid = frame["Inter Gerente"].notna().sum()
        metrics([("Registros con dato", number(valid), "Registros con información de intervención."), ("Con intervención", number(yes_no(frame["Inter Gerente"]).eq("Sí").sum()), "Respuestas afirmativas."), ("Participación del gerente", rate(frame["Inter Gerente"]), "Sí / respuestas Sí o No con dato.")])
        chart("Intervención del gerente", counts(yes_no(frame["Inter Gerente"])), "vertical", STATUS)
        st.caption(f"{number(len(frame) - valid)} registros sin información de intervención se excluyen de la gráfica.")
    elif base == "Leads por Canal":
        data = channels()
        metrics([(row["Categoría"], number(row["Cantidad"]), "Acumulado de las columnas mensuales del análisis original.") for _, row in data.iterrows()])
        chart("Volumen acumulado por canal", data, color_map={"Piso": "#454B57", "Plazas": "#B97888", "Digitales": RED})
        st.caption("Se incluyen Piso, Plazas y Digitales. Se suman las mismas 30 columnas mensuales del dashboard original; se excluyen filas de totales y otros canales.")
except (FileNotFoundError, ValueError, KeyError, pd.errors.ParserError) as error:
    st.error("No fue posible cargar esta sección. Verifica que Base_Datos_GAC esté junto a la aplicación y conserve los archivos y columnas originales.")
    with st.expander("Detalle del problema"):
        st.code(str(error))

st.markdown('<div class="footer">GAC · Gestión de Proyectos &nbsp; / &nbsp; Análisis descriptivo de bases comerciales · Los valores vacíos se excluyen de las distribuciones.</div>', unsafe_allow_html=True)
