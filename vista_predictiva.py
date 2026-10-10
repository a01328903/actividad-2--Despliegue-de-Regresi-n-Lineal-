"""Interfaz de la Etapa II: correlaciones, regresiones y evaluación temporal."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from modelado import CHANNELS, DEFAULT_MULTIPLE, load_monthly, run_analysis

RED, INK, BLUE = "#D9082F", "#454B57", "#236B8E"


def show(fig, key=None):
    fig.update_layout(template="plotly_white", paper_bgcolor="white", plot_bgcolor="white",
                      font=dict(color="#211C1D", size=12), margin=dict(l=20, r=25, t=35, b=35),
                      legend=dict(orientation="h", y=-.2), height=410)
    st.plotly_chart(fig, width="stretch", theme=None, key=key, config={"displaylogo": False})


def fmt(value):
    return f"{value:,.3f}" if np.isfinite(value) else "No definido"


def model_view(name, result, train, test, target):
    st.subheader(f"Regresión lineal {name.lower()}")
    c1, c2, c3 = st.columns(3)
    for column, metric in zip([c1, c2, c3], ["R²", "MAE", "RMSE"]):
        column.metric(f"{metric} · prueba", fmt(result["test_scores"][metric]))
    equation = f"{target}(t) = {result['model'].intercept_:.3f}"
    for feature, coefficient in zip(result["features"], result["model"].coef_):
        equation += f" {coefficient:+.3f} × {feature}"
    st.code(equation, language=None)
    if name == "Simple":
        feature = result["features"][0]
        figure = go.Figure()
        figure.add_scatter(x=train[feature], y=train[target], mode="markers", name="Real · entrenamiento", marker=dict(color=INK, opacity=.55))
        figure.add_scatter(x=test[feature], y=test[target], mode="markers", name="Real · prueba", marker=dict(color=BLUE, size=10))
        figure.add_scatter(x=test[feature], y=result["test_prediction"], mode="markers", name="Predicho · prueba", marker=dict(color=RED, symbol="x", size=11))
        grid = np.linspace(min(train[feature].min(), test[feature].min()), max(train[feature].max(), test[feature].max()), 100)
        figure.add_scatter(x=grid, y=result["model"].predict(pd.DataFrame({feature: grid})), mode="lines", name="Recta ajustada", line=dict(color=RED))
        figure.update_layout(title="Dispersión real y predicha superpuesta", xaxis_title=feature, yaxis_title=f"{target} (t)")
        show(figure, "simple_dispersion")
    figure = go.Figure()
    figure.add_scatter(x=train.Periodo, y=train[target], mode="markers", name="Real · entrenamiento", marker=dict(color=INK, opacity=.45))
    figure.add_scatter(x=test.Periodo, y=test[target], mode="markers+lines", name="Real · prueba", marker=dict(color=BLUE, size=10), line=dict(color=BLUE))
    figure.add_scatter(x=test.Periodo, y=result["test_prediction"], mode="markers+lines", name="Predicho · prueba", marker=dict(color=RED, symbol="x", size=11), line=dict(color=RED, dash="dot"))
    figure.add_vline(x=float(test.Periodo.min()) - .5, line_dash="dash", line_color="#87919E")
    figure.update_layout(title="Valores reales y predichos por periodo", xaxis_title="Periodo consecutivo", yaxis_title=f"Leads {target}")
    show(figure, f"time_{name}")
    if name == "Múltiple":
        st.caption("La superposición por periodo muestra la predicción conjunta de todos los predictores. Una sola recta en dos dimensiones no representa una regresión múltiple.")
    left, right = st.columns(2)
    with left:
        fit = go.Figure(go.Scatter(x=test[target], y=result["test_prediction"], mode="markers", marker=dict(color=RED, size=10), name="Mes de prueba"))
        low = min(test[target].min(), result["test_prediction"].min())
        high = max(test[target].max(), result["test_prediction"].max())
        fit.add_scatter(x=[low, high], y=[low, high], mode="lines", name="Predicción perfecta", line=dict(color=INK, dash="dash"))
        fit.update_layout(title="Real frente a predicho · prueba", xaxis_title="Valor real", yaxis_title="Valor predicho")
        show(fit, f"calibration_{name}")
    with right:
        residual = go.Figure(go.Scatter(x=result["test_prediction"], y=test[target].to_numpy() - result["test_prediction"], mode="markers", marker=dict(color=BLUE, size=10)))
        residual.add_hline(y=0, line_color=INK, line_dash="dash")
        residual.update_layout(title="Residuos · prueba", xaxis_title="Valor predicho", yaxis_title="Real − predicho", showlegend=False)
        show(residual, f"residual_{name}")
    with st.expander("Coeficientes e interpretación", expanded=True):
        st.dataframe(result["coefficients"], hide_index=True, width="stretch", column_config={"Coeficiente": st.column_config.NumberColumn(format="%.4f")})
        st.write("Cada coeficiente expresa el cambio estimado en leads del mes actual por una unidad adicional del predictor del mes anterior, manteniendo los demás predictores constantes. Describe una asociación, no un efecto causal.")
        st.caption(f"R² en entrenamiento: {fmt(result['train_scores']['R²'])}. Número de condición de predictores estandarizados: {fmt(result['condition'])}.")
        if result["rank"] < len(result["features"]) + 1 or result["condition"] > 30:
            st.warning("Los predictores presentan colinealidad: los coeficientes pueden ser inestables. Considera reducir variables.")
    if (result["test_prediction"] < 0).any():
        st.warning("El ajuste lineal produjo conteos negativos. Se conservan para evaluar el error sin ocultarlo; no son conteos operativamente válidos.")


def render_predictive():
    st.caption("ETAPA II · MODELADO PREDICTIVO")
    st.title("Correlaciones y regresión")
    st.write("Estima los leads de un canal en el siguiente mes a partir de los conteos del mes anterior. Explora los datos y compara la regresión simple con la múltiple.")
    try:
        frame = load_monthly()
    except (OSError, ValueError) as error:
        st.error(f"No se pudo cargar data/leads_mensuales.csv: {error}")
        return
    # Reserva el área superior para los resultados; los controles quedan al final.
    results_area = st.container()
    with st.container(border=True):
        st.subheader("Configurar las regresiones")
        c1, c2, c3 = st.columns([1, 1, 2])
        target = c1.selectbox("Canal que se quiere predecir (Y)", CHANNELS, index=4)
        test_percent = c2.slider("Meses reservados para prueba (%)", 20, 35, 20, 5)
        features = [f"{c} (t−1)" for c in CHANNELS]
        simple = c3.selectbox("Predictor de la regresión simple (X)", features, index=4)
        multiple = st.multiselect("Predictores de la regresión múltiple · elige entre 2 y 6", features, default=DEFAULT_MULTIPLE)
    with results_area:
        if len(multiple) < 2:
            st.info("Selecciona al menos dos predictores para comparar ambos modelos.")
            return
        try:
            table, train, test, results, comparison, predictions = run_analysis(frame, target, simple, multiple, test_percent / 100)
        except ValueError as error:
            st.warning(str(error))
            return
        for col, label, value in zip(st.columns(4), ["Meses en la fuente", "Observaciones con rezago", "Meses de entrenamiento", "Meses de prueba"], [len(frame), len(table), len(train), len(test)]):
            col.metric(label, value)
        st.caption(f"Entrenamiento: periodos {train.Periodo.min()}–{train.Periodo.max()}. Prueba: {test.Periodo.min()}–{test.Periodo.max()}. El primer mes sólo aporta predictores. La fuente no especifica los años: se conserva su orden y no se inventan fechas.")
        st.info("Evaluación a un mes de anticipación: para cada mes de prueba se usa el mes anterior ya observado. Los coeficientes permanecen fijos. Esto no es un pronóstico de todos los meses futuros realizado de una sola vez.")
        tabs = st.tabs(["Correlaciones", "Regresión simple", "Regresión múltiple", "Comparación", "Simulador", "Datos y método"])
        with tabs[0]:
            st.subheader("Relaciones entre canales")
            method = st.selectbox("Tipo de correlación", ["Pearson", "Spearman", "Kendall"])
            scope = st.radio("Datos para explorar", ["Sólo entrenamiento", "Todos los meses (exploratorio)"], horizontal=True)
            sample = frame[frame.Periodo <= train.Periodo.max()] if scope == "Sólo entrenamiento" else frame
            if scope != "Sólo entrenamiento":
                st.caption("Esta vista incluye los meses de prueba. No la uses para elegir predictores si deseas mantener esa evaluación independiente.")
            corr = sample[CHANNELS].corr(method=method.lower())
            figure = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale=[BLUE, "#FFFFFF", RED], aspect="auto", labels={"color": "Correlación"})
            figure.update_layout(title=f"{method} · {len(sample)} meses")
            show(figure, "heatmap")
            st.caption({"Pearson": "Pearson mide asociación lineal y puede ser sensible a valores extremos.", "Spearman": "Spearman mide asociación monótona usando rangos.", "Kendall": "Kendall compara la concordancia de pares y contempla empates."}[method] + " La correlación no demuestra causalidad; una tendencia compartida también puede elevarla.")
            st.subheader("Relación con la variable objetivo del mes siguiente")
            related = train[[target, *features]].corr(method=method.lower())[target].drop(target)
            st.dataframe(related.rename("Correlación con Y(t)").rename_axis("Predictor").reset_index(), hide_index=True, width="stretch")
            st.caption("Esta tabla utiliza exclusivamente entrenamiento y relaciona X(t−1) con Y(t). El mapa superior compara canales del mismo mes.")
            box_channels = st.multiselect("Canales para el boxplot", CHANNELS, default=["Piso", "Plazas", "Digitales"])
            if box_channels:
                long = sample.melt(id_vars="Periodo", value_vars=box_channels, var_name="Canal", value_name="Leads")
                show(px.box(long, x="Canal", y="Leads", points="all", color_discrete_sequence=[RED], title="Dispersión y valores extremos por canal"), "boxplot")
            st.caption("Los puntos extremos se conservan. Un boxplot no basta para concluir que un dato es incorrecto.")
        with tabs[1]:
            model_view("Simple", results["Simple"], train, test, target)
        with tabs[2]:
            model_view("Múltiple", results["Múltiple"], train, test, target)
        with tabs[3]:
            st.subheader("Evaluación en los mismos meses de prueba")
            st.dataframe(comparison, hide_index=True, width="stretch", column_config={c: st.column_config.NumberColumn(format="%.3f") for c in ["MAE", "RMSE", "R²", "R² entrenamiento"]})
            st.write("MAE: error absoluto promedio en leads. RMSE: penaliza más los errores grandes. R²: 1 es el ajuste perfecto; puede ser negativo. Persistencia: repetir el valor observado del mes anterior.")
            best = comparison.loc[comparison.MAE.idxmin()]
            st.write(f"En esta partición, el menor MAE corresponde a **{best['Modelo']}** ({best['MAE']:.2f} leads). Este resultado describe esta muestra y no garantiza el desempeño futuro.")
            if (comparison.loc[comparison.Modelo != "Persistencia", "R²"] < 0).all():
                st.warning("Ambas regresiones tienen R² negativo en prueba. Los resultados no respaldan una buena capacidad predictiva en este bloque temporal.")
            long = comparison.melt(id_vars="Modelo", value_vars=["MAE", "RMSE"], var_name="Métrica", value_name="Error (leads)")
            show(px.bar(long, x="Modelo", y="Error (leads)", color="Métrica", barmode="group", color_discrete_sequence=[RED, INK]), "comparison")
            st.dataframe(predictions, hide_index=True, width="stretch")
            st.download_button("Descargar predicciones de prueba", predictions.to_csv(index=False).encode("utf-8-sig"), "predicciones_prueba.csv", "text/csv")
            st.download_button("Descargar métricas", comparison.to_csv(index=False).encode("utf-8-sig"), "metricas_modelos.csv", "text/csv")
        with tabs[4]:
            st.subheader("Escenario de un mes siguiente")
            chosen = st.selectbox("Modelo para el escenario", ["Simple", "Múltiple"])
            selected = results[chosen]
            st.caption("Usa el modelo ajustado sólo con entrenamiento. Los valores iniciales son los del último mes de la fuente; las entradas representan un mes anterior conocido o hipotético.")
            inputs = {}
            outside = []
            for i, feature in enumerate(selected["features"]):
                channel = feature.split(" (")[0]
                value = st.number_input(feature, min_value=0.0, value=float(frame[channel].iloc[-1]), step=1.0, key=f"scenario_{chosen}_{feature}")
                inputs[feature] = value
                if value < train[feature].min() or value > train[feature].max():
                    outside.append(feature)
            estimate = selected["model"].predict(pd.DataFrame([inputs]))[0]
            st.metric(f"Estimación de leads · {target}", f"{estimate:.1f}")
            if outside:
                st.warning("Escenario fuera del rango de entrenamiento en: " + ", ".join(outside) + ". Es una extrapolación.")
            if estimate < 0:
                st.warning("El modelo produjo un conteo negativo, que no es operativamente válido.")
            st.caption("Estimación puntual sin intervalo de predicción. Considera los errores de prueba antes de utilizarla para planear.")
        with tabs[5]:
            st.subheader("Datos utilizados")
            st.dataframe(frame, hide_index=True, width="stretch")
            st.download_button("Descargar datos agregados", frame.to_csv(index=False).encode("utf-8-sig"), "leads_mensuales.csv", "text/csv")
            st.markdown("""
    **Fuente y unidad de análisis.** CSV original «Leads por Canal», encabezado en la cuarta fila. Cada observación es un mes, en el orden de las primeras 30 columnas mensuales. Se transponen los seis canales: Piso, Plazas, Eventos, Planta, Digitales y Otros. Se excluyen Total, Leads GAC, objetivos y proyecciones para evitar relaciones aritméticas artificiales. El archivo de trabajo no contiene datos personales.

    **Preparación.** Los ceros se conservan tal como aparecen en la fuente, incluido el primer mes con todos los canales en cero. Debe confirmarse con el responsable si representan actividad nula o información no capturada. No se imputan valores ni se eliminan observaciones por su magnitud. Los años no están identificados en el CSV.

    **Modelos.** Mínimos cuadrados ordinarios con intercepto. La regresión simple usa un predictor; la múltiple, dos o más. Todos son conteos del mes anterior. El objetivo es el conteo del canal seleccionado en el mes actual. La pérdida del primer mes al crear rezagos es esperada.

    **Evaluación.** Separación cronológica, sin mezclar meses al azar. Las dos regresiones y la persistencia se evalúan en las mismas observaciones. No se reentrenan con los meses de prueba. MAE y RMSE se expresan en leads; R² se deja sin definir si el objetivo de prueba es constante.

    **Alcance.** Sólo hay 30 meses de origen. Las tendencias, la estacionalidad, la autocorrelación y los cambios operativos pueden afectar el resultado. Los gráficos de residuos ayudan a detectar patrones, pero no prueban que se cumplan todos los supuestos. Cambiar predictores repetidamente para mejorar el bloque de prueba convierte ese bloque en validación; sería necesario reservar datos posteriores para una evaluación final.

    **Referencias.** [Regresión lineal](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html), [evaluación temporal](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split), [R²](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html).
    """)
