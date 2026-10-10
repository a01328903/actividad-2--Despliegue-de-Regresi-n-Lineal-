# GAC · Gestión de Proyectos

Dashboard de Streamlit con análisis descriptivo y **Etapa II: modelado predictivo**. Incluye regresión lineal simple, regresión lineal múltiple y evaluación en meses posteriores al entrenamiento.

## Ejecutar en Visual Studio Code

Abre esta carpeta en VS Code y selecciona un intérprete de Python 3.12. Desde su terminal:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app_univariado.py --server.port 8502
```

En Windows, activa el entorno con `.venv\Scripts\Activate.ps1`. Abre `http://localhost:8502`. Ejecuta desde la raíz del proyecto para cargar `.streamlit/config.toml`. Con las extensiones Python y Python Debugger instaladas también puedes usar la configuración de depuración incluida.

La Etapa II funciona con `data/leads_mensuales.csv`, incluido en el proyecto. La Etapa I requiere la carpeta original `Base_Datos_GAC`, conservada localmente y excluida de Git porque contiene información personal. Sin esa carpeta, la app informa cómo habilitar la Etapa I y mantiene disponible la Etapa II.

## Correspondencia con la actividad

| Requisito | Implementación |
|---|---|
| Análisis de correlaciones | Heatmap de Pearson, Spearman y Kendall; tabla de relaciones con el objetivo del siguiente mes |
| Regresión lineal simple | Un predictor con intercepto; ecuación, coeficientes y dispersión real/predicha superpuesta a la recta |
| Regresión lineal múltiple | Entre 2 y 6 predictores rezagados; ecuación, coeficientes y valores reales/predichos superpuestos por periodo |
| Evaluación | Separación cronológica, MAE, RMSE, R², comparación con persistencia y diagnóstico de residuos |
| Diseño y widgets adicionales | Marca GAC, navegación por etapas, pestañas, selectores, multiselectores, slider, entradas numéricas, heatmap, boxplot, tablas y descargas |
| Archivos para VS Code | Aplicación, módulo de modelado, notebook ejecutable, dependencias, configuración de depuración y pruebas |
| GitHub y Canvas | Este proyecto está preparado para versionarse. El enlace debe corresponder al repositorio publicado y accesible al docente; la entrega en Canvas es un paso separado |

## Datos y método

Fuente: `Analisis GAC Full7agosto 2026(Leads por Canal).csv`, encabezado en la cuarta fila. Se transponen las primeras **30 columnas mensuales** de los canales Piso, Plazas, Eventos, Planta, Digitales y Otros. Se excluyen las filas Total y Leads GAC y las columnas de objetivos y proyecciones. La tabla exportada contiene exclusivamente conteos agregados, número de periodo y mes.

La fuente no identifica los años. `Periodo` representa el orden consecutivo original, no una fecha inventada. Los ceros se conservan, incluido el primer mes, que tiene cero en todos los canales. Su significado debe confirmarse con el responsable de la captura. No se imputan valores ni se eliminan extremos.

Se predice **Y(t)** mediante **X(t−1)**. La regresión simple inicial usa Digitales del mes previo y la múltiple añade Plazas y Piso del mes previo. Al crear los rezagos quedan 29 observaciones. Con 20% de prueba se entrenan 23 observaciones (periodos 2–24) y se evalúan 6 (25–30). Los coeficientes se estiman solamente con entrenamiento. Cada predicción de prueba es a un paso, usando el mes anterior real ya observado; no se simula un horizonte completo sin observaciones intermedias.

El análisis exploratorio usa entrenamiento por defecto. El usuario puede visualizar todos los meses, pero elegir modelos tras examinar los datos de prueba compromete la independencia de esa evaluación. No se usan totales como predictores de sus componentes ni información del mismo mes del objetivo.

## Resultados de la configuración inicial

Objetivo: Digitales. Predictores y partición descritos arriba. Conteos sin estandarizar para conservar la interpretación en leads.

| Modelo | MAE (leads) | RMSE (leads) | R² de prueba |
|---|---:|---:|---:|
| Regresión simple | 67.26 | 80.87 | −0.701 |
| Regresión múltiple | 63.88 | 69.72 | −0.264 |
| Persistencia: repetir el mes anterior | 74.00 | 88.68 | −1.046 |

La múltiple presenta el menor error entre estas alternativas en esta partición. Sin embargo, **ambas regresiones tienen R² negativo**, por lo que el resultado no respalda una capacidad predictiva sólida en el bloque evaluado. R² negativo indica mayor error cuadrático que una predicción constante igual a la media del objetivo en prueba; esa media es una referencia matemática retrospectiva, no un pronóstico disponible de antemano.

La muestra es pequeña. La tendencia, estacionalidad, autocorrelación y posibles cambios en la operación limitan las conclusiones. Los coeficientes expresan asociaciones condicionales, no causalidad. No se calculan intervalos de predicción ni se asegura significancia estadística. Los resultados deben validarse con más meses antes de utilizarse para planeación.

## Reproducir el análisis

Abre `Etapa-II-Modelado-Predictivo.ipynb` y ejecuta las celdas en orden con el mismo entorno. Para regenerar las tablas numéricas iniciales:

```bash
python modelado.py
python -m unittest discover -s tests -v
```

`resultados/` contiene métricas, predicciones, correlaciones de entrenamiento, coeficientes y configuración de la ejecución inicial. Son resultados reproducibles, no datos adicionales inventados. Los cambios interactivos del dashboard se descargan desde su pestaña Comparación y no sobrescriben estos archivos.

`Despliegue-Analisis-Univariado.ipynb` conserva la celda que genera la aplicación. Depende de los módulos, tema y logo de esta misma carpeta; no es un instalador autónomo.

## Respaldo y entrega

1. Usa un repositorio de GitHub y sube los archivos del proyecto respetando `.gitignore`. Excluye `Base_Datos_GAC`, `respaldos`, entornos virtuales y credenciales.
2. Comprueba que el docente pueda acceder al repositorio. Si es privado, concede el acceso requerido.
3. Copia el enlace real del repositorio en la actividad de Canvas correspondiente a Evaluación y confirma allí la entrega.

## Referencias técnicas

- [Mínimos cuadrados y LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [Validación de series temporales](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split)
- [Definición e interpretación de R²](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html)
