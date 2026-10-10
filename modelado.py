"""Funciones reproducibles para la Etapa II. No depende de Streamlit."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CHANNELS = ["Piso", "Plazas", "Eventos", "Planta", "Digitales", "Otros"]
DEFAULT_TARGET = "Digitales"
DEFAULT_SIMPLE = "Digitales (t−1)"
DEFAULT_MULTIPLE = ["Digitales (t−1)", "Plazas (t−1)", "Piso (t−1)"]


def load_monthly(path=None):
    path = path or Path(__file__).parent / "data" / "leads_mensuales.csv"
    frame = pd.read_csv(path)
    required = ["Periodo", "Mes", *CHANNELS]
    if not set(required).issubset(frame.columns):
        raise ValueError("La tabla mensual no contiene todas las columnas requeridas.")
    frame = frame[required].copy()
    for column in ["Periodo", *CHANNELS]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if frame.isna().any().any() or not np.isfinite(frame[["Periodo", *CHANNELS]]).all().all():
        raise ValueError("La tabla mensual contiene datos faltantes o no finitos.")
    if (frame[CHANNELS] < 0).any().any() or (frame[CHANNELS] % 1 != 0).any().any():
        raise ValueError("Los conteos deben ser enteros no negativos.")
    if frame["Periodo"].duplicated().any() or (frame["Periodo"] % 1 != 0).any():
        raise ValueError("Los periodos deben ser enteros únicos.")
    frame = frame.sort_values("Periodo").reset_index(drop=True)
    if len(frame) < 16 or not frame["Periodo"].diff().dropna().eq(1).all():
        raise ValueError("Se requieren al menos 16 periodos consecutivos.")
    return frame


def supervised(frame, target):
    if target not in CHANNELS:
        raise ValueError("Selecciona un canal válido como variable objetivo.")
    result = frame[["Periodo", "Mes", target]].copy()
    for channel in CHANNELS:
        result[f"{channel} (t−1)"] = frame[channel].shift(1)
    # Sólo falta el primer mes: no se imputan datos ni se sustituyen faltantes por cero.
    return result.iloc[1:].reset_index(drop=True)


def chronological_split(frame, test_fraction=.2):
    if not .15 <= test_fraction <= .4:
        raise ValueError("La fracción de prueba debe estar entre 0.15 y 0.40.")
    count = int(np.ceil(len(frame) * test_fraction))
    return frame.iloc[:-count].copy(), frame.iloc[-count:].copy()


def scores(actual, predicted):
    # R² no está definido cuando la variable real es constante.
    return {"MAE": float(mean_absolute_error(actual, predicted)),
            "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
            "R²": float(r2_score(actual, predicted)) if pd.Series(actual).nunique() > 1 else np.nan}


def fit_regression(train, test, target, features):
    allowed = [f"{channel} (t−1)" for channel in CHANNELS]
    if not features or len(set(features)) != len(features) or any(f not in allowed for f in features):
        raise ValueError("Selecciona predictores distintos del mes anterior.")
    if len(train) < max(10, len(features) + 3) or len(test) < 2:
        raise ValueError("No hay suficientes observaciones para entrenar y evaluar.")
    if train["Periodo"].max() >= test["Periodo"].min():
        raise ValueError("El entrenamiento debe ocurrir antes que la prueba.")
    if train[target].nunique() < 2:
        raise ValueError("La variable objetivo no varía en entrenamiento.")
    constant = [f for f in features if train[f].nunique() < 2]
    if constant:
        raise ValueError("Predictores constantes: " + ", ".join(constant))
    X = train[features]
    model = LinearRegression().fit(X, train[target])
    fitted, predicted = model.predict(X), model.predict(test[features])
    coefficients = pd.DataFrame({"Variable": ["Intercepto", *features],
                                 "Coeficiente": [model.intercept_, *model.coef_]})
    standardized = X.sub(X.mean()).div(X.std(ddof=0)).to_numpy()
    return {"model": model, "features": features, "train_prediction": fitted,
            "test_prediction": predicted, "train_scores": scores(train[target], fitted),
            "test_scores": scores(test[target], predicted), "coefficients": coefficients,
            "rank": int(np.linalg.matrix_rank(np.column_stack([np.ones(len(X)), X]))),
            "condition": float(np.linalg.cond(standardized))}


def run_analysis(frame, target=DEFAULT_TARGET, simple=DEFAULT_SIMPLE,
                 multiple=None, test_fraction=.2):
    multiple = DEFAULT_MULTIPLE if multiple is None else multiple
    if len(multiple) < 2:
        raise ValueError("La regresión múltiple requiere al menos dos predictores.")
    table = supervised(frame, target)
    train, test = chronological_split(table, test_fraction)
    results = {"Simple": fit_regression(train, test, target, [simple]),
               "Múltiple": fit_regression(train, test, target, multiple)}
    # Predicción a un paso: el mes anterior real ya está observado al predecir t.
    persistence = test[f"{target} (t−1)"].to_numpy()
    comparison = pd.DataFrame([
        {"Modelo": name, "Predictores": len(result["features"]),
         **result["test_scores"], "R² entrenamiento": result["train_scores"]["R²"]}
        for name, result in results.items()
    ] + [{"Modelo": "Persistencia", "Predictores": 1, **scores(test[target], persistence),
          "R² entrenamiento": np.nan}])
    predictions = test[["Periodo", "Mes"]].copy()
    predictions["Real"] = test[target].to_numpy()
    for name, result in results.items():
        predictions[name] = result["test_prediction"]
        predictions[f"Residuo {name}"] = predictions["Real"] - predictions[name]
    predictions["Persistencia"] = persistence
    return table, train, test, results, comparison, predictions


def export_default(output_dir):
    """Resultados de la configuración inicial, regenerables desde la terminal."""
    import json
    frame = load_monthly()
    _, train, test, results, comparison, predictions = run_analysis(frame)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_dir / "metricas_prueba.csv", index=False)
    predictions.to_csv(output_dir / "predicciones_prueba.csv", index=False)
    frame.loc[frame.Periodo <= train.Periodo.max(), CHANNELS].corr().to_csv(output_dir / "correlaciones_entrenamiento.csv")
    for name, result in results.items():
        result["coefficients"].to_csv(output_dir / f"coeficientes_{'simple' if name == 'Simple' else 'multiple'}.csv", index=False)
    settings = {"objetivo": DEFAULT_TARGET, "simple": DEFAULT_SIMPLE, "multiple": DEFAULT_MULTIPLE,
                "fraccion_prueba": .2, "n_entrenamiento": len(train), "n_prueba": len(test),
                "periodos_entrenamiento": train.Periodo.tolist(), "periodos_prueba": test.Periodo.tolist(),
                "evaluacion": "Pronósticos a un paso con valores reales de t−1; coeficientes congelados en entrenamiento."}
    (output_dir / "configuracion.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n")
    return comparison


if __name__ == "__main__":
    print(export_default(Path(__file__).parent / "resultados").to_string(index=False))
