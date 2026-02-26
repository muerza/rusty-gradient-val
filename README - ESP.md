# Rusty Bargain — Predicción del Precio de Autos Usados

El servicio de venta de autos usados **Rusty Bargain** necesita una aplicación que permita a los clientes estimar rápidamente el valor de mercado de su coche. Este proyecto entrena y compara múltiples modelos de *Machine Learning* para predecir precios a partir de especificaciones técnicas, versiones de equipamiento e historial de precios.

## Objetivo

- Predecir el precio de mercado de un auto usado.
- Comparar modelos usando tres criterios clave:
  - **Calidad de predicción** (RMSE)
  - **Tiempo de entrenamiento**
  - **Tiempo de predicción**
- Implementar un modelo de **Regresión Lineal** con **Descenso por Gradiente manual**.

## Dataset

El archivo `car_data.csv` contiene datos históricos de autos usados con características como:

- Tipo de vehículo, marca y modelo
- Año de registro y kilometraje
- Tipo de combustible, transmisión y potencia
- Precio de venta

## Modelos Comparados

| Modelo | Descripción |
|--------|-------------|
| **Regresión Lineal** | Modelo base (prueba de cordura) |
| **Árbol de Decisión** | `DecisionTreeRegressor` |
| **Bosque Aleatorio** | `RandomForestRegressor` |
| **LightGBM** | Gradient Boosting — maneja variables categóricas de forma nativa |
| **XGBoost** | Gradient Boosting — requiere One-Hot Encoding |
| **Regresión Lineal Manual** | Implementación propia con Descenso por Gradiente |

## Tecnologías

- **Python 3**
- **pandas**, **numpy** — manipulación de datos
- **matplotlib** — visualización
- **scikit-learn** — modelos base, preprocesamiento, métricas
- **LightGBM** — gradient boosting
- **XGBoost** — gradient boosting
- **joblib** — persistencia de modelos

## Estructura del Proyecto

```text
rusty-bargain-gradient-autos/
├── rusty-gradient-val.ipynb   # Notebook principal
├── car_data.csv               # Dataset
├── regressor.json             # Configuración del modelo
└── README.md
```

## Resultados

| Modelo               | R2     | MSE       | RMSE     | Time_s | Pred_ms |
|----------------------|--------|-----------|----------|--------|---------|
| LGB                  | 0.9027 | 2,161,786 | 1,470.30 | 18.566 | 1354.08 |
| LGBMR                | 0.9027 | 2,162,785 | 1,470.64 | 13.322 | 870.78  |
| RandomForest         | 0.8960 | 2,312,579 | 1,520.72 | 34.553 | 336.47  |
| XGBoost              | 0.8874 | 2,502,547 | 1,581.94 | 1.246  | 94.11   |
| DecisionTree         | 0.8295 | 3,788,603 | 1,946.43 | 36.059 | 45.07   |
| LinearRegression     | 0.7169 | 6,291,183 | 2,508.22 | 2.266  | 97.76   |
| LinearRegression_GD  | 0.6815 | 7,079,618 | 2,660.76 | 6.084  | 31.70   |

Cada modelo fue evaluado usando **RMSE** sobre el conjunto de prueba, midiendo además el tiempo de entrenamiento y el tiempo de predicción. Los modelos de *gradient boosting* (LightGBM, XGBoost) ofrecen la mejor calidad de predicción, mientras que la regresión lineal funciona como referencia base. Los resultados detallados y las gráficas comparativas se encuentran dentro del notebook.
