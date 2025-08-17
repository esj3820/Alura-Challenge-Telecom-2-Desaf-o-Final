# Alura-Challenge-Telecom-2-Desaf-o-Final

Telecom X
🎯 Misión del Proyecto
El objetivo principal de este proyecto es desarrollar un modelo predictivo robusto para identificar a los clientes con alta probabilidad de cancelar sus servicios (Churn), permitiendo a la empresa anticiparse y tomar acciones proactivas. Este es un problema de clasificación binaria, donde la variable objetivo es Churn (1 si el cliente se va, 0 si no).
La métrica clave priorizada para este proyecto es el Recall (sensibilidad) para la clase positiva (Churn = 1), con el fin de minimizar los falsos negativos y asegurar la identificación de la mayor cantidad posible de clientes en riesgo de abandono.
🧠 Tareas Realizadas
El proyecto siguió una metodología estructurada que incluyó las siguientes tareas:
• Preparación de Datos: Tratamiento, codificación y normalización de la información.
• Análisis de Correlación y Selección de Variables: Identificación de las características más relevantes.
• Entrenamiento de Modelos de Clasificación: Se probaron tres tipos de modelos: Decision Tree, Random Forest y CatBoost.
• Evaluación del Rendimiento: Utilizando métricas clave como Accuracy, Precision, Recall y F1-score.
• Interpretación de Resultados: Incluyendo la importancia de las variables para el modelo.
• Conclusión Estratégica: Señalando los principales factores de influencia en la cancelación de servicios.
🛠️ Bibliotecas Utilizadas
El desarrollo del proyecto se realizó utilizando las siguientes bibliotecas de Python:
• pandas
• matplotlib.pyplot
• numpy
• seaborn
• joblib, pickle
• sklearn.base, sklearn.compose, sklearn.ensemble, sklearn.linear_model, sklearn.metrics, sklearn.model_selection, sklearn.pipeline, sklearn.preprocessing, sklearn.tree
• imblearn.over_sampling (SMOTE, ADASYN, RandomOverSampler)
• imblearn.under_sampling (TomekLinks)
• catboost
• scipy.stats
• statsmodels.stats.outliers_influence
📊 Análisis Exploratorio de Datos (EDA) y Preparación
• Se utilizó un dataset normalizado de un desafío anterior, con registros sin valores nulos.
• La columna customerID fue eliminada al considerarse irrelevante para el modelado.
• Se aplicó codificación (encoding) a las variables:
    ◦ Columnas binarias ('Yes'/'No') se transformaron a 1 y 0.
    ◦ Variables categóricas como Genero_Cliente, Servicio_Internet, Tipo_Contrato, Metodo_Pago se codificaron usando OneHotEncoder.
• Se identificó una fuerte correlación entre Charges.Monthly y Charges.Total, lo que llevó a la eliminación de Charges.Monthly para simplificar el conjunto de datos.
• El análisis de correlación se centró en variables con una correlación absoluta superior al 15% con Churn.
• El análisis de multicolinealidad fue omitido debido a que los modelos seleccionados no son sensibles a este fenómeno.
Hallazgos Clave del EDA:
• Desequilibrio de Clases: Se observó una proporción desigual entre las clases 'Churn No' y 'Churn Sí'.
• Antigüedad (Tenure) y Churn: La mayor deserción ocurre en los primeros 12 meses de permanencia, siendo los meses 0 a 6 los de mayor abandono.
• Categorías con Mayor Churn: Se identificaron las 10 categorías de características con el porcentaje más alto de abandono de clientes, como se muestra en los gráficos generados.
📦 Modelos Probados y Resultados Clave
Se probaron y optimizaron tres modelos de clasificación, enfocándose en maximizar el Recall para la clase 1 (Churn Sí).
1. DecisionTreeClassifier
• Sin optimizaciones: El modelo inicial mostró un Recall de solo el 48% para la clase Churn Sí, considerándose "prácticamente inútil" para el objetivo.
• Con Validación Cruzada: Las métricas se mantuvieron similares (Recall 49%), indicando estabilidad y buena generalización, pero sin mejora en el recall para la clase de interés.
• Con Balanceo de Clases: Se probaron RandomOverSampler, SMOTE, ADASYN y TomekLinks. RandomOverSampler demostró ser la técnica con el mejor Recall (promedio de 0.8122) para la clase Churn Sí, a pesar de una menor precisión, y también el mejor F1-score global.
• Con Optimización de Parámetros (GridSearchCV): Tras la optimización, se lograron mejoras en las métricas de clasificación y la matriz de confusión. El pipeline optimizado fue guardado.
2. CatBoost
• Sin optimizaciones y con Validación Cruzada: Al igual que Decision Tree, CatBoost mostró un bajo Recall para la clase Churn Sí (alrededor del 48-49%) en sus versiones iniciales.
• Con Balanceo de Clases: El balanceo utilizando una relación de pesos de clase calculada (n_class0 / n_class1) ofreció resultados "mucho mejores" en comparación con el balanceo automático (auto_class_weights='Balanced').
• Con Optimización de Parámetros (RandomizedSearchCV): Se optimizaron los hiperparámetros del modelo, priorizando el recall. El modelo optimizado y su pipeline también fueron guardados.
3. RandomForestClassifier
• Sin optimizaciones: Similar a los otros modelos, el RandomForest inicial tuvo un Recall de solo el 48% para la clase Churn Sí, siendo poco útil para el objetivo.
• Con Validación Cruzada: Las métricas se mantuvieron estables (Recall 48%), confirmando la consistencia del modelo.
• Con Balanceo de Clases: Se probaron diversas técnicas de muestreo, incluyendo RandomOverSampler y class_weight='balanced'. RandomOverSampler resultó ser nuevamente la técnica con el mejor Recall para esta clase.
• Con Optimización de Parámetros (RandomizedSearchCV): El RandomForestClassifier optimizado logró un rendimiento significativamente mejor. Predijo correctamente al 90% de los clientes Churn Sí y al 61% de los Churn No. El recall medio para la clase 1 fue de 0.898 con un intervalo de confianza del 95% de (0.871, 0.923). Este modelo es considerado aceptable para el objetivo del proyecto.
• Importancia de Variables (Feature Importance): Se identificaron las características con una importancia superior al 2% para el modelo RandomForest optimizado.
📈 Resumen y Comparación de Modelos Optimizados
Se generó una tabla comparativa y un gráfico de barras para visualizar las métricas clave (Accuracy, Precision_Clase_1, Recall_Clase_1, F1-Score_Clase_1) de las versiones optimizadas de DecisionTree, CatBoost y RandomForest.
El gráfico de Recall para la Clase 1 (Churn=Yes) es la métrica más relevante para este proyecto, mostrando claramente el rendimiento superior del RandomForestClassifier optimizado en la identificación de clientes en riesgo de abandono.
📁 Archivos Clave Generados
• df_codificado.csv: Dataset procesado y codificado.
• graf_01_torta_churn.png: Distribución de Churn.
• graf_02_desercion_primer_año.png: Clientes que abandonaron en el primer año.
• graf_03_top_10_churn.png: Top 10 categorías con mayor porcentaje de churn.
• mapa_de_calor_correlacion_churn.png: Mapa de calor de correlaciones con Churn.
• MC_*.png: Matrices de confusión para cada modelo y técnica de balanceo/optimización.
• ROC_*.png: Curvas ROC para cada modelo y pipeline optimizado.
• graf_DTC_recall_por_tecnica_balaceo.png: Recall y F1-Score de DecisionTree por técnica de balanceo.
• graf_RF_recall_por_tecnica_balaceo.png: Recall y F1-Score de RandomForest por técnica de balanceo.
• graf_RF_importancias.png: Importancia de las características para RandomForest.
• graf_recall_por_modelo.png: Comparación final del Recall de los modelos optimizados.
• pipeline_dt_optimizado.pkl: Pipeline optimizado de Decision Tree.
• pipeline_CatBoost_optimizado.pkl: Pipeline optimizado de CatBoost.
• pipeline_RandomForest_optimizado.pkl: Pipeline optimizado de RandomForest.

--------------------------------------------------------------------------------
