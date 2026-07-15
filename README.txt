
🏋️  Predicción de cancelación y segmentación de clientes en Model Fitness

📋 Contexto del proyecto

- En este proyecto trabajé con la base de datos de Model Fitness, una cadena de gimnasios que busca reducir la cancelación de clientes (churn). El objetivo fue identificar qué características diferencian a los clientes que abandonan el gimnasio de aquellos que permanecen activos, además de construir modelos capaces de predecir la cancelación antes de que ocurra.

- Para complementar el análisis también segmenté a los clientes mediante técnicas de clustering, con el fin de identificar perfiles de riesgo y proponer estrategias de retención basadas en datos.

🎯 Objetivos del análisis

- Predicción de cancelación
- Identificar las variables asociadas con la cancelación de clientes.
- Construir modelos de clasificación para predecir el churn.
- Comparar el desempeño de distintos modelos mediante métricas de evaluación.
- Segmentación de clientes
- Agrupar clientes con características similares mediante técnicas de clustering.
- Identificar los grupos con mayor riesgo de cancelación.
- Generar recomendaciones de negocio enfocadas en mejorar la retención.

🔧 Proceso de trabajo
## Preparación de los datos

Antes del análisis realicé una revisión completa de la calidad de los datos para asegurar resultados confiables.

Las principales tareas fueron:

Estandarización de nombres de columnas.
Verificación de tipos de datos.
Validación de valores nulos.
Redondeo de variables numéricas para mejorar la legibilidad del análisis.
Conclusión

La base de datos no presentó valores faltantes y los tipos de datos eran consistentes. También se estandarizaron los nombres de las columnas para facilitar el análisis. Algunas variables binarias no contaban con una descripción específica, por lo que fueron tratadas únicamente como variables codificadas, evitando realizar interpretaciones que no estuvieran respaldadas por la información disponible.

📊 Análisis exploratorio (EDA)
1. Análisis de las características de los clientes

Se exploró la distribución de las principales variables para identificar diferencias entre los clientes que permanecen activos y aquellos que cancelan su membresía.

(Insertar histogramas y distribuciones)

Conclusión

El análisis exploratorio permitió identificar diferencias importantes entre ambos grupos. Los clientes que cancelan suelen presentar una menor frecuencia de visitas, contratos de corta duración y menor antigüedad dentro del gimnasio. En contraste, los clientes con mayor permanencia muestran una asistencia más constante y un mayor compromiso con los servicios ofrecidos.

2. Relación entre variables

Posteriormente analicé la relación entre las variables mediante una matriz de correlación para identificar los factores más relacionados con la cancelación.

(Insertar matriz de correlación)

Conclusión

La frecuencia de visitas mostró la relación más fuerte con la permanencia de los clientes. También se observó que los contratos de mayor duración y el consumo de servicios adicionales están asociados con una menor probabilidad de cancelación.

🤖 Modelos predictivos
3. Predicción de cancelación

Se construyeron dos modelos de clasificación para estimar la probabilidad de que un cliente cancelara su membresía:

Regresión Logística.
Random Forest.

Los modelos fueron entrenados utilizando un conjunto de entrenamiento y posteriormente evaluados mediante un conjunto de prueba utilizando métricas como Accuracy, Precision y Recall.

(Insertar tabla de métricas)

Conclusión

Ambos modelos lograron identificar clientes con riesgo de cancelación. La comparación de métricas permitió evaluar cuál ofrece un mejor equilibrio entre precisión y capacidad de detección, proporcionando una herramienta útil para implementar estrategias preventivas de retención.

📈 Segmentación de clientes
4. Clustering

Además de la predicción, realicé una segmentación utilizando K-Means para identificar grupos de clientes con características similares.

Antes del agrupamiento, los datos fueron estandarizados y se utilizó un dendrograma para analizar la estructura de los grupos.

(Insertar dendrograma y gráfico de clusters)

Conclusión

La segmentación permitió identificar perfiles claramente diferenciados. Algunos grupos presentan una tasa de cancelación considerablemente mayor, mientras que otros muestran un alto nivel de fidelización.

Estos resultados permiten desarrollar estrategias específicas para cada tipo de cliente en lugar de aplicar acciones generales para toda la base de usuarios.

📊 Resultados
Principales hallazgos
La frecuencia de visitas es el principal indicador asociado con la cancelación.
Los clientes con contratos mensuales presentan la mayor tasa de churn.
Los clientes con menor antigüedad son los más propensos a abandonar el gimnasio.
El uso de servicios adicionales está relacionado con una mayor permanencia.
La segmentación permitió identificar grupos con distintos niveles de riesgo.
Modelado predictivo

Los modelos de clasificación demostraron que es posible anticipar qué clientes tienen mayor probabilidad de cancelar utilizando únicamente la información disponible en la base de datos.

Esta información puede utilizarse para implementar campañas preventivas antes de que ocurra la cancelación.

💡 Recomendaciones
Retención de nuevos clientes

Los clientes con menor antigüedad representan el grupo más vulnerable. Se recomienda implementar programas de acompañamiento durante las primeras semanas para fomentar el hábito de asistencia.

Incrementar la frecuencia de visitas

La asistencia es la variable más relacionada con la permanencia. Acciones como retos, clases grupales, recordatorios automáticos o promociones pueden ayudar a mantener una rutina constante.

Promover contratos de mayor duración

Los contratos largos muestran menores tasas de cancelación. Incentivos para renovar o migrar desde planes mensuales pueden mejorar significativamente la retención.

Impulsar servicios adicionales

Los clientes que utilizan cafetería, masajes u otros servicios presentan un mayor nivel de compromiso con el gimnasio. Promover estos servicios puede fortalecer la fidelización y aumentar el valor del cliente.

📁 Código y documentación
Notebook completo del análisis.
Código utilizado para la limpieza de datos, análisis exploratorio, modelos predictivos y segmentación de clientes.
