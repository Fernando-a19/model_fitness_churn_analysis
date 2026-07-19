#                       Análisis de churn y segmentación en Model Fitness

## 📋 Contexto del proyecto

En este proyecto trabajé con los datos de una cadena de gimnasios llamada **Model Fitness**, cuyo objetivo es reducir la tasa de cancelación de membresías mediante el análisis del comportamiento de sus clientes.

El análisis busca identificar los factores asociados al abandono, construir modelos predictivos capaces de anticipar la cancelación de una membresía y segmentar a los clientes en grupos con características similares para proponer estrategias de retención basadas en datos.

Para ello se emplearon técnicas de **Machine Learning Supervisado** (Regresión Logística y Random Forest) para la predicción del churn y **Machine Learning No Supervisado** (K-Means) para la segmentación de clientes.

---

# 🎯 Objetivos del análisis

## Predicción de abandono

- Construir modelos capaces de predecir si un cliente cancelará su membresía.
- Comparar el desempeño de Regresión Logística y Random Forest.
- Identificar las variables con mayor influencia en la predicción.

## Segmentación de clientes

- Agrupar clientes con características similares.
- Identificar perfiles de riesgo.
- Generar recomendaciones específicas para cada segmento.

---

# 🔧 Preparación de los datos

## Objetivo

Garantizar que la información utilizada para el análisis fuera consistente y adecuada para el entrenamiento de los modelos.

### Actividades realizadas

- Verificación de valores nulos.
- Verificación de registros duplicados.
- Revisión de tipos de datos.
- Estandarización de variables para K-Means.

### Conclusión

El conjunto de datos presentó una buena calidad, por lo que únicamente fue necesario preparar las variables para el proceso de modelado y segmentación.

---

# 📊 Análisis Exploratorio de Datos (EDA)

## 1. Distribución de variables por grupo de churn

### Objetivo

Comparar las características de los clientes que permanecieron, con las de los clientes que cancelaron su membresía.

### Gráfica

![Distribución y cancelación por variable](results/Model%20fitness_files/Model%20fitness_11_0.png)

---

### Conclusión

Los clientes que cancelaron presentan menor antigüedad, menor frecuencia de visitas, contratos más cortos y menor participación en actividades grupales. Estas diferencias permiten detectar señales tempranas de abandono.

---

## 2. Matriz de correlaciones

### Objetivo

Analizar la relación entre las variables para identificar cuáles tienen mayor influencia sobre la cancelación.

### Gráfica

![Matriz de correlaciones](results/Model%20fitness_files/Model%20fitness_13_0.png)

---

### Conclusión

La duración del contrato, la frecuencia de visitas y la antigüedad son las variables más relacionadas con la retención de clientes. La participación en clases grupales y el gasto adicional también muestran una relación importante.

---

# 🤖 Modelado Predictivo

## 3. Comparación de modelos

### Objetivo

Comparar el desempeño de Regresión Logística y Random Forest para seleccionar el modelo más adecuado.

### Gráfica

| Modelo | Accuracy | Precision | Recall |
|---|---:|---:|---:|
| Regresión Logística | 0.9263 | 0.8639 | 0.8333 |
| Random Forest | 0.9213 | 0.8497 | 0.8283 |

---

### Conclusión

Ambos modelos obtuvieron un desempeño cercano al **91 % de accuracy**, demostrando una alta capacidad para identificar clientes con riesgo de abandono.

---

## 4. Interpretabilidad de la Regresión Logística

### Objetivo

Analizar los coeficientes del modelo para comprender qué variables incrementan o disminuyen la probabilidad de churn.

### Gráfica

![Coeficientes de Regresión Logística e importancias de Random Forest](results/Model%20fitness_files/Model%20fitness_25_0.png)

---

### Conclusión

Los coeficientes muestran que la duración del contrato, la frecuencia de visitas y la antigüedad son los factores con mayor influencia sobre la permanencia del cliente.

---

## 5. Importancia de variables (Random Forest)

### Objetivo

Identificar las variables más relevantes según Random Forest y comparar estos resultados con la Regresión Logística.

### Gráfica

![Coeficientes de Regresión Logística e importancias de Random Forest](results/Model%20fitness_files/Model%20fitness_25_0.png)

---

### Conclusión

Random Forest confirma que la frecuencia de visitas, la duración del contrato y la antigüedad son los principales predictores del abandono.

---

# 👥 Segmentación de Clientes

## 6. Dendrograma

### Objetivo

Explorar la estructura de los datos para estimar el número adecuado de grupos antes de aplicar K-Means.

### Gráfica

![Dendrograma de usuarios](results/Model%20fitness_files/Model%20fitness_19_0.png)

---

### Conclusión

El dendrograma muestra una separación natural entre los clientes y sirve como referencia para el proceso de segmentación.

---

## 7. Método del codo y coeficiente Silhouette

### Objetivo

Determinar el número óptimo de clústeres para la segmentación.

### Gráfica

![Método del codo y coeficiente silhouette](results/Model%20fitness_files/Model%20fitness_27_0.png)

---

### Conclusión

Ambos métodos respaldan el uso de **cinco clústeres**, proporcionando un equilibrio adecuado entre simplicidad y capacidad de diferenciación.

---

## 8. Caracterización de los clústeres

### Objetivo

Describir el comportamiento de cada grupo de clientes obtenido mediante K-Means.

### Gráfica

![Perfiles promedio por clúster](results/Model%20fitness_files/Model%20fitness_23_0.png)

---

### Conclusión

Los cinco clústeres presentan perfiles claramente diferenciados en términos de frecuencia de visitas, antigüedad, duración del contrato y riesgo de abandono.

---

## 9. Tasa de churn por clúster

### Objetivo

Identificar qué segmentos presentan la mayor probabilidad de cancelar su membresía.

### Gráfica

![Tasa de churn por clúster](results/Model%20fitness_files/Model%20fitness_29_1.png)

---

### Conclusión

Los clientes con contratos cortos y baja frecuencia de asistencia concentran la mayor tasa de cancelación, mientras que los clientes con contratos largos y alta participación presentan la menor.

---

# 📈 Resultados

## Principales métricas

- Accuracy de los modelos: **≈ 91 %**.
- Ambos modelos presentan un desempeño muy similar.
- Se identificaron cinco segmentos de clientes con distintos niveles de riesgo.
- La duración del contrato y la frecuencia de visitas son los principales factores asociados a la retención.

---

# 📌 Conclusión

El análisis muestra que la cancelación de clientes no ocurre al azar. Existen patrones claros que permiten identificar con anticipación a los usuarios con mayor riesgo de abandonar el gimnasio.

Los principales hallazgos indican que los clientes que asisten con menor frecuencia, tienen poca antigüedad, contratan planes de corta duración o hacen un uso limitado de los servicios adicionales presentan una probabilidad significativamente mayor de cancelar su membresía. En contraste, los clientes con hábitos de asistencia constantes y mayor involucramiento con los servicios del gimnasio muestran una mayor permanencia.

Asimismo, la segmentación de clientes permitió identificar grupos con comportamientos y niveles de riesgo diferentes, lo que evidencia que una estrategia de retención general no es la opción más eficiente.

En conjunto, estos resultados proporcionan una base objetiva para priorizar clientes en riesgo, intervenir antes de que abandonen el gimnasio y diseñar estrategias comerciales enfocadas en mejorar la permanencia y el valor de cada cliente.

---

# 💡 Recomendaciones

- Fortalecer el proceso de incorporación de nuevos clientes durante las primeras semanas.
- Monitorear continuamente la frecuencia de asistencia para detectar señales tempranas de abandono.
- Incentivar la contratación de planes de mayor duración.
- Promover el uso de servicios adicionales para incrementar el compromiso del cliente.
- Diseñar estrategias de retención específicas para cada segmento identificado.

---

# 📁 Código y documentación

- 📓 Notebook completo
- 📂 Código auxiliar
- 📊 Presentación ejecutiva de resultados