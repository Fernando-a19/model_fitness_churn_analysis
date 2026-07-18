# ¿Qué clientes abandonan el gimnasio? Análisis de churn y segmentación en Model Fitness

## 📋 Contexto del proyecto

En este proyecto trabajé con los datos de una cadena de gimnasios llamada **Model Fitness**, que busca reducir la tasa de cancelación de membresías. El objetivo fue identificar patrones de comportamiento que permitan predecir qué clientes están en riesgo de abandonar, segmentar a los usuarios en grupos con características similares y proponer estrategias de retención basadas en datos.

Para lograr esto, se utilizaron técnicas de **machine learning supervisado** (regresión logística y random forest) para predecir el churn, y **clustering** (K-Means) para segmentar a los clientes en perfiles diferenciados.

---

## 🎯 Objetivos del análisis

### Predicción de abandono

* Construir modelos capaces de predecir si un cliente cancelará su membresía en el próximo mes.
* Comparar el rendimiento de la regresión logística y el random forest.
* Identificar las variables más importantes en la predicción del churn.

### Segmentación de clientes

* Agrupar a los clientes en segmentos con comportamientos similares mediante clustering.
* Caracterizar cada segmento y determinar cuál presenta mayor riesgo de abandono.
* Proporcionar recomendaciones diferenciadas por perfil.

---

## 🔧 Proceso de trabajo

### Preparación de los datos

Antes del análisis realicé un proceso de limpieza para asegurar que los resultados fueran confiables.

Las principales tareas fueron:

* Verificación de valores nulos y duplicados.
* Análisis de la distribución de variables numéricas y categóricas.
* Estandarización de variables para los modelos de clustering.

---

## 📊 Análisis exploratorio (EDA)

### 1. Distribución de variables por grupo de churn

Primero analicé cómo se distribuyen las principales características entre los clientes que abandonaron y los que permanecieron.

### Conclusión

Los clientes que cancelaron su membresía presentan patrones claramente diferenciados: menor antigüedad, menor frecuencia de visitas, contratos más cortos y menor participación en actividades grupales. Estas diferencias permiten anticipar el comportamiento futuro de cada usuario.

---

### 2. Matriz de correlaciones

Analicé la relación entre las variables para identificar cuáles tienen mayor impacto en la predicción del churn.

### Conclusión

Las variables con mayor correlación con el abandono fueron la **duración del contrato**, la **frecuencia de visitas en el último mes** y la **antigüedad del cliente**. Variables como la participación en clases grupales y el gasto adicional también mostraron correlación significativa con la retención.

---

## 🤖 Modelos predictivos

### 3. Regresión logística y Random Forest

Se entrenaron y compararon dos modelos de clasificación para predecir el churn.

**Resultados obtenidos:**

* **Regresión Logística:** Accuracy ≈ 91%, AUC-ROC elevado, buen equilibrio entre precisión y recall.
* **Random Forest:** Accuracy ≈ 91-92%, rendimiento similar con ligera ventaja en algunas métricas.

Ambos modelos ofrecen un rendimiento muy similar, lo que confirma que las variables seleccionadas tienen alto poder predictivo.

---

## 👥 Segmentación con K-Means

### 4. Dendrograma y clústeres

Se utilizó un dendrograma para determinar el número óptimo de grupos, y posteriormente se aplicó K-Means con **5 clústeres**.

Los clústeres se caracterizaron por:

* **Clúster 0:** Clientes con contratos cortos, baja frecuencia, alto riesgo de churn.
* **Clúster 1:** Clientes con alta antigüedad, contratos largos, muy leales.
* **Clúster 2:** Perfil intermedio, moderada frecuencia de visitas.
* **Clúster 3:** Clientes recientes con buena frecuencia, potencial de retención.
* **Clúster 4:** Alta participación en actividades, bajo riesgo de abandono.

---

## 📊 Resultados

### Principales métricas

* Accuracy de los modelos: **~91%**.
* El clúster con mayor tasa de churn presentó contratos promedio de 1 mes y frecuencia de visitas inferior a 1 por semana.
* Los clientes más leales tienen contratos de 12 meses y visitan el gimnasio al menos 3 veces por semana.

### Factores clave de retención

Los factores más relevantes para la retención fueron:

1. **Duración del contrato:** contratos más largos se asocian fuertemente con mayor retención.
2. **Frecuencia de visitas:** clientes más activos tienen menor probabilidad de abandonar.
3. **Participación en clases grupales:** genera vínculos con la comunidad del gimnasio.
4. **Antigüedad:** los clientes con más tiempo son significativamente más leales.

---

## 💡 Recomendaciones

### Estrategias de retención por segmento

**Para clientes de alto riesgo (Clúster 0):**

Implementar campañas de reactivación en las primeras semanas. Ofrecer incentivos para migrar a contratos de mayor duración. Asignar un seguimiento personalizado.

**Para clientes con potencial (Clústeres 2 y 3):**

Promover la participación en actividades grupales para fortalecer el vínculo con el gimnasio. Ofrecer descuentos por renovación anticipada.

**Para clientes leales (Clústeres 1 y 4):**

Implementar programas de fidelización y beneficios exclusivos. Convertirlos en embajadores del gimnasio mediante referidos.

### Acciones generales

* Incentivar contratos anuales desde el momento de la inscripción.
* Diseñar un programa de bienvenida para los primeros 30 días, que es el periodo crítico de abandono.
* Monitorear semanalmente la frecuencia de visitas para detectar señales tempranas de desenganche.

---

## 📁 Código y Documentación

- [Notebook completo](./notebooks/)
- [Código auxiliar](./src/)

