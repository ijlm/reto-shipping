# Ejercicio 1: Análisis de Ofertas Relámpago

## Descripción

Análisis exploratorio de datos sobre ofertas relámpago. El objetivo es identificar patrones de comportamiento, factores de éxito y oportunidades de optimización mediante técnicas de análisis estadístico y visualización.

## Dataset

**Ubicación:** `data/ofertas_relampago.csv`

Las ofertas relámpago son promociones de duración limitada con stock comprometido y descuentos sobre el precio original. El dataset contiene información sobre el desempeño de estas ofertas.

### Columnas Principales

- Identificadores de oferta y producto
- Fechas de inicio y fin
- Stock comprometido y vendido
- Precios de oferta
- Categoría del producto

## Análisis Propuesto

### 1. Análisis Descriptivo

- Dimensiones y estructura del dataset
- Tipos de datos y valores nulos
- Estadísticas descriptivas básicas

### 2. Análisis Temporal

- Distribución de ofertas por hora y día
- Duración promedio de ofertas
- Patrones estacionales
- Mejor momento para publicar ofertas

### 3. Análisis de Performance

- Tasa de conversión: conversiones / vistas
- Sell-through rate: stock vendido / stock comprometido
- Distribución de descuentos
- Relación entre descuento y conversión

### 4. Análisis por Categoría

- Performance por categoría de producto
- Descuentos promedio por categoría
- Engagement por segmento
- Identificación de categorías top-performing

### 5. Correlaciones

- Matriz de correlación entre variables numéricas
- Relación descuento-conversión
- Relación stock-ventas
- Relación duración-performance

## Métricas Clave

### Ventas

```
sell_out_rate = (SOLD_QUANTITY/INVOLVED_STOCK) * 100
```

## Distribuciones

- Histogramas de variables clave
- Distribución de categorías

### Relaciones

- Scatter plots: descuento vs conversión
- Scatter plots: precio vs ventas
- Heatmap de correlaciones

### Performance

- Top ofertas por métrica
- Comparación entre segmentos
- KPIs por categoría

### Temporales

- Series temporales de métricas
- Patrones por hora y día
- Evolución de KPIs

## Preguntas a Responder

1. ¿Qué factores determinan el éxito de una oferta?
2. ¿Existen patrones temporales relevantes?
3. ¿Cómo varían las métricas por categoría?
4. ¿Qué oportunidades de optimización existen?

## Insights Principales

### Por Vertical

**Accessories (ACC):**

- Eficiencia: 21% de conversión (mejor ratio)
- Revenue o ventas: $27K (bajo)
- Característica: Productos baratos, poco stock
- Conclusión: Alta conversión pero bajo impacto económico

**Consumer Electronics (CE):**

- Eficiencia: 10% de conversión (peor ratio)
- Revenue o ventas: $363K (segundo mejor)
- Característica: Productos caros, mucho stock
- Conclusión: Baja conversión pero alto valor por venta

**Beauty & Health:**

- Eficiencia: 20% de conversión (excelente)
- Revenue o ventas: $547K (el mejor)
- Característica: Balance precio-demanda óptimo
- Conclusión: Mejor vertical en términos de ventas y conversión

### Factores de Éxito

**Factor #1: Origen del Vendedor**

- Vendedores origen (A): 5x más impacto en ventas
- Señal de experiencia y seller mejor preparados ?
- Mayor compromiso con la oferta

**Factor #2: Stock Inicial**

- Stock >50 unidades: Indicador de compromiso
- Correlación positiva con ventas totales
- Evita agotamiento temprano, que se identifico en lagunas categorias más ventas que stock.

**Factor #3: Categoría**

- Tecnología (celulares, TVs, tablets): Alto valor
- Belleza & Salud: Balance perfecto
- Evitar: Accesorios de bajo valor, si lo que se quiere es maximizar ingreso por ventas.
- Maximizar: Accesorios de bajo valor, si lo que se quiere es atraer customers.

**Factor #4: Factor estacional / tiempo**

- Horarios óptimos: 5 PM y 10 PM (mayor conversión)
- Días óptimos: Lunes-Miércoles (alineado con fechas especiales)
- Evitar: Fines de semana (muy baja conversión)

## Recomendaciones

1. **Priorizar vendedores tipo origen (A)** - Mayor garantía de éxito
2. **Exigir stock mínimo >50 unidades** - Compromiso y disponibilidad
3. **Enfocar en verticales rentables:**
   - Tecnología (alto valor)
   - Belleza & Salud (balance perfecto)
4. **Optimizar timing:**
   - Horarios: 5 PM y 10 PM
   - Días: Lunes-Miércoles

## Entregables

- Notebook Jupyter con análisis completo
- Visualizaciones relevantes
- Insights documentados
- Conclusiones y recomendaciones

## Estructura de Análisis

1. Carga y exploración inicial de datos
2. Limpieza y preprocesamiento
3. Análisis univariado
4. Análisis bivariado y multivariado
5. Segmentación y clustering
6. Conclusiones y recomendaciones

## Consideraciones

- Validar hipótesis con pruebas estadísticas
- Documentar decisiones de preprocesamiento
- Interpretar resultados en contexto de negocio
- Identificar limitaciones del análisis
