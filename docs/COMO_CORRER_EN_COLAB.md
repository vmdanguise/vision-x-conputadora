# Cómo correr el experimento en Colab

Guía para ejecutar `VisionXComputadora.ipynb` con las correcciones aplicadas y el
experimento de capas congeladas (Etapa 4B).

## 0. Antes de empezar

1. Subir `VisionXComputadora.ipynb` a Colab (o abrirlo desde Drive).
2. **Activar la GPU:** `Entorno de ejecución > Cambiar tipo de entorno de ejecución > T4 GPU`.
   La Etapa 1 corta con `SystemExit` si no hay GPU, así que si te olvidás te vas a enterar enseguida.
3. Verificar que la Etapa 1 imprima `CUDA disponible: True` y el nombre de la GPU.

## 1. Orden de ejecución

Celdas de arriba hacia abajo. Tiempos estimados en T4:

| Celda | Etapa | Tiempo | Notas |
|---|---|---|---|
| 1 | 1 · Entorno + dataset | ~5 min | Instala paquetes y baja el dataset de Kaggle |
| 2 | 2 · EDA | ~3 min | **PAUSA OBLIGATORIA**: ver punto 2 |
| 3 | 3 · Split + YOLO | ~3 min | Debe imprimir 70 / 15 / 15 |
| 4 | 4 · Entrenamiento base | 20–30 min | Genera `best.pt` |
| 5 | **4B · Experimento freeze** | **60–90 min** | 3 entrenamientos |
| 6 | 5 · Evaluación | ~10 min | Barrido de umbrales |
| 7–8 | 6–7 · Gráficos y boxes | ~5 min | |
| 9 | 8 · Galería 10k | ~3 min | |
| 10 | 9 · ONNX | ~3 min | |
| 11–12 | 10–11 · Informe + ZIP | ~2 min | **Descarga el ZIP** |
| 13 | 12 · Demo | opcional | Subís una imagen a mano |

**Total: ~2 a 2,5 horas.**

## 2. Pausa obligatoria en la Etapa 2

La celda imprime una grilla de ejemplos. **Mirarla** y confirmar que
`CLASE_PILETA_XML = "2"` corresponde efectivamente a las piletas.

Si las cajas marcadas rodean autos en vez de piletas, hay que invertir el mapeo antes
de seguir. Esto ya pasó una vez: con el mapeo invertido el pipeline entrena sin un solo
error sobre **cero** instancias de la clase objetivo y da mAP = 0.

## 3. Qué verificar en cada etapa corregida

**Etapa 3 — split.** Tiene que imprimir algo como:

```
Train: 2623 (70.0 %) | Val: 562 (15.0 %) | Test: 563 (15.0 %)
```

Si los porcentajes no dan, los `assert` cortan la ejecución. Antes esto daba 35/35/30
en silencio.

**Etapa 4B — freeze.** Para cada configuración imprime los parámetros entrenables.
Los valores esperados (ya verificados localmente) son:

| `freeze` | Entrenables | % | Tensores congelados |
|---|---|---|---|
| 0 | 2.636 M | 100 % | 1 |
| 10 | 1.589 M | 60.3 % | 100 |
| 24 | 0.878 M | 33.3 % | 172 |

Si los tres dan el mismo porcentaje, el `freeze` no se está aplicando: **avisame antes
de seguir**, porque la comparación no serviría.

**Etapa 5 — umbral.** Si imprime `AVISO: el óptimo cayó en el borde inferior`, hay que
ampliar el barrido por debajo de 0.05. Sin el aviso, el umbral es confiable.

**Etapa 9 — ONNX.** El delta de mAP tiene que ser **≈ 0** (±0.02). Es la misma red en
otro formato. Un delta grande indica un error de medición, no una mejora real.

## 4. Si Colab se desconecta

La Etapa 4B guarda `freeze_comparativa.csv` **después de cada corrida** y copia los
`results.csv` a `resultados/freeze_runs/`. Si se corta en la tercera, las dos primeras
ya están en disco.

Para retomar sin repetir todo:

1. Reejecutar las celdas 1 a 3 (son rápidas, ~10 min).
2. Si `best.pt` se perdió, hay que repetir la Etapa 4.
3. En la Etapa 4B, editar `CONFIGS` y dejar solo las configuraciones que falten.

**Importante:** `/content` es efímero. Si la sesión muere sin descargar el ZIP, se pierde
todo. Si querés evitarlo, montá Drive al principio:

```python
from google.colab import drive
drive.mount("/content/drive")
```

y cambiar `BASE = Path("/content")` por una ruta dentro de `/content/drive/MyDrive/`.

## 5. Qué necesito de vuelta

Para actualizar el informe con los resultados reales, pasame el **ZIP completo** de la
Etapa 11. Lo mínimo indispensable:

- `resultados/freeze_comparativa.csv` — la tabla del experimento
- `resultados/freeze_runs/results_freeze_{0,10,24}.csv` — curvas por época
- `resultados/metricas.csv` — métricas de la corrida con el split corregido
- `resultados/trade_off_onnx.csv` — comparación ONNX ya válida
- `resultados/freeze_comparativa.png` y el resto de los gráficos

Con eso reemplazo §5.6 del informe (hoy dice que el experimento no se ejecutó),
actualizo las métricas de §7 y §10, y regenero el PDF.

## 6. Si algo falla

Copiame el mensaje de error **completo** con la celda donde ocurrió. Los errores más
probables:

| Síntoma | Causa | Solución |
|---|---|---|
| `SystemExit: Sin GPU` | Entorno sin GPU | Activar T4 y reejecutar desde la celda 1 |
| `CUDA out of memory` en la 4B | Otra corrida no liberó memoria | Reiniciar el entorno y reejecutar desde la 1 |
| `RuntimeError: freeze... 0 trainable` | `freeze` ≥ 25 | No debería pasar: `CONFIGS` usa 0/10/24 |
| mAP = 0 en todo | Mapeo de clases invertido | Volver a la Etapa 2 (punto 2 de esta guía) |
| `assert` del split | Proporciones mal | Avisame: es un bug, no un problema del entorno |
