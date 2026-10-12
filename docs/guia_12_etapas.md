# 🛰️ Detección de Piletas en Imágenes Satelitales
## 12 prompts para Google Colab — modelo único YOLOv5n (1 clase: pileta)

> Cómo usar esto: copiá cada prompt (bloque gris) y pegalo en Colab con Gemini activado
> (o en la IA que uses para generar código). Ejecutá las etapas en orden 1 → 12.
> Runtime recomendado: **GPU T4 gratis**.
>
> Archivos del proyecto (los 3 consistentes entre sí):
> `prompts_colab_deteccion_piletas_autos.md` (esta guía) ·
> `VisionXComputadora.ipynb` (notebook, 1 celda por etapa, 12 etapas) ·
> `visionxcomputadora.py` (mismo código en script, 1 clase: pileta).

---

## 1. Análisis del dataset (real, verificado en ejecución)

**Dataset:** [Swimming Pool And Car Detection](https://www.kaggle.com/datasets/kbhartiya83/swimming-pool-and-car-detection) (Kaggle, kbhartiya83). Licencia GPL 2.

| Concepto | Detalle verificado |
|---|---|
| Tipo de datos | Imágenes satelitales/aéreas + anotaciones **XML Pascal VOC** (no CSV) |
| Clases crudas en los XML | Valores `"1"` (mayoritaria, ~13.000 boxes) y `"2"` (~3.150 boxes) |
| `training` | **3.748 imágenes** con su XML cada una (~16.000 anotaciones en total) |
| `test_data_images/` | ~10.200 imágenes **SIN etiquetas** (solo inferencia visual) |
| Tamaño total | ~118 MB (entra cómoda en el disco gratis de Colab) |

⚠️ **Lección aprendida (bug real que tuvimos):** los valores de clase del XML **hay que confirmarlos a ojo** con la grilla de la Etapa 2. Si se asume mal el mapeo, el entrenamiento queda con cero piletas (nos pasó: "Piletas: 0" en todos los splits). La Etapa 2 lo blinda con assert sobre `CLASE_PILETA_XML`; el resto de clases se descarta.

## 2. Análisis del problema

- **Tarea:** detección de objetos supervisada con 1 clase (pileta) y ~3.700 imágenes → se usa **transfer learning** desde pesos COCO (`yolov5nu.pt`).
- **Modelo único:** YOLOv5n, el más chico (~2.5 M params, ~5 MB) y rápido de entrenar de la familia (~10–15 min en T4 a 320 px). Con `nc=1` la cabeza Detect es mínima. Ideal para Colab gratis e inferencia en CPU.
- **Salida final:** bounding boxes de piletas + confianza.
- **Restricción dura:** Colab gratis (GPU T4 16 GB, sesión ~12 h, cortes aleatorios) → todo liviano, con `best.pt`, early stopping y código re-ejecutable por etapas.

## 3. El modelo: YOLOv5n

Detector de **1 etapa, anchor-free**: divide la imagen en grilla y predice, en una sola pasada, clase + box + confianza para cada celda, en 3 escalas (objetos chicos/medianos/grandes). La variante **nano** usa el backbone más angosto y profundo-compacto de la familia: ~2.5 M parámetros. Parte de COCO y se hace fine-tuning a 1 clase (`0=pileta`).

⚠️ **Precisión importante:** cargamos `yolov5nu.pt`, no `yolov5n.pt`. La `u` significa backbone de YOLOv5 **con la cabeza anchor-free desacoplada de YOLOv8** (incluye pérdida DFL). Por eso **no hay anclas que ajustar**. Se verifica en los propios datos: `results.csv` tiene columna `train/dfl_loss`, que el YOLOv5 original no tiene. El modelo tiene **25 módulos (0–24)** y `Detect` es el 24 — dato que importa para el parámetro `freeze` de la Etapa 4B.

## 4. Medidas e índices del modelo

- **Detección:** mAP@[0.50:0.95], mAP@0.50, mAP@0.75, AP pileta @0.5, mAR@100 (vía `torchmetrics`, GT y preds en píxeles originales).
- **Clasificación (IoU 0.5, umbral óptimo):** Precisión, Recall, F1, TP/FP/FN, IoU promedio de TP y curva PR de pileta.
- **Umbral óptimo:** barrido 0.05–0.90 en VAL maximizando F1 de pileta (en VAL, nunca en TEST).
- **Entrenamiento:** pérdidas box/cls/dfl por época + mAP de val (de `results.csv`), épocas ejecutadas y mejor época.
- **Costo:** parámetros (M), peso .pt/.onnx (MB), GFLOPs, ms/imagen y FPS en GPU y CPU, tabla trade-off ONNX.

## 5. Restricciones del tier gratis y cómo se mitigan

- `imgsz=320`, batch 8, 25 épocas con **early stopping** (paciencia 6) y AMP automático de ultralytics.
- `workers=2`, checkpoint `best.pt` automático (sobrevive cortes: reejecutás desde la Etapa 5 sin reentrenar).
- Inferencia masiva de las 10.2k solo con bandera `TODAS=True` (por defecto, galería de 50).
- El código frena con mensaje claro si no hay GPU (entrenar en CPU es inviable; inferir suelto sí se puede).

---

## ETAPA 1 — Entorno, GPU y descarga automática
**Qué hace:** deja Colab listo y baja el dataset sin intervención.
**Por qué:** sin GPU el entrenamiento tarda horas por época; el código lo verifica primero y frena con instrucciones en vez de quemar tiempo. La descarga usa `kagglehub.dataset_download` (sin `kaggle.json`, sin adapter PANDAS que solo sirve para tabulares) y guarda todo en `RUTA_DATASET`, variable que usan las demás etapas.
**Qué mirar:** que diga una GPU (T4), versiones impresas y conteo de imágenes por carpeta.

```text
Generá el código de una celda de Google Colab que haga lo siguiente, con comentarios cortos en español en cada bloque:
1. Instalar con subprocess (no con !pip, para que también sirva como script .py): kagglehub, ultralytics, albumentations, torchmetrics, thop, onnx, onnxruntime.
2. Verificar torch.cuda.is_available(): si hay GPU imprimir su nombre; si NO hay, frenar con SystemExit y el mensaje de activar GPU T4 en Entorno > Cambiar tipo de entorno. Definir DEVICE=0.
3. Imprimir versiones de torch, ultralytics, albumentations y torchmetrics. Fijar semillas en 42 (random, numpy, torch, cuda) con función set_seed.
4. Descargar AUTOMÁTICAMENTE con import kagglehub; RUTA_DATASET = Path(kagglehub.dataset_download("kbhartiya83/swimming-pool-and-car-detection")). Imprimir la ruta. Prohibido pedir kaggle.json o interacción.
5. Crear /content/runs, /content/resultados y /content/yolo_data. Mostrar el árbol del dataset (2 niveles) con conteo de JPG/JPEG/PNG por carpeta.
Explicá al final qué quedó en cada carpeta y que RUTA_DATASET la usan todas las etapas siguientes.
```

## ETAPA 2 — EDA + verificación de clases
**Qué hace:** parsea los XML Pascal VOC, describe el dataset y dibuja una grilla 2×4 con boxes coloreados por clase cruda.
**Por qué:** es el paso que evita el peor bug posible — entrenar sin piletas. Los XML traen clases `"1"` y `"2"` sin decir cuál es cuál; solo el ojo (piletas = rectángulos grandes junto a casas) lo resuelve. El código fija `CLASE_PILETA_XML` con assert que revienta si el valor no existe; el resto de clases se descarta.
**Qué mirar:** la grilla `resultados/eda_ejemplos.png`. Si el color asumido no corresponde a piletas, cambiar la variable y re-ejecutar desde acá.

```text
Generá el código de Colab de la EDA, con comentarios cortos en español:
1. Buscar recursivamente los *.xml bajo RUTA_DATASET. Parsear cada uno (filename, size, objects con name y bndbox) con xml.etree y armar un DataFrame con columnas image_id,xmin,ymin,xmax,ymax,class (class = texto crudo del XML).
2. Imprimir: cantidad de XML, clases crudas únicas con conteo, cantidad de imágenes anotadas y de anotaciones, y las primeras 5 filas.
3. Localizar las imágenes en disco (misma carpeta del XML o subcarpetas, extensiones jpg/jpeg/png) con función localizar_imagen(nombre) que busque por stem.
4. Dibujar grilla 2x4 con 8 imágenes aleatorias (semilla 42) y sus boxes, un color por clase cruda con leyenda del valor. Guardar en resultados/eda_ejemplos.png.
5. Definir CLASE_PILETA_XML="2" con assert de existencia e imprimir el mapeo fijado (XML->YOLO 0=pileta, resto descartado).
Explicá que este es el único paso con decisión humana y por qué un mapeo mal deja cero piletas en el entrenamiento.
```

## ETAPA 3 — Split 70/15/15 + conversión a formato YOLO
**Qué hace:** divide estratificado por presencia de pileta y genera `images/` + `labels/` + `data.yaml` con `nc: 1`.
**Por qué:** YOLO no lee XML: necesita un `.txt` por imagen con `class cx cy w h` normalizado. El split estratificado garantiza que las piletas (minoritarias) caigan en train, val y test; si alguna combinación tiene <2 muestras, se usa fallback sin estratificar (si no, `train_test_split` revienta). Solo se escriben boxes de piletas; el resto se ignora. Al final se verifica que los boxes convertidos sumen exactamente los de pileta del XML.
**Qué mirar:** los porcentajes impresos (tienen que ser 70/15/15), conteos de piletas por split (ningún split en 0) y los "OK" de la verificación txt-vs-XML.

⚠️ **Bug real que tuvimos (corregido):** el código hacía `resto, df_test = split_seguro(df_img)` y después partía **`resto` (el 70 %)** a la mitad, dejando **35/35/30** en vez de 70/15/15. Se entrenó con la mitad de los datos disponibles sin que nada fallara. La versión corregida es `df_train, temp = split_seguro(df_img)` y después se parte **`temp` (el 30 %)**, con asserts que revientan si las proporciones no dan.

```text
Generá el código de Colab que prepare los datos YOLO, con comentarios cortos en español:
1. Armar DataFrame por imagen (tiene_pileta según CLASE_PILETA_XML) + strat_key. Dividir 70/15/15 con train_test_split semilla 42 y stratify, con fallback sin estratificar si algún strat_key tiene <2 muestras. OJO: train_test_split(df, test_size=0.30) devuelve (70 %, 30 %) — el 15/15 sale de partir el SEGUNDO elemento, no el primero. Agregar asserts de que cada split quedó en su proporción y que no se solapan. Imprimir tamaños, porcentajes y conteo de piletas por split.
2. Para cada split, copiar cada imagen a yolo_data/images/{split}/ y escribir yolo_data/labels/{split}/{stem}.txt con líneas "0 cx cy w h" normalizadas por el W,H REAL de cada imagen (leer con PIL, nada hardcodeado). Mapeo: pileta->0; ignorar el resto.
3. Escribir yolo_data/data.yaml con path/train/val/test absolutos, nc: 1, names ['pileta']. Mostrarlo.
4. Verificar por split que la cantidad de líneas en los txt coincida con los boxes de pileta del XML e imprimir OK o REVISAR.
Explicá por qué la normalización usa el tamaño real de cada imagen y por qué el fallback del split existe.
```

## ETAPA 4 — Entrenamiento YOLOv5n
**Qué hace:** fine-tuning desde `yolov5nu.pt` (COCO) con early stopping.
**Por qué:** con ~2.600 imágenes de train, partir de COCO es lo que hace que 25 épocas alcancen. ultralytics maneja solo AMP, checkpoint `best.pt`/`last.pt`, `results.csv` y restauración del mejor peso al cortar por paciencia. Tiempo esperado en T4: 10–15 min.
**Qué mirar:** que el mAP de val suba en el log, el tiempo total y que exista `runs/yolov5n/weights/best.pt`.

```text
Generá el código de Colab que entrene, con comentarios cortos en español:
1. from ultralytics import YOLO; modelo = YOLO("yolov5nu.pt") (se descarga solo). Entrenar con data=data.yaml, imgsz=320, batch=8, epochs=25, patience=6, seed=42, workers=2, device=DEVICE, project=/content/runs, name="yolov5n", exist_ok=True. Medir e imprimir el tiempo total en minutos.
2. Verificar con assert que existe runs/yolov5n/weights/best.pt e imprimir su ruta (variables BEST_PT y RESULTS_CSV para las etapas siguientes).
Explicá qué es early stopping con paciencia 6 y por qué best.pt permite retomar sin reentrenar si se corta la sesión.
```

## ETAPA 4B — Experimento: capas congeladas vs no congeladas
**Qué hace:** entrena la misma arquitectura con 3 estrategias de transfer learning (`freeze=0`, `10`, `24`) y compara mAP, tiempo por época, memoria y parámetros entrenables.
**Por qué:** sin esto, la sección de capas congeladas del informe es solo teoría. Con esto hay una tabla medida.
**Costo:** 3 entrenamientos, ~20–30 min cada uno en T4 (60–90 min total). Se puede saltear con `CORRER_FREEZE = False`.
**Qué mirar:** `resultados/freeze_comparativa.csv` y el PNG con las 3 curvas de mAP.

**Datos verificados cargando el modelo** (no asumir):

| `freeze` | Entrenables | Qué queda libre |
|---|---|---|
| 0 | 2.636 M (100 %) | Todo: fine-tuning completo |
| 10 | 1.589 M (60.3 %) | Neck + head |
| 24 | 0.878 M (33.3 %) | Solo la cabeza `Detect` |
| 25 | 0 M | ❌ ultralytics aborta con `RuntimeError` |

⚠️ **Dos trampas que encontramos:**
1. **`freeze=22` no es "solo la cabeza".** YOLOv5 tiene 25 módulos y `Detect` es el 24, así que `freeze=22` deja los módulos 23 **y** 24 entrenables (45 %). Para feature extraction real hay que usar `freeze=24`.
2. **No contar los parámetros entrenables sobre el modelo recién cargado.** `yolov5nu.pt` viene con `nc=80` y ultralytics reconstruye la cabeza para `nc=1` al entrenar, así que el conteo previo sobreestima en 19.200 params. Hay que medirlo con un callback `on_train_start`, que es el único momento donde el freeze ya se aplicó y el modelo todavía no se fusionó.

```text
Generá el código de Colab del experimento de transfer learning, con comentarios cortos en español:
1. Bandera CORRER_FREEZE y EPOCHS_FREEZE=EPOCHS. Lista CONFIGS con 3 tuplas (etiqueta, freeze, descripción): ("completo",0), ("backbone",10), ("solo_cabeza",24). Aclarar en comentario que yolov5nu tiene 25 módulos (0..24), Detect es el 24, y que freeze=25 deja 0 entrenables y aborta.
2. Función medir_entrenables(modelo, dict) que registre un callback on_train_start guardando params con requires_grad, total y cantidad de tensores congelados. NO contar sobre el modelo recién cargado: viene con nc=80 y la cabeza se reconstruye para nc=1.
3. Para cada config: cargar YOLO("yolov5nu.pt"), registrar el callback, resetear torch.cuda.reset_peak_memory_stats(), entrenar con freeze=N y los mismos imgsz/batch/epochs/seed que la Etapa 4, name=f"freeze_{N}". Medir minutos y pico de memoria.
4. Leer results.csv de cada corrida (hacer .strip() a los nombres de columna), tomar el máximo de metrics/mAP50(B) y mAP50-95(B) y la mejor época. Armar DataFrame y guardarlo en resultados/freeze_comparativa.csv.
5. Imprimir la tabla y los deltas respecto de freeze=0 (diferencia de mAP, % más rápido o más lento por época, % entrenable). Graficar las 3 curvas de mAP50-95 de val por época en resultados/freeze_comparativa.png.
Explicá qué significa congelar capas, por qué con dominio satelital vs COCO conviene fine-tuning completo, y cómo leer el trade-off entre mAP y costo.
```

## ETAPA 5 — Evaluación + umbral óptimo
**Qué hace:** mide en TEST (mAP con torchmetrics + P/R/F1/IoU con matching greedy propio mono-clase), elige el umbral óptimo en VAL y mide velocidad/tamaño reales.
**Por qué es así:**
- GT y predicciones se comparan en **píxeles originales** (YOLO ya devuelve los boxes reescalados; no hay reescalado manual que pueda fallar).
- mAP se calcula con `conf=0.001` (sin filtrar) y las métricas clásicas con el umbral óptimo: son dos preguntas distintas (ranking vs decisión).
- El barrido de umbrales predice VAL **una sola vez** y después solo filtra por score: 18 umbrales sin re-evaluar 18 veces (la versión vieja hacía 56 pasadas y colgaba la T4).
- Helpers únicos sin duplicar: `predecir(ruta, conf, iou, device, modelo_ref)`, `par_torch()` y `coleccionar_pares()` se reutilizan en la Etapa 9 para el ONNX.
- Nada estimado a ojo: MB del archivo real, params del modelo real, FPS medidos con warmup + sincronización CUDA.
**Qué mirar:** `resultados/metricas.csv`, el umbral óptimo impreso y que el AP de pileta sea > 0 (si da 0, volver a la Etapa 2: mapeo mal).

⚠️ **Bug real que tuvimos (corregido):** el barrido arrancaba en **0.25** y el óptimo salió justo **0.25**, es decir pegado al borde inferior del rango. Eso no es un máximo, es el rango cortado: el F1 real podía seguir subiendo con umbrales más bajos y nunca lo medimos. Ahora el barrido arranca en **0.05** y, si el óptimo igual cae en cualquiera de los dos extremos, el código imprime un aviso explícito en vez de dejarlo pasar en silencio. **Regla general:** si un óptimo de un barrido coincide con el primer o el último valor probado, el barrido está mal acotado.

```text
Generá el código de Colab de la evaluación, con comentarios cortos en español:
1. Cargar YOLO(BEST_PT). Funciones: gt_original(nombre) que lee boxes de pileta del XML en píxeles originales como array (N,4); predecir(ruta, conf, iou, device, modelo_ref=None) que usa model.predict y devuelve boxes/scores/cls (manejar cero detecciones); clave_por_img(stem) que resuelve la extensión del archivo; iou_una(a,b); metricas_greedy mono-clase que devuelve TP/FP/FN/P/R/F1/IoU_prom; par_torch() y coleccionar_pares() para armar preds/targets sin duplicar código.
2. Predecir todo TEST con conf=0.001 vía coleccionar_pares y calcular con torchmetrics: mAP@[.5:.95], mAP@.5, mAP@.75 (cada uno con su instancia), AP pileta @.5 y mAR@100. Imprimirlos.
3. Predecir VAL una vez con conf=0.01; barrer umbrales 0.05-0.90 paso 0.05 recalculando F1 de pileta (lista SWEEP de tuplas thr,P,R,F1); OPT_THR = umbral de F1 máximo. Imprimir la tabla del barrido y un AVISO si OPT_THR cae en el primer o el último umbral probado (señal de que el rango no cubre el máximo real).
4. Con OPT_THR, calcular métricas clásicas en TEST e imprimir P/R/F1/IoU/TP-FP-FN de pileta.
5. Velocidad: warmup 5 imgs, promedio de 30 en GPU con torch.cuda.synchronize (ms/img y FPS) + 10 en CPU con device="cpu". Tamaño: MB del .pt, millones de params de model.model, GFLOPs con thop en try/except (N/D si falla).
6. Guardar todo en resultados/metricas.csv (columnas metrica,valor).
Explicá la diferencia entre mAP (ranking, conf baja) y P/R/F1 (decisión, umbral óptimo) y por qué el umbral se elige en VAL y no en TEST.
```

## ETAPA 6 — Gráficos
**Qué hace:** convierte `results.csv` y el barrido en 3 PNG: pérdidas, mAP de val, F1 vs umbral y curva PR de pileta.
**Por qué:** las curvas dicen si el modelo convergió (pérdidas bajando, mAP subiendo y estable) o si hay overfitting (mAP de val que cae mientras el train sigue bajando). La curva PR muestra el trade-off precisión/recall de un vistazo.
**Qué mirar:** mAP de val estable al final (early stopping hizo su trabajo).

```text
Generá el código de Colab de los gráficos, con comentarios cortos en español:
1. Leer RESULTS_CSV. Plot 1: train/box_loss, train/cls_loss (+train/dfl_loss si existe) por época. Plot 2: metrics/mAP50(B) y metrics/mAP50-95(B) por época. Guardar resultados/curvas_entrenamiento.png. Imprimir épocas entrenadas.
2. Con SWEEP (thr,P,R,F1): plot F1 pileta vs umbral + línea vertical en OPT_THR (resultados/f1_vs_umbral.png). Plot PR usando (R,P) del barrido (resultados/curvas_pr.png).
Explicá cómo leer cada gráfico y qué patrón indicaría overfitting.
```

## ETAPA 7 — Bounding boxes sobre el test etiquetado
**Qué hace:** dibuja GT vs predicción lado a lado en 8 imágenes de test.
**Por qué:** las métricas son números; el ojo detecta lo que los números esconden (boxes sistemáticamente corridos o piletas ignoradas). Colores fijos: GT rojo fino, pileta azul + score.
**Qué mirar:** `resultados/bboxes_test/comparativa_gt_vs_pred.png` buscando falsos positivos y piletas perdidas.

```text
Generá el código de Colab de visualización, con comentarios cortos en español:
1. Función dibujar(imagen_rgb, dets, gt=None) con OpenCV: dets es lista [(box,score)] de pileta; GT (array) en rojo fino (0,0,255), pileta en azul (255,0,0) con texto "pileta score". Recordar que OpenCV usa BGR: convertir al entrar y al salir.
2. Elegir 8 índices aleatorios del test (semilla 42). Grilla 4x2 de pares GT | YOLOv5n (thr=OPT_THR) reutilizando test_pred/test_gt de la Etapa 5 (sin re-predecir). Guardar en resultados/bboxes_test/comparativa_gt_vs_pred.png.
Explicá el código de colores y los 2 errores típicos a buscar a ojo.
```

## ETAPA 8 — Inferencia sobre las 10.2k sin etiquetas
**Qué hace:** galería de 50 imágenes reales sin anotar + modo completo opcional a CSV.
**Por qué:** en las 10.2k no hay GT, así que no hay métricas: el valor es cualitativo (¿generaliza a otras zonas?) más el CSV (`image_id,xmin,ymin,xmax,ymax,class,conf`, class 0=pileta). Por defecto solo la galería (rápido); `TODAS=True` corre todo con estimación de tiempo previa.
**Qué mirar:** `resultados/galeria_10k.png`. Si ves muchos falsos positivos en agua/techos, el umbral está bajo para ese dominio.

```text
Generá el código de Colab de inferencia masiva, con comentarios cortos en español:
1. Ubicar la carpeta con >1000 imágenes (test_data_images o raíz) y listar jpg/jpeg/png. Tomar muestra aleatoria de 50 (semilla 42), predecir con OPT_THR, dibujar con dibujar() y armar galería 5x10 en resultados/galeria_10k.png.
2. Bandera TODAS=False: si True, estimar tiempo con ms_gpu de la Etapa 5, predecir todo con tqdm y guardar resultados/detecciones_10k.csv (columnas image_id,xmin,ymin,xmax,ymax,class,conf; class 0=pileta; coordenadas originales).
Explicá por qué acá no existen métricas y para qué sirve el CSV.
```

## ETAPA 9 — Export ONNX + trade-off real (única exportación)
**Qué hace:** exporta a ONNX UNA sola vez (si ya existe lo reutiliza) y mide peso, FPS-CPU y mAP antes/después. Todo medido, nada inventado. La Etapa 11 solo empaqueta, no exporta.
**Por qué:** el `.pt` necesita PyTorch; el `.onnx` corre en cualquier lado (CPU, OpenVINO, etc.). La tabla dice si la portabilidad sale gratis o cuesta mAP. El mAP del ONNX se mide con el mismo método de la Etapa 5 sobre 100 imágenes de test (no con APIs internas de ultralytics que cambian entre versiones).
**Qué mirar:** `resultados/trade_off_onnx.csv`: delta de mAP ≈ 0 y FPS-CPU mayor en ONNX es el resultado esperado.

⚠️ **Bug real que tuvimos (corregido):** se medía el ONNX sobre **100 imágenes de VAL** y se lo comparaba contra el mAP del `.pt` sobre **TEST completo**. Son conjuntos distintos, así que el delta no medía la pérdida del export: medía la diferencia entre dos muestras de datos. Ahora los dos modelos se evalúan sobre **el mismo subconjunto de test** (`sub_cmp = imgs_test[:100]`) y el CSV guarda `map50_pt_sub`, `map50_onnx_sub`, `n_imgs_comparacion` y, aparte, `map50_pt_test_completo`. **Regla general:** una comparación A vs B solo vale si lo único que cambia entre las dos mediciones es A y B.

```text
Generá el código de Colab de exportación, con comentarios cortos en español:
1. ÚNICA exportación: onnx_path = runs/yolov5n/weights/best.onnx; si no existe, onnx_path = Path(modelo.export(format="onnx")). Si ya existe, reutilizar sin re-exportar. Imprimir MB del .onnx vs MB del .pt.
2. Medir FPS en CPU (device="cpu", 10 imágenes) del .pt y del YOLO(onnx) con helper fps_cpu. Imprimir ambos.
3. Medir mAP@.5 del .pt Y del ONNX sobre EL MISMO subconjunto (sub_cmp = imgs_test[:100]), reutilizando coleccionar_pares(sub_cmp, conf=0.001) y coleccionar_pares(sub_cmp, conf=0.001, modelo_ref=modelo_onnx). Imprimir el delta con signo. NO comparar contra el mAP de TEST completo: serían conjuntos distintos y el delta no significaría nada. Sin duplicar el loop de extracción.
4. Guardar resultados/trade_off_onnx.csv (peso_pt_MB, peso_onnx_MB, fps_cpu_pt, fps_cpu_onnx, map50_pt_sub, map50_onnx_sub, delta_map50, n_imgs_comparacion, map50_pt_test_completo).
Explicá qué es ONNX y por qué el delta de mAP debería ser ~0 (misma red, otro formato).
```

## ETAPA 10 — Informe final
**Qué hace:** escribe `informe_final.md` con los números reales: datos, entrenamiento, métricas, costo, veredicto y mejoras. Usa f-strings sobre las variables ya calculadas; ningún número se escribe a mano.
**Por qué:** el informe es la entrega: alguien que no corrió nada entiende qué se hizo y qué dio. Incluye la mejor época (argmax del mAP de val) y explicación de por qué el umbral se eligió en VAL.
**Qué mirar:** que todos los números del informe coincidan con `metricas.csv`.

```text
Generá el código de Colab del informe, con comentarios cortos en español:
1. Calcular mejor_epoca = argmax de metrics/mAP50-95(B) en df_res + 1.
2. Escribir resultados/informe_final.md con f-strings: datos (imágenes, anotaciones, split 70/15/15 por pileta, mapeo XML->YOLO 0=pileta), entrenamiento (pesos COCO, imgsz, batch, épocas, mejor época), métricas TEST con umbral óptimo (mAP, AP pileta, P/R/F1/IoU), costo (params, MB, GFLOPs, ms/FPS GPU/CPU, referencia al CSV de ONNX, nota nc=1), veredicto (chico/rápido/ideal CPU) y 5 mejoras futuras (TTA flip, imgsz 640, más aumentos/copy-paste, barrido de hparams, TFLite/OpenVINO).
3. Imprimir el informe.
Explicá por qué el informe se genera con código y no a mano (trazabilidad de números).
```

## ETAPA 11 — Empaquetar y descargar
**Qué hace:** copia `best.pt`, `.onnx` (ya exportado una sola vez en la Etapa 9) y `data.yaml` a `resultados/modelos/`, zipea todo y lo descarga. No exporta nada.
**Por qué:** el ZIP es el artefacto portable: con `modelos/` + `metricas.csv` + PNGs + informe se puede demoar o seguir trabajando sin reentrenar. El `try/except` en `files.download` permite correr el script fuera de Colab sin romper.
**Qué mirar:** el listado de archivos del ZIP y su tamaño en MB.

```text
Generá el código de Colab de empaquetado, con comentarios cortos en español:
1. Crear resultados/modelos/, copiar BEST_PT como yolov5n_best.pt, el .onnx si existe y data.yaml (sin import shutil duplicado: ya está en la Etapa 3).
2. Crear /content/yolov5n_piletas_resultados.zip con shutil.make_archive desde resultados/. Imprimir ruta, tamaño en MB y listado relativo de archivos.
3. Descargar con files.download dentro de try/except (fallback: imprimir la ruta si no estamos en Colab).
Explicá qué contiene cada parte del ZIP.
```

## ETAPA 12 — Demo: subí UNA imagen y detectá 🏊
**Qué hace:** pegás un link http (o subís el archivo) y devuelve la imagen original con boxes de pileta + resumen legible.
**Por qué es simple y correcta:** YOLO ya entrega los boxes en píxeles originales (deshace su letterbox interno), así que se dibujan directo sin reescalado manual (el reescalado a mano era fuente clásica de bugs: ejes x/y mezclados). Usa `OPT_THR` de la Etapa 5, no 0.5 a ciegas.
**Qué mirar:** `resultados/demo_resultado.png` y el texto "Detectado: N pileta(s) [...]". Si no detecta nada, el mensaje sugiere bajar el umbral.

```text
Generá el código de Colab de la demo final, con comentarios cortos en español:
1. Variable URL_IMAGEN = "" (pegar link directo .jpg/.png). Si está vacía, fallback a files.upload() con try/except y SystemExit claro si tampoco hay upload. Descargar con requests (headers de navegador, timeout 20).
2. Abrir con PIL, imprimir nombre y tamaño original. Predecir con predecir() y conf=OPT_THR sobre un temporal que se borra al final.
3. Dibujar con dibujar() sobre la ORIGINAL, mostrar grande con matplotlib y guardar resultados/demo_resultado.png.
4. Imprimir "Detectado: N pileta(s) [scores]"; si cero, mensaje con el umbral usado y sugerencia de bajarlo o cambiar de imagen.
Explicá en 2 líneas por qué no hay reescalado manual (YOLO devuelve píxeles originales).
```

---

## 6. Sugerencias y mejoras futuras

1. **TTA con flip horizontal** en la demo: suele dar +1/2 puntos de mAP a cambio de x2 tiempo.
2. **`imgsz=640`** si conseguís más VRAM: ayuda sobre todo a piletas chicas/lejanas.
3. **Más aumentos si faltan piletas**: mosaic ya viene activo en YOLO; sumar copy-paste o más jitter.
4. **Barrido corto de hiperparámetros** (lr, weight_decay): 2–3 corridas de 10 épocas y quedarse con la mejor.
5. **TFLite/OpenVINO** si la demo sale de Colab hacia CPU o móvil.
6. **Monitoreo de overfitting** en `curvas_entrenamiento.png`: si el mAP de val cae mientras el train baja, cortar antes o congelar backbone.
7. **Si una sesión se corta**: reejecutar Etapas 1–3 (rápidas) y cargar `best.pt` en la 5 sin reentrenar.

## 7. Tiempos estimados en T4 gratis

| Etapas | Tiempo aprox. |
|---|---|
| 1–3 (setup, EDA, split+YOLO) | ~10 min |
| 4 (entrenamiento YOLOv5n) | 20–30 min |
| 4B (experimento freeze, 3 corridas) | 60–90 min (opcional: `CORRER_FREEZE = False`) |
| 5 (evaluación + barrido) | ~10 min |
| 6–7 (gráficos, boxes) | ~5 min |
| 8 (galería 50) | ~3 min (completa: +30–40 min) |
| 9–12 (ONNX, informe, ZIP, demo) | ~10 min |
| **Total sin 4B** | **~60–75 min** |
| **Total con 4B** | **~2–2.5 h** |

> La Etapa 4 ahora tarda el doble que antes (10–15 min → 20–30 min) porque el split corregido entrena con el 70 % de las imágenes en vez del 35 %. No es una regresión: antes se estaba desperdiciando la mitad del dataset.

> Orden: 1 → 12 en secuencia. La única pausa obligatoria es la **Etapa 2** (confirmar a ojo el mapeo de clases en la grilla antes de seguir).
