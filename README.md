# Visión por Computadora — Trabajo Práctico Integrador

Detección de piletas de natación en imágenes satelitales con **YOLOv5n**, incluyendo
*transfer learning* medido experimentalmente, análisis de métricas de detección y
auditoría crítica del propio pipeline.

**Maestría en Inteligencia Artificial** · Universidad de Palermo
Electiva: Visión por Computadora · Profesor: Ing. Marcelo Bozunovsky

| | |
|---|---|
| **mAP@0.5** | 0.7620 |
| **F1** | 0.7986 (P = 0.9049 · R = 0.7146) |
| **Modelo** | 2.64 M parámetros · 5.48 MB · 0.967 GFLOPs |
| **Velocidad** | 11.75 ms/img en GPU (85.1 FPS) · ~29 FPS en CPU |
| **Hallazgo principal** | Congelar el *backbone* (`freeze=10`) mejora el mAP **+16.6 %** |
| **Entorno** | Google Colab, GPU T4 (tier gratuito) |

📄 **[Informe completo (PDF, 24 páginas)](informe/informe_final.pdf)** · [versión Markdown](informe/informe_final.md)

---

## 1. El sentido del trabajo

### Qué pide la materia

El examen final de la asignatura requiere un **trabajo práctico integrador**: desarrollar
una solución de visión por computador aplicable a un problema específico, recorriendo
**todas las etapas del pipeline** —desde la selección y procesamiento de datos hasta la
implementación y evaluación del modelo final— acompañada de un informe técnico que
documente los pasos seguidos, los desafíos enfrentados y los resultados obtenidos.

La evaluación pondera: conocimiento específico aplicado (60 %), originalidad en la
presentación (25 %) y claridad en la defensa (15 %).

### Por qué detección de piletas

El caso elegido es **detección de objetos pequeños en imágenes aéreas**, un problema con
tres propiedades que lo hacen útil como ejercicio:

1. **Es un problema real, no un dataset de juguete.** La detección de piletas en imágenes
   satelitales se usa efectivamente para fiscalización tributaria y catastral.
2. **Tiene una dificultad genuina:** objetos muy pequeños respecto del encuadre, vista
   cenital sin perspectiva, y una **distancia de dominio grande** respecto de COCO (donde
   se pre-entrenó el modelo), lo que vuelve interesante la discusión de *transfer learning*.
3. **Cabe en el presupuesto de una GPU gratuita**, condición necesaria para que el trabajo
   sea reproducible por cualquiera.

> **Nota de alcance.** Las piletas son el caso de estudio, no el objetivo final. El interés
> de fondo es el análisis automático de **planos e imágenes técnicas completas**; este TP
> funciona como demostración controlada del pipeline sobre un problema acotado y medible.

### Qué se buscó demostrar

El trabajo no se propone maximizar una métrica, sino **entender qué dice cada número y
dónde falla el método**. Dos ejes:

- **Transfer learning medido, no argumentado.** Se compararon tres estrategias de
  congelamiento a igual semilla, resolución y presupuesto de épocas. El resultado
  **refutó la hipótesis de partida del propio trabajo** (ver §3).
- **Auditoría del propio pipeline.** El informe documenta **cuatro defectos metodológicos
  propios** (§10) en lugar de omitirlos. Comparten un patrón: son **fallas silenciosas**
  que producen números plausibles y atraviesan la ejecución completa sin lanzar un error.

Esa es la conclusión transversal: **en visión por computadora el riesgo principal no es el
código que falla, sino el que funciona sobre premisas equivocadas.**

---

## 2. Construcción del pipeline

El pipeline tiene **12 etapas** (más una 4B), pensadas para ejecutarse en orden en Colab.
Cada etapa es una celda del notebook.

| Etapa | Qué hace | Tiempo en T4 |
|---|---|---|
| 1 | Entorno, verificación de GPU y descarga automática del dataset | ~5 min |
| 2 | EDA + **verificación visual del mapeo de clases** ⚠️ | ~3 min |
| 3 | Split 70/15/15 estratificado + conversión a formato YOLO | ~3 min |
| 4 | Entrenamiento (*fine-tuning* desde pesos COCO) | ~7 min |
| **4B** | **Experimento: capas congeladas vs. no congeladas** | ~28 min |
| 5 | Evaluación en test + barrido del umbral óptimo en validación | ~10 min |
| 6 | Gráficos: pérdidas, mAP por época, F1 vs. umbral, curva PR | ~3 min |
| 7 | Comparativa visual *ground truth* vs. predicción | ~2 min |
| 8 | Inferencia sobre las ~10.200 imágenes sin etiquetar | ~3 min |
| 9 | Exportación a ONNX + *trade-off* medido | ~3 min |
| 10 | Informe final con los números reales | ~1 min |
| 11 | Empaquetado y descarga del ZIP | ~1 min |
| 12 | Demo: subir una imagen y detectar | opcional |

**Total: ~1 h sin la Etapa 4B, ~1,5 h con ella.**

### Decisiones de diseño

**Modelo: YOLOv5n (variante *nano*).** El más chico de la familia: 2.64 M parámetros. Es
el único que permite entrenar e inferir con holgura en el tier gratuito de Colab. Se parte
de pesos pre-entrenados en COCO.

**Una sola clase (`nc=1`).** El dataset también anota autos, pero se descartan. No por
irrelevantes: incluirlos obligaría al modelo a resolver un problema adicional de
discriminación entre clases sin beneficio para el objetivo.

**Resolución 320 px.** Límite impuesto por la memoria y el tiempo de la T4. Es, con
diferencia, **la limitación de mayor impacto** del trabajo: explica la brecha entre
mAP@0.5 (0.76) y mAP@0.75 (0.36).

**El umbral de confianza se elige en validación, nunca en test.** Elegirlo en test sería
*data leakage*: las métricas reportadas dejarían de estimar el rendimiento en datos no
vistos.

**El modelo real es *anchor-free*.** Se carga `yolov5nu.pt`, no `yolov5n.pt`. La `u`
indica *backbone* de YOLOv5 con la **cabeza anchor-free desacoplada de YOLOv8** (con
pérdida DFL). Se verifica en los datos: `results.csv` tiene columna `train/dfl_loss`, que
el YOLOv5 original no produce.

### La pausa obligatoria de la Etapa 2

La Etapa 2 **requiere intervención humana**: hay que mirar una grilla de ejemplos y
confirmar que la clase `"2"` del XML corresponde efectivamente a las piletas.

No es un paso decorativo. Con el mapeo invertido, el pipeline entrena **sin un solo error**
sobre **cero** instancias de la clase objetivo y reporta mAP = 0. Ya ocurrió una vez
durante el desarrollo, y es el origen de la conclusión metodológica del informe.

---

## 3. Resultados

### Transfer learning: el hallazgo principal

Tres estrategias entrenadas con **idéntica semilla (42), resolución (320 px), batch (8) y
presupuesto de épocas (25)**, variando únicamente `freeze`:

| Estrategia | `freeze` | Entrenables | Épocas | mAP@0.5 val | mAP@[.5:.95] val | s/época | Memoria |
|---|---|---|---|---|---|---|---|
| Fine-tuning completo | 0 | 2.636 M (100 %) | **14** (early stop) | 0.7966 | 0.4048 | 31.5 | 0.38 GB |
| **Backbone congelado** | **10** | **1.589 M (60.3 %)** | **25** | **0.8141** | **0.4718** | **25.7** | **0.30 GB** |
| Solo la cabeza | 24 | 0.878 M (33.3 %) | 25 | 0.7338 | 0.3639 | 23.3 | 0.26 GB |

**`freeze=10` gana en las cuatro dimensiones a la vez:** +16.6 % de mAP@[.5:.95], 18 %
menos de tiempo por época, 21 % menos de memoria y 40 % menos de parámetros entrenables.
No hay *trade-off* que negociar.

**Esto refuta la hipótesis del propio trabajo.** El modelo final se entrenó con
`freeze=0`, argumentando que la distancia de dominio entre COCO y las imágenes aéreas
exigía adaptar toda la red. El experimento muestra que ese razonamiento estaba **mal
calibrado**: `freeze=24` sí es peor que la base (−10.1 %), confirmando que hay que adaptar
el *backbone*, pero **el óptimo está en el medio, no en el extremo**.

**El mecanismo medido es la regularización.** `freeze=0` fue la única configuración que
sobreajustó: se detuvo por *early stopping* en la época 14, con su mejor época en la 8 y
`val/box_loss` subiendo de 1.620 a 1.875. `freeze=10` llegó a la época 25 con la pérdida
de validación en su mínimo y el mAP **todavía subiendo**.

### Métricas en test (563 imágenes, 466 piletas)

| Métrica | Valor | Qué significa |
|---|---|---|
| mAP@0.5 | **0.7620** | Calidad del *ranking* con criterio de localización laxo |
| mAP@[.5:.95] | 0.3882 | Promedio sobre 10 umbrales de IoU: métrica estándar de COCO |
| mAP@0.75 | 0.3563 | Con criterio estricto: **cae a la mitad** |
| mAR@100 | 0.4964 | Techo de *recall* permitiendo 100 detecciones por imagen |
| Precisión | 0.9049 | De cada 100 avisos, ~90 son piletas reales |
| Recall | 0.7146 | Encuentra ~71 de cada 100 piletas |
| F1 | 0.7986 | Media armónica de las dos anteriores |
| IoU promedio | 0.7753 | Calidad de las cajas acertadas |
| Umbral óptimo | 0.20 | Maximiza F1 **en validación** (valor interior del rango) |

**Matriz de confusión:** TP = 333, FP = 35, FN = 133. Validada contra el recuento
independiente de los XML (466 piletas en test).

### Las dos lecturas que importan

**El modelo clasifica bien pero localiza con imprecisión.** La brecha entre mAP@0.5 (0.76)
y mAP@0.75 (0.36) es el hallazgo más informativo: el problema no es *reconocer* piletas
sino *delimitarlas* con precisión. Es la consecuencia esperable de entrenar a 320 px.

**Es un modelo conservador.** 133 falsos negativos contra 35 falsos positivos: confiable
cuando avisa, pero omite una de cada cuatro piletas. Para fiscalización, esa asimetría
es la indeseada —conviene bajar el umbral y ganar *recall* a costa de precisión, que es el
recurso abundante—.

### La corrección que no mejoró nada

Corregir el split (de 35/35/30 a 70/15/15 real) **duplicó los datos de entrenamiento y
bajó las métricas de test**:

| Métrica | Corrida original | Corrida final | Δ |
|---|---|---|---|
| mAP@0.5 | 0.7846 | 0.7620 | −0.0226 |
| Recall | 0.7414 | 0.7146 | −0.0268 |
| **Precisión** | 0.8912 | **0.9049** | **+0.0137** |

La predicción explícita del informe anterior —"debería mejorar los resultados,
especialmente el recall"— **no se cumplió**. Dos causas: los dos conjuntos de test son
particiones distintas y no comparables entre sí, y el modelo empezó a sobreajustar antes
(época 14 en lugar de 25). La predicción fallida se deja registrada en el informe en lugar
de borrarla.

### Los cuatro defectos auditados

| # | Defecto | Impacto | Estado |
|---|---|---|---|
| 1 | El split real era **35/35/30**, no 70/15/15 | Se entrenó con la mitad de los datos | ✅ Corregido y reejecutado: 2.623/562/563 |
| 2 | El *delta* de mAP del ONNX (+0.0864) no era comparable | Medía dos conjuntos distintos | ✅ Corregido: **delta = 0.000000** |
| 3 | El barrido de umbrales arrancaba en **0.25** y el óptimo dio **0.25** | Un óptimo en el borde no es un óptimo | ✅ Corregido: óptimo = **0.20**, interior |
| 4 | La matriz de confusión del informe anterior estaba **subestimada ×4** | TP=172/FP=21/FN=60 en lugar de 688/84/240 | ✅ Corregido: validado contra los XML |

El cuarto se detectó **al validar esta corrida** y es el más incómodo: no fue un error de
código sino de **razonamiento**, dentro de la sección que el informe presentaba como su
auditoría. El sistema `P = TP/(TP+FP)`, `R = TP/(TP+FN)` tiene infinitas soluciones
enteras; se tomó la mínima y se la declaró única. Sobrevivió porque la verificación
aritmética que la acompañaba era correcta. **Que un resultado sea consistente con los
datos no lo vuelve único.**

---

## 4. Estructura del repositorio

```
.
├── notebooks/
│   └── deteccion_piletas.ipynb        # Notebook ejecutado, con outputs (evidencia)
├── src/
│   └── deteccion_piletas.py           # Mismo pipeline como script
├── informe/
│   ├── informe_final.pdf              # Informe de 24 páginas (entregable)
│   ├── informe_final.md               # Fuente en Markdown
│   ├── build_pdf.py                   # Genera el PDF (Chrome headless)
│   └── img/                           # 7 figuras del informe
├── docs/
│   ├── COMO_CORRER_EN_COLAB.md        # Guía de ejecución paso a paso
│   └── guia_12_etapas.md              # Detalle de cada etapa, con los bugs documentados
└── resultados/
    ├── corrida_final/                 # 12-10: pipeline corregido
    │   ├── metricas.csv               #   las 16 métricas reportadas
    │   ├── trade_off_onnx.csv         #   .pt vs .onnx (delta = 0)
    │   ├── freeze_comparativa.csv     #   el experimento de capas congeladas
    │   └── freeze_runs/*.csv          #   historial por época de las 3 corridas
    └── corrida_original/              # 28-09: con el split defectuoso (comparación)
```

**Qué no está en el repo y por qué:**

- **Los pesos** (`best.pt`, `best.onnx`, ~16 MB): se regeneran corriendo el pipeline.
- **El dataset**: se descarga automáticamente desde Kaggle en la Etapa 1.
- **La bibliografía en PDF**: material con derechos de autor. Se cita en §6.

---

## 5. Cómo reproducirlo

### En Google Colab (recomendado)

1. Abrir `notebooks/deteccion_piletas.ipynb` en Colab.
2. **Activar la GPU:** `Entorno de ejecución > Cambiar tipo de entorno > T4 GPU`.
   La Etapa 1 corta con `SystemExit` si no detecta GPU.
3. Ejecutar las celdas en orden. **Detenerse en la Etapa 2** y verificar el mapeo visual.
4. Descargar el ZIP de la Etapa 11 antes de cerrar la sesión: `/content` es efímero.

La guía detallada, con los valores esperados en cada chequeo y qué hacer si Colab se
desconecta, está en **[docs/COMO_CORRER_EN_COLAB.md](docs/COMO_CORRER_EN_COLAB.md)**.

### Dataset

[Swimming Pool and Car Detection](https://www.kaggle.com/datasets/kbhartiya83/swimming-pool-and-car-detection)
(Kaggle, público). Imágenes satelitales con anotaciones en XML estilo Pascal VOC.
**No requiere `kaggle.json`:** se baja automáticamente vía `kagglehub`.

### Entorno de la corrida reportada

`ultralytics` 8.4.175 · `torch` 2.11.0+cu130 · Python 3.13.15 · GPU Tesla T4.

> **Sobre la versión de `ultralytics`:** conviene fijarla. Entre 8.4.164 y 8.4.175 la rama
> `model.24.cv3` de la cabeza pasó de 64 a 80 canales, cambiando el recuento de parámetros
> del mismo modelo de 2.51 M a 2.64 M (+127.152). El recuento de parámetros de un modelo
> "pre-entrenado" no es una constante de la arquitectura.

> **Sobre entrenar localmente:** no es viable sin GPU CUDA. Un detector en CPU tarda horas
> por época, así que el script corta a propósito en la Etapa 1 en lugar de quemar tiempo.
> Para *inferencia* suelta, la CPU sí alcanza (~29 FPS).

---

## 6. Materia y bibliografía

### Programa de la asignatura

| Unidad | Módulo | Contenidos |
|---|---|---|
| 1 · Introducción | 1 | Fundamentos e historia de la visión por computador. Problemas clásicos: segmentación, detección, reconstrucción 3D |
| 2 · Procesamiento | 2 | Filtros y máscaras. Operación de convolución. Detección de bordes: Sobel, Prewitt, Canny |
| | 3 | PyTorch y OpenCV. Tensores, normalización, representaciones 1D/2D/3D |
| | 4 | Transformaciones 2D y 3D: euclidianas, afines, homografías. Geometría de múltiples vistas |
| 3 · Deep Learning | 5 | CNNs: capas convolucionales, *pooling*, *fully connected*. Entrenamiento y regularización |
| | 6 | RCNN / Fast RCNN / Mask RCNN. LeNet, AlexNet, GoogleNet, VGG, ResNet |
| | 7 | **YOLO y GANs.** *Bounding boxes*, detección en tiempo real. UNet. Generación de imágenes |
| 4 · Ética | 8 | Sesgos algorítmicos, datos biométricos, reconocimiento facial, el caso COMPAS |

Este TP se ubica principalmente en el **Módulo 7** (YOLO y detección con *bounding boxes*),
apoyándose en los Módulos 5 y 6 para la parte de CNNs y arquitecturas de detección.

Carga: 21 horas teóricas + 13 horas prácticas. Modalidad a distancia.

### Bibliografía obligatoria

- Crespo Cano, R. (s.f.). *Curso de introducción a OpenCV y Python*. Jupyter.
- Goodfellow, I., Bengio, Y. & Courville, A. (2016). *[Deep Learning](https://www.deeplearningbook.org/)*. MIT Press.
- Gil-Delgado, L. M. (2023). *Redes neuronales aplicadas a la generación de imágenes*. Universidad Politécnica de Madrid.
- Loncomilla, P. (2016). *Deep learning: redes convolucionales*. INAOE.
- Morales, E. & Sucar, E. (2022). *Introducción a Visión computacional*. INAOE.

### Bibliografía complementaria

- Minichino, J. & Howse, J. (2020). *Learning OpenCV 4 Computer Vision with Python 3* (3.ª ed.). Packt.
- Szeliski, R. (2022). *[Computer Vision: Algorithms and Applications](https://szeliski.org/Book/)* (2.ª ed.). University of Washington.

### Lecturas adicionales usadas en este TP

- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
- Duda, R., Hart, P. & Stork, D. (2000). *Pattern Classification* (2.ª ed.). Wiley.
- Gonzalez, R. & Woods, R. *Digital Image Processing*. Pearson.
- Buduma, N. *Fundamentals of Deep Learning*. O'Reilly.
- Krizhevsky, A., Sutskever, I. & Hinton, G. (2012). *[ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)* (AlexNet).
- Simonyan, K. & Zisserman, A. (2014). *[Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)* (VGG).

> Los PDFs de estos textos **no se incluyen** en el repositorio por tratarse de material
> con derechos de autor.

### Referencias técnicas

- [Ultralytics YOLO](https://docs.ultralytics.com/) — framework de entrenamiento
- Redmon, J. et al. (2016). *[You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)*
- Lin, T.-Y. et al. (2014). *[Microsoft COCO: Common Objects in Context](https://arxiv.org/abs/1405.0312)* — origen de los pesos y del protocolo de mAP

---

## 7. Próximos pasos

Ordenados por relación entre impacto esperado y esfuerzo (detalle en §11 del informe):

1. **Reentrenar el modelo final con `freeze=10`.** Conclusión directa del experimento: la
   configuración entregada es la peor de las dos razonables, y la ganadora mejora el mAP
   un 16.6 % con menos tiempo y memoria. Es la única mejora con evidencia propia.
2. **Entrenar a `imgsz=640`:** vía directa para atacar el mAP@0.75 = 0.3563 y los 133
   falsos negativos, concentrados en piletas pequeñas.
3. **Repetir el experimento de `freeze` con 3–5 semillas y evaluar en test**, no solo en
   validación: daría estimación de varianza al hallazgo principal.
4. **Explorar valores intermedios de `freeze`** (4, 7, 13, 16): si el óptimo es interior,
   conviene localizarlo con más de tres puntos.
5. **Ajustar la regularización** ahora que se conoce el patrón de sobreajuste, antes de
   agregar más datos.

---

*Trabajo académico — Maestría en Inteligencia Artificial, Universidad de Palermo.*
