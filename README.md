# Visión por Computadora — Trabajo Práctico Integrador

Detección de piletas de natación en imágenes satelitales con **YOLOv5n**, incluyendo
*transfer learning*, análisis de métricas de detección y auditoría crítica del propio
pipeline.

**Maestría en Inteligencia Artificial** · Universidad de Palermo
Electiva: Visión por Computadora · Profesor: Ing. Marcelo Bozunovsky

| | |
|---|---|
| **mAP@0.5** | 0.7846 |
| **F1** | 0.8094 (P = 0.8912 · R = 0.7414) |
| **Modelo** | 2.51 M parámetros · 5.23 MB · 0.897 GFLOPs |
| **Velocidad** | 9.54 ms/img en GPU (104.8 FPS) · ~27 FPS en CPU |
| **Entorno** | Google Colab, GPU T4 (tier gratuito) |

📄 **[Informe completo (PDF, 19 páginas)](informe/informe_final.pdf)** · [versión Markdown](informe/informe_final.md)

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

- **Transfer learning explicado y medido.** Qué significa congelar capas, cómo se aplica
  en Ultralytics, y cuántos parámetros queda entrenando cada estrategia — verificado
  cargando el modelo real, no citando valores de memoria.
- **Auditoría del propio pipeline.** El informe documenta **tres defectos metodológicos
  propios** (§10) en lugar de omitirlos. Los tres comparten un patrón: son **fallas
  silenciosas** que producen números plausibles y atraviesan la ejecución completa sin
  lanzar un solo error.

Esa es la conclusión transversal del trabajo: **en visión por computadora el riesgo
principal no es el código que falla, sino el que funciona sobre premisas equivocadas.**

---

## 2. Construcción del pipeline

El pipeline tiene **12 etapas** (más una 4B opcional), pensadas para ejecutarse en orden
en Colab. Cada etapa es una celda del notebook.

| Etapa | Qué hace | Tiempo en T4 |
|---|---|---|
| 1 | Entorno, verificación de GPU y descarga automática del dataset | ~5 min |
| 2 | EDA + **verificación visual del mapeo de clases** ⚠️ | ~3 min |
| 3 | Split 70/15/15 estratificado + conversión a formato YOLO | ~3 min |
| 4 | Entrenamiento (*fine-tuning* desde pesos COCO) | 20–30 min |
| **4B** | **Experimento: capas congeladas vs. no congeladas** (opcional) | 60–90 min |
| 5 | Evaluación en test + barrido del umbral óptimo en validación | ~10 min |
| 6 | Gráficos: pérdidas, mAP por época, F1 vs. umbral, curva PR | ~3 min |
| 7 | Comparativa visual *ground truth* vs. predicción | ~2 min |
| 8 | Inferencia sobre las ~10.200 imágenes sin etiquetar | ~3 min |
| 9 | Exportación a ONNX + *trade-off* medido | ~3 min |
| 10 | Informe final con los números reales | ~1 min |
| 11 | Empaquetado y descarga del ZIP | ~1 min |
| 12 | Demo: subir una imagen y detectar | opcional |

**Total: ~1 h sin la Etapa 4B, ~2,5 h con ella.**

### Decisiones de diseño

**Modelo: YOLOv5n (variante *nano*).** El más chico de la familia: 2.51 M parámetros. Es
el único que permite entrenar e inferir con holgura en el tier gratuito de Colab. Se parte
de pesos pre-entrenados en COCO.

**Una sola clase (`nc=1`).** El dataset también anota autos, pero se descartan. No por
irrelevantes: incluirlos obligaría al modelo a resolver un problema adicional de
discriminación entre clases sin beneficio para el objetivo.

**Resolución 320 px.** Límite impuesto por la memoria y el tiempo de la T4. Es, con
diferencia, **la limitación de mayor impacto** del trabajo: explica la brecha entre
mAP@0.5 (0.78) y mAP@0.75 (0.40).

**El umbral de confianza se elige en validación, nunca en test.** Elegirlo en test sería
*data leakage*: las métricas reportadas dejarían de estimar el rendimiento en datos no
vistos.

**El modelo real es *anchor-free*.** Se carga `yolov5nu.pt`, no `yolov5n.pt`. La `u`
indica *backbone* de YOLOv5 con la **cabeza anchor-free desacoplada de YOLOv8** (con
pérdida DFL). Se verifica en los datos: `results.csv` tiene columna `train/dfl_loss`, que
el YOLOv5 original no produce. Esto corrige una afirmación incorrecta de la documentación
inicial del proyecto.

### La pausa obligatoria de la Etapa 2

La Etapa 2 **requiere intervención humana**: hay que mirar una grilla de ejemplos y
confirmar que la clase `"2"` del XML corresponde efectivamente a las piletas.

No es un paso decorativo. Con el mapeo invertido, el pipeline entrena **sin un solo error**
sobre **cero** instancias de la clase objetivo y reporta mAP = 0. Ya ocurrió una vez
durante el desarrollo, y es el origen de la conclusión metodológica del informe.

---

## 3. Resultados

### Métricas en test (1.125 imágenes, 232 piletas reales)

| Métrica | Valor | Qué significa |
|---|---|---|
| mAP@0.5 | **0.7846** | Calidad del *ranking* con criterio de localización laxo |
| mAP@[.5:.95] | 0.4203 | Promedio sobre 10 umbrales de IoU: métrica estándar de COCO |
| mAP@0.75 | 0.4003 | Con criterio estricto: **cae a la mitad** |
| mAR@100 | 0.5173 | Techo de *recall* permitiendo 100 detecciones por imagen |
| Precisión | 0.8912 | De cada 100 avisos, ~89 son piletas reales |
| Recall | 0.7414 | Encuentra ~74 de cada 100 piletas |
| F1 | 0.8094 | Media armónica de las dos anteriores |
| IoU promedio | 0.7868 | Calidad de las cajas acertadas |
| Umbral óptimo | 0.25 | Maximiza F1 **en validación** |

**Matriz de confusión** (reconstruida algebraicamente a partir de P y R): TP = 172,
FP = 21, FN = 60.

### Las dos lecturas que importan

**El modelo clasifica bien pero localiza con imprecisión.** La brecha entre mAP@0.5 (0.78)
y mAP@0.75 (0.40) es el hallazgo más informativo: el problema no es *reconocer* piletas
sino *delimitarlas* con precisión. Es la consecuencia esperable de entrenar a 320 px.

**Es un modelo conservador.** 60 falsos negativos contra 21 falsos positivos: confiable
cuando avisa, pero omite una de cada cuatro piletas. Para fiscalización, esa asimetría
es probablemente la indeseada —conviene bajar el umbral y ganar *recall* a costa de
precisión, que es el recurso abundante.

### Transfer learning: los números verificados

Valores obtenidos **cargando `yolov5nu.pt` y ejecutando el *trainer***, no citados de
memoria. El modelo tiene **25 módulos (0–24)** y `Detect` es el **24**:

| `freeze` | Entrenables | % | Qué queda libre |
|---|---|---|---|
| 0 | 2.636 M | 100 % | Toda la red (**el usado en este TP**) |
| 10 | 1.589 M | 60.3 % | Neck + head |
| 24 | 0.878 M | 33.3 % | Solo la cabeza `Detect` |
| 25 | 0 M | 0 % | Ultralytics aborta con `RuntimeError` |

Dos detalles que surgieron al verificarlo:

1. **`freeze=22` no es "solo la cabeza"**, aunque sea el valor que suele citarse: deja los
   módulos 23 *y* 24 entrenables (45 %). El valor correcto para *feature extraction* en
   esta arquitectura es **24**.
2. **El conteo no puede hacerse sobre el modelo recién cargado.** `yolov5nu.pt` viene con
   `nc=80` y Ultralytics reconstruye la cabeza para `nc=1` al entrenar; contar antes
   sobreestima en 19.200 parámetros. Hay que medirlo en un callback `on_train_start`.

> ⚠️ **Estado del experimento.** La **mecánica** está medida y verificada, pero el **mAP
> comparativo de cada estrategia todavía no**: la Etapa 4B está implementada y probada
> *end-to-end*, pendiente de ejecución en GPU. El informe declara explícitamente esta
> brecha (§5.6) en lugar de rellenarla con cifras de literatura.

### Los tres defectos auditados

Documentados en §10 del informe. **Los tres están corregidos en el código**, pero las
métricas de arriba provienen de la corrida original, **anterior** a las correcciones.

| # | Defecto | Impacto | Corrección |
|---|---|---|---|
| 1 | El split real era **35/35/30**, no 70/15/15 | Se entrenó con **la mitad** de los datos disponibles | Partir el 30 %, no el 70 %, + `assert` de proporciones |
| 2 | El *delta* de mAP del ONNX (+0.0864) **no era comparable** | Medía dos conjuntos distintos, no el costo del *export* | Ambos modelos sobre el mismo subconjunto de test |
| 3 | El barrido de umbrales arrancaba en **0.25** y el óptimo dio **0.25** | Un óptimo en el borde del rango no es un óptimo | Rango desde 0.05 + aviso si vuelve a caer en un extremo |

**Implicación positiva del defecto 1:** las métricas reportadas son **conservadoras**. Se
alcanzó mAP@0.5 = 0.78 entrenando con 1.311 imágenes en lugar de ~2.600.

Las correcciones no se limitan a arreglar cada caso: agregan *asserts*, avisos y registro
explícito del conjunto usado en cada comparación, para que el próximo error del mismo tipo
**haga ruido** en vez de llegar hasta el informe.

---

## 4. Estructura del repositorio

```
.
├── notebooks/
│   └── deteccion_piletas.ipynb        # Notebook principal (14 celdas, listo para Colab)
├── src/
│   └── deteccion_piletas.py           # Mismo pipeline como script
├── informe/
│   ├── informe_final.pdf              # Informe de 19 páginas (entregable)
│   ├── informe_final.md               # Fuente en Markdown
│   ├── build_pdf.py                   # Genera el PDF (Chrome headless)
│   └── img/                           # Figuras del informe
├── docs/
│   ├── COMO_CORRER_EN_COLAB.md        # Guía de ejecución paso a paso
│   └── guia_12_etapas.md              # Detalle de cada etapa, con los bugs documentados
└── resultados_corrida_original/
    ├── metricas.csv                   # Las 16 métricas reales
    ├── trade_off_onnx.csv             # Comparación .pt vs .onnx (ver defecto 2)
    └── modelos/data.yaml              # Configuración del dataset
```

**Qué no está en el repo y por qué:**

- **Los pesos** (`best.pt`, `best.onnx`, ~15 MB): se regeneran corriendo el pipeline.
- **El dataset**: se descarga automáticamente desde Kaggle en la Etapa 1.
- **`runs/` y `resultados/`**: artefactos de ejecución, ignorados por `.gitignore`.
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

### Entorno

`ultralytics`, `torch`, `torchmetrics`, `kagglehub`, `albumentations`, `thop`, `onnx`,
`onnxruntime`. La Etapa 1 los instala sola.

> **Sobre entrenar localmente:** no es viable sin GPU CUDA. Un detector en CPU tarda horas
> por época, así que el script corta a propósito en la Etapa 1 en lugar de quemar tiempo.
> Para *inferencia* suelta, la CPU sí alcanza (~27 FPS).

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

1. **Reejecutar el pipeline corregido.** Las tres correcciones están aplicadas pero no
   reentrenadas: requieren GPU. Una sola corrida resuelve los tres defectos.
2. **Ejecutar la Etapa 4B** (ya implementada): convertiría la sección de capas congeladas
   de exposición teórica en resultado medido. ~60–90 min en T4.
3. **Entrenar a `imgsz=640`:** vía directa para atacar el bajo mAP@0.75 y los falsos
   negativos en objetos pequeños.
4. **Ampliar el presupuesto de épocas:** el *early stopping* no se activó en 25 épocas, lo
   que indica que el modelo seguía mejorando cuando se agotó el presupuesto.

---

*Trabajo académico — Maestría en Inteligencia Artificial, Universidad de Palermo.*
