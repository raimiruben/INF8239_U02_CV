\# Model Card · Fashion-MNIST CNN



\## Modelo y versión



\- \*\*Modelo evaluado:\*\* `small\_cnn` (guardado en `models/best\_cnn.keras`), comparado con un baseline `dense\_baseline` que no se guarda.

\- \*\*Proyecto:\*\* inf8239-u02-cv 0.1.0 (INF-8239, U02.LAB07 · Ejercicio 04). \*\*Autor:\*\* Raimi De Jesús. \*\*Corrida de referencia:\*\* 2026-10-08.

\- \*\*Entorno:\*\* Windows 11, Python 3.12.14, TensorFlow 2.21.0, Keras 3.15.1.

\- \*\*Arquitectura CNN:\*\* Conv2D(32) → MaxPooling → Conv2D(64) → MaxPooling → GlobalAveragePooling2D → Dropout(0,25) → Dense(10, softmax).

\- \*\*Arquitectura del baseline:\*\* Flatten → Dense(64, ReLU) → Dense(10, softmax).

\- \*\*Entrenamiento común:\*\* Adam, entropía cruzada categórica dispersa, lote de 128, 8 épocas, EarlyStopping (paciencia 2, restaura los mejores pesos), semilla 42 y `TF\_DETERMINISTIC\_OPS=1`.

\- \*\*Reproducibilidad:\*\* con esa semilla y operaciones deterministas, reentrenar el denso reprodujo el F1 macro exactamente en este equipo. No se garantiza igualdad binaria entre plataformas.



\## Uso previsto



Ejercicio académico para comparar de forma reproducible un baseline denso y una CNN pequeña sobre la misma partición, y practicar evaluación por clase, análisis de errores, medición de costo y documentación.



\## Usos fuera de alcance



No debe usarse para clasificar fotografías reales de prendas, ni como evidencia de preparación para imágenes médicas, OCR, vigilancia, reconocimiento de personas u otros dominios. Un buen resultado en este benchmark no se transfiere a esos casos.



\## Dataset y particiones



\- \*\*Dataset:\*\* Fashion-MNIST, con 60 000 imágenes de entrenamiento y 10 000 de prueba, en escala de grises, 28×28 y 10 clases. Se descarga en la caché de TensorFlow.

\- \*\*Particiones:\*\* 54 000 para entrenar y 6 000 para validar (las últimas 6 000 del conjunto de entrenamiento, sin barajar); 10 000 de prueba, con 1 000 por clase. Ambos modelos usan exactamente la misma partición.

\- \*\*Etiquetas:\*\* 0 camiseta, 1 pantalón, 2 suéter, 3 vestido, 4 abrigo, 5 sandalia, 6 camisa, 7 zapatilla, 8 bolso, 9 botín (orden estándar del dataset; el script solo muestra los números).



\## Preprocesamiento



Normalización de los píxeles con `normalize\_images` y validación de forma y rango con `validate\_images` (`src/inf8239\_u02\_cv/data.py`). Las pruebas (`uv run pytest -q`) verifican forma y rango de las imágenes, número de clases y salida del modelo.



\## Métricas globales y por clase



Conjunto de prueba (10 000 imágenes):



| | Denso | CNN |

|---|---|---|

| F1 macro | 0,865 | 0,789 |

| Exactitud | 0,866 | 0,793 |



| Clase | Recall denso | Recall CNN | F1 denso | F1 CNN |

|---|---|---|---|---|

| 0 camiseta | 0,848 | 0,806 | 0,819 | 0,715 |

| 1 pantalón | 0,969 | 0,931 | 0,975 | 0,959 |

| 2 suéter | 0,684 | 0,643 | 0,752 | 0,674 |

| 3 vestido | 0,903 | 0,843 | 0,869 | 0,796 |

| 4 abrigo | 0,870 | 0,664 | 0,782 | 0,662 |

| 5 sandalia | 0,962 | 0,884 | 0,948 | 0,916 |

| 6 camisa | 0,597 | 0,373 | 0,652 | 0,427 |

| 7 zapatilla | 0,940 | 0,953 | 0,934 | 0,901 |

| 8 bolso | 0,953 | 0,937 | 0,963 | 0,926 |

| 9 botín | 0,937 | 0,896 | 0,952 | 0,915 |



La clase más difícil para ambos modelos es la camisa. En la CNN, 274 de 1 000 camisas se predicen como camiseta y el resto se reparte entre abrigo, suéter, vestido y bolso. El triángulo suéter–abrigo–camisa concentra las mayores confusiones.



\## Comparación de costo



\- \*\*Hardware:\*\* AMD Ryzen 7 3700U (4 núcleos, 8 hilos), 5,9 GB de RAM, solo CPU (TensorFlow ≥ 2.11 no usa GPU en Windows nativo).

\- \*\*Parámetros:\*\* denso 50 890; CNN 19 466.

\- \*\*Tiempo de entrenamiento:\*\* denso 13,0 s; CNN 187,6 s (14,4×). El script completo tardó unos 4 min 36 s.

\- \*\*Tiempo de inferencia:\*\* denso 0,0546 ms/imagen; CNN 0,1522 ms/imagen (2,8×). Mediana de 7 repeticiones tras calentamiento, sobre las 10 000 imágenes de prueba.

\- \*\*Tamaño del archivo (sin optimizador):\*\* denso 220 815 B; CNN 105 347 B.



\## Limitaciones y riesgos



\- \*\*Benchmark didáctico:\*\* imágenes 28×28 de prendas con fondo uniforme; no representan fotos reales ni otros dominios.

\- \*\*La CNN rindió menos que el baseline\*\* (−0,076 de F1 macro) y cuesta más en entrenamiento e inferencia, aunque su archivo es más pequeño. Sus pérdidas seguían bajando a las 8 épocas, y su diseño (promediado global más dropout) puede limitar su capacidad. No se verificó cuál de las dos causas pesa más.

\- \*\*Evidencia limitada:\*\* una sola partición, una sola semilla y sin intervalos de incertidumbre. Algunas etiquetas del conjunto son ambiguas o discutibles a esta resolución.



\## Supervisión y monitoreo



Cualquier uso fuera del ejercicio exige validar con datos del dominio real, revisar a mano las predicciones de camisa, abrigo y suéter, y no tomar decisiones automáticas con base en estas métricas. Si se desplegara, habría que monitorear el recall por clase y no solo la exactitud global, y reevaluar cuando cambie la distribución de las imágenes de entrada.

