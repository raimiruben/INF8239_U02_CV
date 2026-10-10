# INF-8239 · Unidad 02 · Proyecto de visión

Autor académico: Edwin Ramón José Nolasco

## CPU

```bash
uv python install 3.12
uv sync --extra cpu
uv run python scripts/check\\\\\\\_runtime.py
uv run python scripts/train\\\\\\\_cv.py --epochs 8
```

## GPU

La ruta GPU se utiliza solamente en WSL2/Linux con NVIDIA correctamente configurada:

```bash
uv sync --extra gpu
```

Complete `MODEL\\\\\\\_CARD.md` después de analizar métricas, errores y costo.





\## Resultados



Misma partición (54 000 entrenamiento / 6 000 validación / 10 000 prueba), semilla 42, 8 épocas, CPU. Detalle por clase, hardware y limitaciones en \[`MODEL\\\\\\\_CARD.md`](MODEL\_CARD.md).



| | Denso | CNN |

|---|---|---|

| F1 macro | 0,865 | 0,789 |

| Exactitud | 0,866 | 0,793 |

| Parámetros | 50 890 | 19 466 |

| Entrenamiento | 13,0 s | 187,6 s |

| Inferencia (mediana de 7) | 0,0546 ms/imagen | 0,1522 ms/imagen |

| Archivo (sin optimizador) | 220 815 B | 105 347 B |



Artefactos: `reports/cv\\\\\\\_metrics.json`, `reports/confusion\\\\\\\_cnn.png`, `reports/cnn\\\\\\\_errors.png` y `models/best\\\\\\\_cnn.keras`.



\## Errores comentados



`reports/cnn\\\\\\\_errors.png` muestra los \*\*primeros 16 errores\*\* de la CNN en el conjunto de prueba (R = clase real, P = predicha; filas y columnas de arriba abajo y de izquierda a derecha). No son una muestra aleatoria: ilustran patrones, y la frecuencia real está en `reports/confusion\\\\\\\_cnn.png`.



\- \*\*Abrigo predicho como suéter (R:4 P:2, 4 casos):\*\* fila 1 columnas 2 a 4, y fila 2 columna 3. Son chaquetas y cárdigan oscuros o estampados de manga larga; a 28×28 píxeles se pierden el cierre y las solapas, y queda la silueta de un suéter.

\- \*\*Camisa predicha como abrigo (R:6 P:4, 2 casos) y suéter predicho como abrigo (R:2 P:4, 1 caso):\*\* fila 2 columna 4, fila 4 columnas 3 y 4. Camisas de manga larga y tejidos texturizados que se parecen a una chaqueta fina.

\- \*\*Camisa predicha como camiseta (R:6 P:0):\*\* fila 3 columna 4. Camisa de manga corta cuyo único rasgo distintivo (botones y cuello) apenas se ve.

\- \*\*Vestido y camiseta (R:3 P:0 en tres casos, R:0 P:3 en uno):\*\* fila 3 columnas 1 a 3 y fila 4 columna 1. La longitud decide entre ambas clases y a esta resolución es ambigua: vestidos cortos o estampados salen como camiseta, y una prenda larga y lisa etiquetada como camiseta sale como vestido.

\- \*\*Calzado (4 casos):\*\* botín → zapatilla, sandalia → zapatilla, botín → sandalia y zapatilla → botín (fila 1 col. 1, fila 2 cols. 1 y 2, fila 4 col. 2). Las zapatillas altas y los botines bajos se parecen. En la fila 2 columna 2 el zapato abierto con correa parece más una sandalia que el botín de su etiqueta, lo que sugiere una etiqueta discutible.



Conclusión: la mitad de los errores mostrados (8 de 16) ocurren entre prendas superiores de manga larga, el mismo triángulo suéter–abrigo–camisa que domina la matriz de confusión.



\## Scripts y reproducibilidad



\- `scripts/bench\\\\\\\_inference.py`: mide el tiempo de inferencia (mediana de 7 repeticiones tras calentamiento) y el tamaño de ambos modelos sin optimizador.

\- `scripts/dense\\\\\\\_per\\\\\\\_class.py`: reentrena solo el baseline denso con la misma partición y semilla para obtener su reporte por clase. Reproduce exactamente el F1 macro de `cv\\\\\\\_metrics.json` en este equipo.



```bash

uv run python scripts/bench\\\\\\\_inference.py

uv run python scripts/dense\\\\\\\_per\\\\\\\_class.py

uv run pytest -q

```



Entorno de la corrida: Windows 11, Python 3.12.14, TensorFlow 2.21.0, solo CPU. No se garantiza igualdad binaria entre plataformas.







\## Cierre interpretativo — LAB07: visión computacional con Fashion-MNIST



\*\*Resultado principal.\*\* Con la misma partición (54 000 / 6 000 / 10 000), semilla 42 y 8 épocas, el baseline denso superó a la CNN pequeña: F1 macro 0,865 frente a 0,789.



\*\*Evidencia predictiva.\*\* Exactitud 0,866 frente a 0,793. El denso fue mejor en F1 en las 10 clases, con una brecha que va de 0,016 en pantalón a 0,225 en camisa. Las pérdidas de la CNN seguían bajando en la época 8 (pérdida de validación 0,587) y el EarlyStopping no se activó en ninguno de los dos modelos.



\*\*Clase más difícil.\*\* La camisa (clase 6) en ambos modelos: F1 0,652 en el denso y 0,427 en la CNN (recall 0,373). En la CNN, 274 de 1 000 camisas se predicen como camiseta, y el triángulo suéter–abrigo–camisa concentra las mayores confusiones. Los errores visuales muestran prendas de manga larga que, a 28×28 píxeles, pierden los detalles de cierre, cuello y botones.



\*\*Costo comparado.\*\* La CNN tiene menos parámetros (19 466 frente a 50 890) y un archivo de menos de la mitad de tamaño (105 KB frente a 221 KB, sin optimizador), pero entrena 14,4× más lento (187,6 s frente a 13,0 s) e infiere 2,8× más lento (0,152 frente a 0,055 ms por imagen) en CPU (Ryzen 7 3700U). Los parámetros no miden el costo: las convoluciones exigen muchas más operaciones por imagen.



\*\*¿La mejora justifica el costo?\*\* No en este experimento: no hay mejora, porque la CNN es peor y más costosa en cómputo. La conclusión vale para esta configuración de 8 épocas y esta arquitectura, no para las CNN en general.



\*\*Limitación del benchmark.\*\* Fashion-MNIST es un benchmark didáctico de imágenes 28×28 con fondo uniforme. Se usó una sola partición y una sola semilla, sin intervalos de incertidumbre, y nada garantiza que el resultado se transfiera a fotografías reales ni a otros dominios.



\*\*Decisión antes de usar otro dominio.\*\* No trasladar ninguno de los dos modelos tal cual. Validar con datos reales del dominio y por clase, entrenar la CNN hasta que converja antes de volver a compararla, y definir qué errores son críticos y quién los revisa. Siguiente experimento: repetir con más épocas para distinguir si la CNN no convergió o si su arquitectura limita su capacidad.







\## Evidencia visual



!\[Curvas de entrenamiento y validación](reports/training\_curves.png)



!\[Matriz de confusión de la CNN](reports/confusion\_cnn.png)



!\[Primeros 16 errores de la CNN](reports/cnn\_errors.png)



!\[Distribución de clases por partición](reports/audit\_class\_distribution.png)



\## Ejecución completa (orden recomendado)



```bash

uv python install 3.12

uv sync --extra cpu

uv run pytest -q                              # 3 pruebas

uv run python scripts/audit\_cv.py             # auditoría y huellas de las particiones

uv run python scripts/train\_cv.py --epochs 8  # baseline denso y CNN: métricas, matriz, errores y modelo

uv run python scripts/plot\_curves.py          # curvas de entrenamiento y validación

uv run python scripts/bench\_inference.py      # inferencia (mediana de 7) y tamaño de los modelos

uv run python scripts/dense\_per\_class.py      # reporte por clase del baseline denso

```



| Script | Archivos que genera |

|---|---|

| `scripts/audit\_cv.py` | `reports/audit\_cv.json`, `reports/audit\_class\_distribution.png` |

| `scripts/train\_cv.py` | `reports/cv\_metrics.json`, `reports/confusion\_cnn.png`, `reports/cnn\_errors.png`, `models/best\_cnn.keras` |

| `scripts/plot\_curves.py` | `reports/training\_curves.png`, `reports/history.json` |

| `scripts/bench\_inference.py` y `scripts/dense\_per\_class.py` | Solo imprimen en consola |



El `.gitignore` excluye `models/\*.keras`, `reports/\*.png` y `reports/\*.json`. Los resultados de referencia se versionaron con `git add -f`, así que volver a ejecutar los scripts los deja modificados en `git status`. Los tiempos absolutos varían entre ejecuciones (dependen de la carga de la CPU); con semilla 42 y operaciones deterministas, las métricas se reproducen exactamente en este equipo.



\## Alternativa con pip



Requiere Python 3.12. `requirements.txt` se generó desde `uv.lock` con `uv export --no-hashes --no-emit-project --extra cpu -o requirements.txt`.



```powershell

python -m venv .venv

.venv\\Scripts\\Activate.ps1

pip install -r requirements.txt

pytest -q

$env:PYTHONPATH = "src"

python scripts/train\_cv.py --epochs 8

```



En Linux o macOS: `source .venv/bin/activate` y `PYTHONPATH=src python scripts/train\_cv.py --epochs 8`. En Colab: `%pip install -r requirements-colab.txt` y ejecutar los scripts con `PYTHONPATH=src`.



\## Notebook ejecutado



`notebooks/ejercicio04.ipynb` reúne en una sola ejecución la auditoría (con comprobación de las huellas), el entrenamiento de ambos modelos con la misma partición y semilla, las curvas, las métricas por clase, las matrices de confusión, los errores, el costo, las pruebas y la decisión técnica. Se entrega ya ejecutado, y para reejecutarlo:



```bash

uv run --with nbconvert --with ipykernel python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 notebooks/ejercicio04.ipynb

```



Tarda unos 7 minutos en CPU y no sobrescribe `reports/` ni `models/`. La diferencia del F1 macro frente a `reports/cv\_metrics.json` es 0 en ambos modelos.

