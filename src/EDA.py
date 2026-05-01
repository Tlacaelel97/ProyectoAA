# EDA.py
import json
import pathlib
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
import random
import matplotlib.pyplot as plt


def scan_labels(base_path) -> pd.DataFrame:
    """
    Escanea el directorio base para encontrar todas las máscaras de etiquetas (labelids) y extrae metadatos relevantes.
    Argumentos:
        - base_path: Ruta al directorio que contiene las máscaras de etiquetas (labelids).
    Retorna:
        - DataFrame con columnas: 'path' (ruta completa al archivo), 'sequence' (nombre de la secuencia).
    """
    label_paths = list(pathlib.Path(base_path).rglob("*_labelids.png"))
    data = []
    for p in label_paths:
        # Extraer metadatos del nombre del directorio superior
        sequence_name = p.parent.name
        data.append({"path": str(p), "sequence": sequence_name, "filename": p.name})

    json.dump(data, open("data_summary.json", "w"), indent=4)
    return pd.DataFrame(data)


def run_eda_stats(df, num_classes=64) -> tuple[np.ndarray, np.ndarray]:
    """
    Realiza un análisis exploratorio de datos (EDA) para entender la distribución de etiquetas en las máscaras.
    Argumentos:
        - df: DataFrame que contiene las rutas a las máscaras de etiquetas.
        - num_classes: Número total de clases (etiquetas) esperadas en las máscaras.
    Retorna:
        - pixel_counts: Array con el conteo total de píxeles por clase.
        - image_presence: Array con el conteo de imágenes que contienen cada clase.
    """
    # Inicializar contadores
    pixel_counts = np.zeros(num_classes, dtype=np.int64)
    image_presence = np.zeros(num_classes, dtype=np.int64)

    print("Iniciando análisis de distribución de etiquetas...")
    for _, row in tqdm(df.iterrows(), total=len(df)):

        # Leer máscara (label)
        mask = cv2.imread(row["path"], cv2.IMREAD_GRAYSCALE)

        if mask is None:
            continue
        # 1. Conteo de píxeles
        counts = np.bincount(mask.flatten(), minlength=num_classes)
        pixel_counts += counts[:num_classes]

        # 2. Presencia en imagen (Image-wise)
        present_classes = np.unique(mask)
        for c in present_classes:
            if c < num_classes:
                image_presence[c] += 1

    return pixel_counts, image_presence


def visual_inspection(labels_base_path, images_base_path, num_samples=3) -> None:
    """
    Realiza una inspección visual de un número aleatorio de muestras para verificar la correspondencia entre las máscaras de etiquetas y las imágenes RGB.
    Argumentos:
        - labels_base_path: Ruta al directorio que contiene las máscaras de etiquetas (labelids).
        - images_base_path: Ruta al directorio que contiene las imágenes RGB.
        - num_samples: Número de muestras aleatorias a visualizar.
    """
    label_files = list(pathlib.Path(labels_base_path).rglob("*_labelids.png"))
    samples = random.sample(label_files, num_samples)

    fig, axes = plt.subplots(num_samples, 2, figsize=(20, 5 * num_samples))

    for i, lp in enumerate(samples):
        sequence_name = lp.parent.name
        # Reemplazamos el sufijo de la etiqueta por el de la imagen visible
        base_name = lp.name.replace("_labelids.png", "_vis.png")
        rp = pathlib.Path(images_base_path) / sequence_name / base_name

        if not rp.exists():
            print(f"Error: No se encontró {rp.name} en {rp.parent}")
            continue

        img_vis = cv2.cvtColor(cv2.imread(str(rp)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(lp), cv2.IMREAD_GRAYSCALE)

        axes[i, 0].imshow(img_vis)
        axes[i, 0].set_title(f"RGB (Visible): {sequence_name}")
        axes[i, 1].imshow(mask, cmap="nipy_spectral", vmin=0, vmax=63)
        axes[i, 1].set_title("Máscara de Grano Fino")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    base_path = "data/goose2D/goose_2d_train/labels/train"
    df = scan_labels(base_path)
    print(df.head())

    pixel_stats, presence_stats = run_eda_stats(df)
    print("Distribución de píxeles por clase:")
    for i, count in enumerate(pixel_stats):
        print(f"Clase {i}: {count} píxeles")
