# EDA.py
import json
import pathlib
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm


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


if __name__ == "__main__":
    base_path = "data/goose2D/goose_2d_train/labels/train"
    df = scan_labels(base_path)
    print(df.head())

    pixel_stats, presence_stats = run_eda_stats(df)
    print("Distribución de píxeles por clase:")
    for i, count in enumerate(pixel_stats):
        print(f"Clase {i}: {count} píxeles")
