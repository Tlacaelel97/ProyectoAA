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


def run_eda_stats(df: pd.DataFrame, num_classes=64) -> tuple[np.ndarray, np.ndarray]:
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


def analyze_imbalance(
    pixel_counts, mapping_csv="data/goose2D/goose_2d_train/goose_label_mapping.csv"
) -> pd.DataFrame:
    # Cargar mapeo y preparar datos
    mapping = pd.read_csv(mapping_csv)
    df = pd.DataFrame(
        {"label_key": np.arange(len(pixel_counts)), "pixel_count": pixel_counts}
    )
    df = pd.merge(mapping, df, on="label_key")

    # Calcular Frecuencias
    total_pixels = df["pixel_count"].sum()
    df["frequency"] = df["pixel_count"] / total_pixels

    # Median Frequency Balancing (MFB)
    # Solo consideramos clases con presencia > 0 para no sesgar la mediana
    present_df = df[df["pixel_count"] > 0].copy()
    median_freq = present_df["frequency"].median()

    # El peso es Mediana / Frecuencia de la clase
    df["weight"] = 0.0  # Default para clases ausentes
    df.loc[df["pixel_count"] > 0, "weight"] = median_freq / df["frequency"]

    # Normalización de Pesos
    active_weights = df.loc[df["weight"] > 0, "weight"]
    df.loc[df["weight"] > 0, "weight"] = df["weight"] / active_weights.mean()

    # Identificar Clases Críticas
    print("\n--- RESUMEN ESTRATÉGICO DE DESBALANCEO ---")
    print(
        f"Ratio de Desbalanceo (Max/Min): {df['pixel_count'].max() / present_df['pixel_count'].min():.2e}"
    )

    print("\nClases Dominantes (VQ-VAE tenderá a colapsar aquí):")
    print(
        df.sort_values(by="pixel_count", ascending=False)[
            ["class_name", "frequency"]
        ].head(5)
    )

    print("\nClases Raras (Requieren Pesos Altos):")
    print(
        df[df["pixel_count"] > 0]
        .sort_values(by="pixel_count")[["class_name", "weight"]]
        .head(5)
    )

    # Guardar pesos
    weights_dict = df.set_index("label_key")["weight"].to_dict()
    import json

    with open("class_weights.json", "w") as f:
        json.dump(weights_dict, f)

    return df


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

        # Extraemos el Frame ID (los 4 dígitos entre '__' y '_')
        # Ejemplo: 2022-09-14_garching_uebungsplatz__0000_... -> 0000
        try:
            frame_id = lp.name.split("__")[1].split("_")[0]
        except (IndexError, AttributeError):
            continue

        img_folder = pathlib.Path(images_base_path) / sequence_name

        search_pattern = f"*_{frame_id}_*vis.png"
        rgb_candidates = list(img_folder.glob(search_pattern))

        if not rgb_candidates:
            print(
                f"Error: No se encontró imagen vis para frame {frame_id} en {img_folder}"
            )
            continue

        # Seleccionamos el primer candidato
        rp = rgb_candidates[0]

        img = cv2.cvtColor(cv2.imread(str(rp)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(lp), cv2.IMREAD_GRAYSCALE)

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Imagen RGB (Sensor: Windshield)\n{rp.name}")
        axes[i, 1].imshow(mask, cmap="nipy_spectral", vmin=0, vmax=63)
        axes[i, 1].set_title(f"Máscara Fine-Grained (64 clases)\n{lp.name}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    LABELS_BASE_PATH = "data/goose2D/goose_2d_train/labels/train"
    IMAGES_BASE_PATH = "data/goose2D/goose_2d_train/images/train"
    visual_inspection(LABELS_BASE_PATH, IMAGES_BASE_PATH, num_samples=3)
