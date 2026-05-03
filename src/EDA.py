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


def analyze_geometry(images_base_path) -> pd.DataFrame:
    """
    Analiza las resoluciones y Aspect Ratios de todas las secuencias.
    Ayuda a definir la resolución de entrada fija para el VQ-VAE.
    """
    img_root = pathlib.Path(images_base_path)
    resolutions = []

    print("Escaneando geometría de sensores...")
    # Iteramos por las carpetas de secuencias (vuelos)
    for seq_dir in tqdm(list(img_root.iterdir())):
        if not seq_dir.is_dir():
            continue

        # Tomamos una muestra para no saturar (el sensor es constante por secuencia)
        sample_file = next(seq_dir.glob("*_vis.png"), None)
        if sample_file:
            img = cv2.imread(str(sample_file))
            if img is not None:
                h, w = img.shape[:2]
                resolutions.append(
                    {
                        "sequence": seq_dir.name,
                        "width": w,
                        "height": h,
                        "aspect_ratio": round(w / h, 2),
                    }
                )

    df_geom = pd.DataFrame(resolutions)
    summary = (
        df_geom.groupby(["width", "height", "aspect_ratio"])
        .size()
        .reset_index(name="count")
    )

    print("\n--- REPORTE DE GEOMETRÍA ---")
    print(summary)
    return df_geom


def triple_visual_inspection(
    labels_base_path: str, images_base_path: str, num_samples=3
) -> None:
    """
    Visualización Side-by-Side: RGB, NIR y Máscara Fine-Grained.
    Valida la calidad de las etiquetas y la utilidad del infrarrojo.
    """
    label_root = pathlib.Path(labels_base_path)
    image_root = pathlib.Path(images_base_path)
    label_files = list(label_root.rglob("*_labelids.png"))
    samples = random.sample(label_files, num_samples)

    fig, axes = plt.subplots(num_samples, 3, figsize=(22, 5 * num_samples))

    for i, lp in enumerate(samples):
        seq_name = lp.parent.name
        # Extraer Frame ID de forma robusta
        frame_id = lp.name.split("__")[1].split("_")[0]
        img_folder = image_root / seq_name

        # Localizar RGB y NIR
        vis_p = next(img_folder.glob(f"*_{frame_id}_*vis.png"), None)
        nir_p = next(img_folder.glob(f"*_{frame_id}_*nir.png"), None)

        if not vis_p or not nir_p:
            continue

        # Lectura
        img_vis = cv2.cvtColor(cv2.imread(str(vis_p)), cv2.COLOR_BGR2RGB)
        img_nir = cv2.imread(str(nir_p), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(lp), cv2.IMREAD_GRAYSCALE)

        axes[i, 0].imshow(img_vis)
        axes[i, 0].set_title(f"RGB: {seq_name}")
        axes[i, 1].imshow(img_nir, cmap="gray")
        axes[i, 1].set_title(f"NIR (Infrarrojo)")
        axes[i, 2].imshow(mask, cmap="nipy_spectral", vmin=0, vmax=63)
        axes[i, 2].set_title(f"Fine-Grained Mask: {frame_id}")

        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    plt.show()


def calculate_rgb_normalization_stats(images_base_path, sample_size=1000) -> tuple:
    """
    Calcula Media y Std para la normalización en PyTorch.
    Usa una submuestra representativa para manejar los 22.5 GB de datos.
    """
    img_root = pathlib.Path(images_base_path)
    all_vis_files = list(img_root.rglob("*_vis.png"))

    selected_files = random.sample(all_vis_files, min(sample_size, len(all_vis_files)))

    pixel_sum = np.zeros(3)
    pixel_sq_sum = np.zeros(3)
    count = 0

    print(f"Calculando estadísticas sobre {len(selected_files)} imágenes...")
    for fp in tqdm(selected_files):
        img = cv2.imread(str(fp))
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        pixel_sum += np.mean(img, axis=(0, 1))
        pixel_sq_sum += np.mean(img**2, axis=(0, 1))
        count += 1

    mean = pixel_sum / count
    std = np.sqrt((pixel_sq_sum / count) - mean**2)

    print("\n--- PARÁMETROS DE NORMALIZACIÓN ---")
    print(f"Mean (R, G, B): {mean}")
    print(f"Std  (R, G, B): {std}")
    return mean, std


if __name__ == "__main__":
    LABELS_PATH = "data/goose2D/goose_2d_train/labels/train"
    IMAGES_PATH = "data/goose2D/goose_2d_train/images/train"

    # 1. Análisis de Geometría
    analyze_geometry(IMAGES_PATH)

    # 2. Inspección Triple
    triple_visual_inspection(LABELS_PATH, IMAGES_PATH, num_samples=3)

    # 3. Normalización
    mu, sigma = calculate_rgb_normalization_stats(IMAGES_PATH)
