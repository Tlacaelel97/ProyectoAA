import pathlib
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

def scan_labels(base_path):
    label_paths = list(pathlib.Path(base_path).rglob('*.png'))
    data = []
    for p in label_paths:
        # Extraer metadatos del nombre del directorio superior
        sequence_name = p.parent.name
        data.append({
            "path": str(p),
            "sequence": sequence_name,
            "filename": p.name
        })
    return pd.DataFrame(data)

   

def run_eda_stats(df, num_classes=64):
    # Inicializar contadores
    pixel_counts = np.zeros(num_classes, dtype=np.int64)
    image_presence = np.zeros(num_classes, dtype=np.int64)
    
    print("Iniciando análisis de distribución de etiquetas...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # Leer máscara (label)
        mask = cv2.imread(row['path'], cv2.IMREAD_GRAYSCALE)
        
        # 1. Conteo de píxeles (Pixel-wise)
        counts = np.bincount(mask.flatten(), minlength=num_classes)
        pixel_counts += counts
        
        # 2. Presencia en imagen (Image-wise)
        present_classes = np.unique(mask)
        for c in present_classes:
            if c < num_classes:
                image_presence[c] += 1
                
    return pixel_counts, image_presence

if __name__ == "__main__":
    base_path = "data/goose2D/2d_challenge"
    df = scan_labels(base_path)
    print(df.head())

    pixel_stats, presence_stats = run_eda_stats(df)
    print("Distribución de píxeles por clase:")
    for i, count in enumerate(pixel_stats):
        print(f"Clase {i}: {count} píxeles")    