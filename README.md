# Proyecto Final

## Data

Los datos deben estar dentro del proyecto en una carpeta llamada `data`

    data/
    └── goose2D/
        ├── goose_2d_train/           # Dataset original
        │   ├── images/
        │   │   └── train/                 # 23 Secuencias de imágenes RGB
        │   │       ├── 2022-07-22_flight/
        │   │       │   └── *.png
        │   │       ├── 2022-07-27_hoehenkirchner_forst/
        │   │       └── ... (otras 21 carpetas)
        │   └── labels/
        │       └── train/                 # Etiquetas detalladas[cite: 4]
        │           ├── 2022-07-22_flight/
        │           │   ├── *_labelids.png     # IDs de clase 0-63 
        │           │   ├── *_color.png        # Máscaras RGB
        │           │   └── *_instanceids.png  # IDs de objetos
        │           └── ...


Esto lo puedes lograr descargando directamente el dataset desde la pagina de goose, y descomprimiendolo dentro de la carpeta `data`. 

## src

Las funciones que se usan en el proyecto estan dentro de la carpeta `src` 

- `EDA.py`
- `dataset.py`

## Pruebas

Para ir ejecutando las pruebas se puede hacer desde el notebook `source.py` Se recomienda tener un venv e instalar los `requirements.txt`

```bash
pip install -r requirements.txt
```

Para consultar la versión de pytorch de acuerdo a tu sistema visita https://pytorch.org/get-started/locally/
