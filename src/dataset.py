import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import numpy as np
import pathlib


class GOOSEDataset(Dataset):
    """
    Dataset optimizado para el reto GOOSE basado en los hallazgos del EDA:
    - Resolución MuCAR-3: 2048 x 1000
    - Aspect Ratio: 2.05
    """

    def __init__(self, labels_path, images_path, is_train=True):
        self.labels_root = pathlib.Path(labels_path)
        self.images_root = pathlib.Path(images_path)
        self.label_files = list(self.labels_root.rglob("*_labelids.png"))

        # Estrategia de Preprocesamiento validada:
        # 1. CenterCrop para preservar escala sin deformar el AR de 2.05
        # 2. Normalización con valores calculados: Mu[0.318, 0.307, 0.303]
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.CenterCrop(1000),
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.3187, 0.3076, 0.3031], std=[0.2431, 0.2417, 0.2537]
                ),
            ]
        )

        self.label_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.CenterCrop(1000),
                transforms.Resize(
                    (512, 512), interpolation=transforms.InterpolationMode.NEAREST
                ),
            ]
        )

    def __len__(self):
        return len(self.label_files)

    def __getitem__(self, idx):
        lp = self.label_files[idx]
        seq_name = lp.parent.name
        frame_id = lp.name.split("__")[1].split("_")[0]

        # Búsqueda robusta por sensor visible (windshield_vis)
        img_folder = self.images_root / seq_name
        rp = next(img_folder.glob(f"*_{frame_id}_*vis.png"), None)

        if rp is None:
            raise FileNotFoundError(f"Frame {frame_id} no encontrado en {seq_name}")

        image = cv2.cvtColor(cv2.imread(str(rp)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(lp), cv2.IMREAD_GRAYSCALE)

        return (
            self.transform(image),
            torch.from_numpy(np.array(self.label_transform(mask))).long(),
        )
