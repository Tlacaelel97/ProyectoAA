import torch
import torch.nn as nn
import torch.nn.functional as F

# Importamos la configuración y el modelo base
from src.model import VQVAE, VQVAEConfig


class SegmentationHead(nn.Module):
    """
    Cabeza de segmentación que proyecta el espacio latente discreto
    a los logits de las 64 clases semánticas.
    """

    def __init__(self, embedding_dim, num_hiddens, num_classes=64):
        """
        Inicializa la cabeza de segmentación.
        Args:
            embedding_dim (int): Dimensión del espacio latente del VQ-VAE.
            num_hiddens (int): Número de canales en las capas ocultas.
            num_classes (int): Número de clases semánticas para segmentación.
        """
        super(SegmentationHead, self).__init__()

        # Procesamiento del mapa latente 64x64
        self._conv_1 = nn.Conv2d(embedding_dim, num_hiddens, kernel_size=3, padding=1)

        # Upsampling para recuperar la resolución 512x512
        # 64x64 -> 128x128
        self._upsample_1 = nn.ConvTranspose2d(
            num_hiddens, num_hiddens // 2, kernel_size=4, stride=2, padding=1
        )
        # 128x128 -> 256x256
        self._upsample_2 = nn.ConvTranspose2d(
            num_hiddens // 2, num_hiddens // 4, kernel_size=4, stride=2, padding=1
        )
        # 256x256 -> 512x512
        self._upsample_3 = nn.ConvTranspose2d(
            num_hiddens // 4, num_classes, kernel_size=4, stride=2, padding=1
        )

    def forward(self, quantized) -> torch.Tensor:
        """
        Propaga el mapa latente cuantizado a través de la cabeza de segmentación.
        Args:
            quantized (Tensor): Mapa latente cuantizado de forma (B, embedding_dim, 64, 64).
        Returns:
            Tensor: Logits de segmentación de forma (B, num_classes, 512, 512).
        """
        x = F.relu(self._conv_1(quantized))
        x = F.relu(self._upsample_1(x))
        x = F.relu(self._upsample_2(x))
        return self._upsample_3(x)


class GOOSESegmentationModel(nn.Module):
    """
    Modelo unificado que integra el extractor VQ-VAE con la cabeza de segmentación.
    """

    def __init__(self, vqvae_trained_model, num_classes=64):
        """
        Inicializa el modelo de segmentación GOOSE.
        Args:
            vqvae_trained_model (VQVAE): Modelo VQ-VAE preentrenado y congelado.
            num_classes (int): Número de clases semánticas para segmentación.
        """
        super(GOOSESegmentationModel, self).__init__()

        # Extraemos y congelamos los componentes del VQ-VAE
        self.encoder = vqvae_trained_model._encoder
        self.pre_vq_conv = vqvae_trained_model._pre_vq_conv
        self.vq = vqvae_trained_model._vq

        # Instanciamos la cabeza de segmentación usando la config del modelo base
        self.seg_head = SegmentationHead(
            embedding_dim=vqvae_trained_model.config.embedding_dim,
            num_hiddens=vqvae_trained_model.config.num_hiddens,
            num_classes=num_classes,
        )

    def forward(self, x) -> torch.Tensor:
        """
        Propaga la imagen de entrada a través del extractor VQ-VAE y luego a la cabeza de segmentación.
        Args:
            x (Tensor): Imagen de entrada de forma (B, 3, 512, 512).
        Returns:
            Tensor: Logits de segmentación de forma (B, num_classes, 512, 512).
        """
        # El flujo utiliza el extractor de características discretas
        with torch.no_grad():  # Congelamos el extractor para Fase 3 inicial
            z = self.pre_vq_conv(self.encoder(x))
            _, quantized, _ = self.vq(z)

        # La segmentación se entrena sobre la representación cuantizada
        return self.seg_head(quantized)
