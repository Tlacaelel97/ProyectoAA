from typing import Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VQVAEConfig:
    """
    Objeto centralizado para la configuración de hiperparámetros del modelo.
    Facilita la experimentación y evita la redundancia de argumentos.
    """

    in_channels: int = 3
    num_hiddens: int = 128
    num_residual_layers: int = 2
    num_residual_hiddens: int = 32
    num_embeddings: int = 512
    embedding_dim: int = 64
    commitment_cost: float = 0.25


class ResidualBlock(nn.Module):
    """
    Bloque residual para preservar detalles de alta frecuencia en entornos rurales.
    """

    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        """
        Constructor del bloque residual.
        Args:
            in_channels: Número de canales de entrada.
            num_hiddens: Número de canales de salida del bloque residual.
            num_residual_hiddens: Número de canales en la capa oculta dentro del bloque residual.
        """
        super(ResidualBlock, self).__init__()
        # sequential de capas dentro del bloque residual
        self._block = nn.Sequential(
            nn.ReLU(True),  # activación ReLU para introducir no linealidad
            nn.Conv2d(  # convolución para procesar características dentro del bloque residual
                in_channels=in_channels,
                out_channels=num_residual_hiddens,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.ReLU(True),
            nn.Conv2d(
                in_channels=num_residual_hiddens,
                out_channels=num_hiddens,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )

    def forward(self, x):
        return x + self._block(x)


class Encoder(nn.Module):
    """
    Encoder que reduce la resolución espacial de 512x512 a 64x64.
    """

    def __init__(
        self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens
    ):
        """
        Constructor del encoder.
        Args:
            in_channels: Número de canales de entrada.
            num_hiddens: Número de canales de salida del encoder.
            num_residual_layers: Número de bloques residuales.
            num_residual_hiddens: Número de canales en la capa oculta dentro de cada bloque residual.
        """
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_hiddens // 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self._conv_2 = nn.Conv2d(
            in_channels=num_hiddens // 2,
            out_channels=num_hiddens,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self._conv_3 = nn.Conv2d(
            in_channels=num_hiddens,
            out_channels=num_hiddens,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        # Stack de bloques residuales para preservar detalles de alta frecuencia
        self._residual_stack = nn.Sequential(
            *[
                ResidualBlock(num_hiddens, num_hiddens, num_residual_hiddens)
                for _ in range(num_residual_layers)
            ]
        )

    def forward(self, x):
        x = F.relu(self._conv_1(x))
        x = F.relu(self._conv_2(x))
        x = F.relu(self._conv_3(x))
        return self._residual_stack(x)


class VectorQuantizer(nn.Module):
    """
    Capa de cuantización vectorial.
    Implementa la búsqueda del vecino más cercano en el codebook.
    """

    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        """
        Constructor de la capa de cuantización vectorial.
        Args:
            num_embeddings: Número de vectores en el codebook.
            embedding_dim: Dimensión de cada vector en el codebook.
            commitment_cost: Factor de compromiso para la pérdida de cuantización.
        """
        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(
            -1 / self._num_embeddings, 1 / self._num_embeddings
        )
        self._commitment_cost = commitment_cost

    def forward(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Cuantiza los inputs utilizando el codebook.
        Args:
            inputs: Tensor de forma (B, C, H, W) con las características a cuantizar.
        Returns:
            loss: Pérdida de cuantización.
            quantized: Tensor cuantizado de forma (B, C, H, W).
            encoding_indices: Índices de los vectores del codebook utilizados para cuantizar cada vector de entrada.
        """
        # BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self._embedding_dim)

        # Distancia L2: ||z_e(x) - e||^2
        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self._embedding.weight**2, dim=1)
            - 2 * torch.matmul(flat_input, self._embedding.weight.t())
        )

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(
            encoding_indices.shape[0], self._num_embeddings, device=inputs.device
        )
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Pérdidas VQ
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()

        return loss, quantized.permute(0, 3, 1, 2).contiguous(), encoding_indices


class Decoder(nn.Module):
    """
    Decoder que realiza el upsampling simétrico para reconstruir
    la imagen original a partir del espacio latente cuantizado.
    """

    def __init__(
        self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens
    ):
        super(Decoder, self).__init__()

        # Convolución inicial para procesar el mapa cuantizado
        self._conv_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_hiddens,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # Stack residual para recuperar texturas finas antes del upsampling
        self._residual_stack = nn.Sequential(
            *[
                ResidualBlock(num_hiddens, num_hiddens, num_residual_hiddens)
                for _ in range(num_residual_layers)
            ]
        )

        # Bloques de Upsampling (Transposed Convolutions)
        # 64x64 -> 128x128
        self._upsample_1 = nn.ConvTranspose2d(
            in_channels=num_hiddens,
            out_channels=num_hiddens // 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        # 128x128 -> 256x256
        self._upsample_2 = nn.ConvTranspose2d(
            in_channels=num_hiddens // 2,
            out_channels=num_hiddens // 4,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        # 256x256 -> 512x512
        self._upsample_3 = nn.ConvTranspose2d(
            in_channels=num_hiddens // 4,
            out_channels=3,  # Salida RGB
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def forward(self, x):
        x = self._conv_1(x)
        x = self._residual_stack(x)
        x = F.relu(self._upsample_1(x))
        x = F.relu(self._upsample_2(x))
        # La última capa no lleva activación ReLU para permitir el rango completo de color
        return self._upsample_3(x)


class VQVAE(nn.Module):
    """
    Orquestador principal. Instancia los submódulos utilizando la configuración centralizada.
    """

    def __init__(self, config: VQVAEConfig):
        super(VQVAE, self).__init__()
        self.config = config

        self._encoder = Encoder(
            in_channels=self.config.in_channels,
            num_hiddens=self.config.num_hiddens,
            num_residual_layers=self.config.num_residual_layers,
            num_residual_hiddens=self.config.num_residual_hiddens,
        )

        self._pre_vq_conv = nn.Conv2d(
            in_channels=self.config.num_hiddens,
            out_channels=self.config.embedding_dim,
            kernel_size=1,
            stride=1,
        )

        self._vq = VectorQuantizer(
            num_embeddings=self.config.num_embeddings,
            embedding_dim=self.config.embedding_dim,
            commitment_cost=self.config.commitment_cost,
        )

        self._decoder = Decoder(
            in_channels=self.config.embedding_dim,
            num_hiddens=self.config.num_hiddens,
            num_residual_layers=self.config.num_residual_layers,
            num_residual_hiddens=self.config.num_residual_hiddens,
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Flujo de datos completo: Encoder -> Bottleneck -> VectorQuantizer -> Decoder
        """
        z = self._encoder(x)
        z = self._pre_vq_conv(z)

        loss, quantized, indices = self._vq(z)
        x_recon = self._decoder(quantized)

        return loss, x_recon, indices
