from abc import ABC, abstractmethod

import einops
import math
import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import Resize
from tqdm import tqdm
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoProcessor,
    CLIPModel,
    CLIPProcessor,
    ViTMAEForPreTraining,
)


def get_device():
    """自动检测可用设备，有GPU则用GPU，否则用CPU"""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_processor_vit(model_name):
    model_name = model_name.lower()
    if model_name == "laion/CLIP-ViT-B-16-laion2B-s34B-b88K".lower():
        # 使用本地模型路径
        model, _, processor = open_clip.create_model_and_transforms(
            model_name="ViT-B-16",
            pretrained="./open_clip/open_clip_model.safetensors"
        )
        vit = model.visual
    else:
        raise ValueError(f"Unsupported model {model_name}.")
    return processor, vit



class BaseTiViT(nn.Module, ABC):
    def __init__(self, processor, vit, layer_idx, aggregation, patch_size, stride):
        super().__init__()
        self.processor = processor
        self.vit = vit
        self.layer_idx = layer_idx
        self.aggregation = aggregation
        self.patch_size = patch_size
        self.stride = stride
        self.processor = processor
        self.truncate_layers()

    @abstractmethod
    def truncate_layers(self):
        """Truncate transformer layers"""
        pass

    @abstractmethod
    def forward_vit(self, inputs):
        """Forward pass through ViT to extract hidden representations"""
        pass

    def forward(self, inputs):
        inputs = self.ts2image_transformation(
            inputs, patch_size=self.patch_size, stride=self.stride
        )
        hidden = self.forward_vit(inputs)

        return self.aggregate_hidden_representations(
            hidden, aggregation=self.aggregation
        )

    def aggregate_hidden_representations(self, hidden_states, aggregation):
        if aggregation == "mean":
            pooled = hidden_states.mean(dim=1)
        elif aggregation == "cls_token":
            pooled = hidden_states[:, 0, :]
        else:
            raise ValueError(f"Unsupported aggregation {aggregation}")

        return pooled

    def ts2image_transformation(
        self,
        x,
        patch_size,
        stride,
        image_size=224,
    ):
        # x: B x T x D
        # Normalization using robust scaling
        median = x.median(1, keepdim=True)[0]
        q_tensor = torch.tensor([0.75, 0.25], device=x.device, dtype=x.dtype)
        q75, q25 = torch.quantile(x, q_tensor, dim=1, keepdim=True)
        x = x - median
        iqr = q75 - q25
        x = x / (iqr + 1e-5)

        x = einops.rearrange(x, "b t d -> b d t")
        T = x.shape[-1]

        if stride == 1:  # No overlapping patches
            pad_left = 0
            if T % patch_size != 0:
                pad_left = patch_size - T % patch_size
            x_pad = F.pad(x, (pad_left, 0), mode="replicate")
            x_2d = einops.rearrange(x_pad, "b d (p f) -> (b d) 1 f p", f=patch_size)
        elif stride > 0 and stride < 1:  # Overlapping patches
            pad_left = 0
            if int(patch_size * stride) == 0:
                stride_len = 1
            else:
                stride_len = int(patch_size * stride)
            remainder = (T - patch_size) % stride_len
            if remainder != 0:
                pad_left = stride_len - remainder
            x_pad = F.pad(x, (pad_left, 0), mode="replicate")
            x_2d = x_pad.unfold(dimension=2, size=patch_size, step=stride_len)
        else:
            raise ValueError(
                f"Stride is set to {stride}, but should be a fraction of the patch size, and thus lie between 0 and 1."
            )

        # Adjust contrast
        min_vals = x_2d.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        max_vals = x_2d.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        x_2d = (x_2d - min_vals) / (max_vals - min_vals + 1e-5)
        x_2d = torch.pow(x_2d, 0.8)

        # Resize to ViT input resolution
        x_resized = Resize((image_size, image_size), interpolation=0, antialias=False)(
            x_2d
        )

        # Generate grayscale images
        image_input = einops.repeat(x_resized, "b 1 h w -> b c h w", c=3)

        return image_input




class TiViT_OpenCLIP(BaseTiViT):
    def __init__(self, processor, vit, layer_idx, aggregation, patch_size, stride):
        self.hidden_representations = {}
        super().__init__(processor, vit, layer_idx, aggregation, patch_size, stride)
        self.processor.transforms = [self.processor.transforms[-1]]

    def truncate_layers(self):
        if self.layer_idx is not None and self.layer_idx != -1:
            self.vit.transformer.resblocks = self.vit.transformer.resblocks[
                : self.layer_idx
            ]

    def forward_vit(self, inputs):
        hidden_states = []

        inputs = self.processor(inputs)

        x = self.vit._embeds(inputs)
        hidden_states.append(x)

        for blk in self.vit.transformer.resblocks:
            x = blk(x)
            hidden_states.append(x)

        if self.layer_idx is not None:
            return x
        else:
            return torch.stack(hidden_states, dim=-1)

def get_patch_size(patch_size, T):
    """
    Calculate patch size based on time series length.

    Args:
        patch_size: 'sqrt', 'linspace', or an integer
        T: time series length

    Returns:
        patch_size: integer or list of integers
    """
    if patch_size == "sqrt":
        return int(math.sqrt(T))
    elif patch_size == "linspace":
        return (
            np.linspace(1,
                        math.ceil(T // 2),
                        min(math.ceil(T // 2), 20),
                        ).astype(int)
            .tolist()
        )
    return patch_size


def embed(model, dataloaders, channels, device):
    """
    Extract embeddings from time series data using TiViT model.

    Args:
        model: TiViT model
        dataloader: DataLoader yielding batches of shape (B, C, T)
        channels: number of channels
        device: device to run on

    Returns:
        embeds: torch.Tensor of shape (B, C, D) - normalized embeddings
    """
    batch_embeds = []
    for (batch,) in tqdm(dataloaders, desc="Extracting embeddings"):
        batch_embeds_dim = []
        for dim in range(channels):
            batch_dim = batch[:, dim, :].unsqueeze(-1)  # (B, T, 1)
            with torch.no_grad():
                batch_dim = batch_dim.to(device)
                outputs = model(batch_dim)  # (B, D)
            batch_embeds_dim.append(outputs)
        # 按channel维度拼接: list of (B, D) -> (B, C, D)
        batch_embeds.append(torch.stack(batch_embeds_dim, dim=1))

    # 拼接所有batch: (N, C, D)
    embeds = torch.cat(batch_embeds, dim=0)

    # L2 normalization
    embeds = F.normalize(embeds, p=2, dim=-1)

    return embeds       

def get_tivit(
    model_name,
    model_layer,
    aggregation,
    stride,
    patch_size,
    device=None,
):
    """创建TiViT模型并移动到指定设备

    Args:
        model_name: CLIP模型名称
        model_layer: 提取的ViT层索引
        aggregation: 池化方式 ('mean' 或 'cls_token')
        stride: patch stride
        patch_size: patch大小
        device: 设备，默认为自动检测

    Returns:
        tivit: TiViT模型
    """
    if device is None:
        device = get_device()

    processor, vit = get_processor_vit(model_name)
    TiViTClass = TiViT_OpenCLIP

    tivit = TiViTClass(
        processor=processor,
        vit=vit,
        layer_idx=model_layer,
        aggregation=aggregation,
        patch_size=patch_size,
        stride=stride,
    )
    tivit = tivit.to(device)

    return tivit

def get_TS_Tivit_embed(Tivit_model, train_loader, channels, device=None):
    """提取时间序列的TiViT嵌入向量

    Args:
        Tivit_model: TiViT模型
        train_loader: 数据加载器
        channels: 通道数
        device: 设备，默认为自动检测

    Returns:
        vision_embeds: torch.Tensor of shape (N, C, D)
    """
    if device is None:
        device = get_device()

    vision_embeds = embed(Tivit_model, train_loader, channels, device)
    return vision_embeds


if __name__ == "__main__":
    from torch.utils.data import DataLoader, TensorDataset

    print("=" * 50)
    print("Testing TiViT")
    print("=" * 50)

    # 1. 测试 get_device
    print("\n[1] Testing get_device...")
    test_device = get_device()
    print(f"    Device: {test_device}")

    # 参数设置
    model_name = "laion/CLIP-ViT-B-16-laion2B-s34B-b88K"
    model_layer = 6
    aggregation = "mean"
    stride = 0.1
    patch_size = "sqrt"
    seq_len = 96

    # 计算实际 patch_size
    actual_patch_size = get_patch_size(patch_size, seq_len)

    # 2. 创建TiViT模型
    print("\n[2] Loading TiViT model...")
    print(f"    model_name: {model_name}")
    print(f"    model_layer: {model_layer}")
    print(f"    aggregation: {aggregation}")
    print(f"    patch_size: {patch_size} -> {actual_patch_size}")
    print(f"    stride: {stride}")

    # 由于本地可能没有GPU，使用CPU
    use_cpu = True  # 改为False如果有GPU
    device = torch.device("cpu") if use_cpu else get_device()
    print(f"    device: {device}")

    try:
        tivit = get_tivit(
            model_name=model_name,
            model_layer=model_layer,
            aggregation=aggregation,
            stride=stride,
            patch_size=actual_patch_size,  # 使用转换后的实际值
            device=device,
        )
        tivit.eval()
        print("    Model loaded successfully!")
    except Exception as e:
        print(f"    Failed to load model: {e}")
        print("    (可能是首次运行需要下载模型，或本地模型路径错误)")
        exit(1)

    # 3. 测试单样本前向传播
    print("\n[3] Testing forward pass (single sample)...")
    batch_size = 2
    features = 1

    # 输入shape: (B, T, D) - 注意TiViT期望的是 (B, T, D)
    x = torch.randn(batch_size, seq_len, features)
    print(f"    Input shape: {x.shape}")

    with torch.no_grad():
        output = tivit(x)

    print(f"    Output shape: {output.shape}")
    print(f"    Output dtype: {output.dtype}")
    print(f"    Output device: {output.device}")

    # 4. 测试 embed 函数 (使用DataLoader) - 正确处理多通道
    print("\n[6] Testing embed function with DataLoader...")

    # 创建模拟数据
    n_samples = 10
    channels = 7
    fake_data = torch.randn(n_samples, channels, seq_len)
    fake_dataset = TensorDataset(fake_data)
    fake_loader = DataLoader(fake_dataset, batch_size=2, shuffle=False)

    # 直接用tivit模型
    embeds = embed(tivit, fake_loader, channels, device)
    print(f"    Embeddings shape: {embeds.shape}")
    print(f"    Expected: ({n_samples}, {channels}, 512)")
    print(f"    Is normalized: {torch.allclose(embeds.norm(p=2, dim=-1), torch.ones_like(embeds.norm(p=2, dim=-1)), atol=1e-5)}")

    # 5. 测试 get_TS_Tivit_embed
    print("\n[7] Testing get_TS_Tivit_embed...")
    embeds2 = get_TS_Tivit_embed(tivit, fake_loader, channels, device=device)
    print(f"    Embeddings shape: {embeds2.shape}")

    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)