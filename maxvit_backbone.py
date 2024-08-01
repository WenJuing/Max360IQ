import torch
import torch.nn as nn
import numpy as np


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )
    
    def forward(self, x):
        return self.conv(x)


class Stem(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.conv1 = ConvBlock(cfg.img_channels, cfg.dim, 3, stride=2, padding=1)
        self.conv2 = ConvBlock(cfg.dim, cfg.dim, 3, stride=1, padding=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        
        return x
    


class MaxViTBlock(nn.Module):
    """MaxViT block, consists of MBConv, Block Attention and Grid Attnetion.
    
    Args:
        downsample (bool): value is True in the first block of each stage,
                           value is False when in_channels and out_channels is equal.
        num_heads (int): number of heads.
        drop_path (float): droppath rate after MSA or MLP.
    """
    def __init__(self, in_channels, out_channels, downsample, num_heads, drop_path, cfg):
        super().__init__()
        self.mbconv = MBConv(
            in_channels=in_channels,
            out_channels=out_channels,
            downsample=downsample,
            cfg=cfg
        )
        self.block_attention = TransformerBlock(
            dim=out_channels,
            num_heads=num_heads,
            drop_path=drop_path,
            attn_size=cfg.window_size,
            partition_function=window_partition,
            reverse_function=window_reverse,
            cfg=cfg
        )
        self.grid_attention = TransformerBlock(
            dim=out_channels,
            num_heads=num_heads,
            drop_path=drop_path,
            attn_size=cfg.grid_size,
            partition_function=grid_partition,
            reverse_function=grid_reverse,
            cfg=cfg
        )
        
    def forward(self, x):
        x = self.mbconv(x)
        x = self.block_attention(x)
        x = self.grid_attention(x)
        
        return x
    

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        """ Constructor for DepthwiseSeparableConv """
        super().__init__()
        self.dw_conv = nn.Sequential(
            ConvBlock(in_channels, in_channels, kernel_size, stride, padding=1, groups=in_channels),
            ConvBlock(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        return self.dw_conv(x)
    

def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)
    
    
class SqueezeExcitation(nn.Module):
    """SE Block.
    paper `Squeeze-and-Excitation Networks`

    Args:
        rd_rate (float): reduce rate of in_channels.
    """
    def __init__(self, in_channels, cfg):
        super().__init__()
        rd_channels = int(in_channels * cfg.rd_rate)
        self.proj1 = nn.Conv2d(in_channels, rd_channels, 1)
        self.gelu = nn.GELU()
        self.proj2 = nn.Conv2d(rd_channels, in_channels, 1)
        self.gate = nn.Sigmoid()
        
    def forward(self, x):
        _x = x
        # [B, C, H, W] -> mean -> [B, C, 1, 1]
        x = x.mean((2, 3), keepdim=True)
        # [B, C, 1, 1] -> proj1 -> [B, rd_C, 1, 1] -> proj2 -> [B, C, 1, 1]
        x = self.proj2(self.gelu(self.proj1(x)))
        
        return _x * self.gate(x)
    

class MBConv(nn.Module):
    """MBConv structure.
    Args:
        downsample (bool): value is True in the first block of each stage,
                           value is False when in_channels and out_channels is equal.
        """
    def __init__(self, in_channels, out_channels, downsample, cfg):
        super().__init__()
        self.math_branch = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels, 1),
            DepthwiseSeparableConv(in_channels, out_channels, 3, stride=2 if downsample else 1),
            SqueezeExcitation(out_channels, cfg),
            nn.Conv2d(out_channels, out_channels, 1)
        )
        
        self.sub_branch = nn.Sequential(
                nn.MaxPool2d(2, 2),
                nn.Conv2d(in_channels, out_channels, 1),
            ) if downsample else nn.Identity()
        
    def forward(self, x):
        output = self.math_branch(x)
        x = output + self.sub_branch(x)
        
        return x
    
    
def window_partition(x, P=7):
    """Deviding the tensor into windows.
    
    Args:
        x (tensor): input tensor.
        P (int): window size.
    """
    B, C, H, W = x.shape
    # [B, C, H, W] -> reshape -> [B, C, H/P, P, W/P, P] -> permute -> [B, H/P, W/P, P, P, C]
    x = x.reshape(B, C, H // P, P, W // P, P).permute(0, 2, 4, 3, 5, 1).contiguous()
    # [B, H // P, W // P, P, P, C] -> reshape -> [B*HW/P/P, P*P, C] = [_B, n, C]
    x = x.reshape(-1, P*P, C)
    
    return x


def window_reverse(x, H, W):
    """The reverse operation about window partition.
    
    Args:
        x (tensor): input tensor.
        H (int): original H of x.
        W (int): original W of x.
    """
    _B, n, C = x.shape
    P = int(np.sqrt(n))
    # [_B, n, C] -> reshape -> [B, H/P, W/P, P, P, C] -> permute -> [B, C, H/P, P, W/P, P] -> reshape -> [B, C, H, W]
    x = x.reshape(-1, H // P, W // P, P, P, C).permute(0, 5, 1, 3, 2, 4).contiguous().reshape(-1, C, H, W)
    
    return x
    

def grid_partition(x, G=8):
    """Deviding the tensor into grids.
    
    Args:
        x (tensor): input tensor.
        G (int): grid size.
    """
    B, C, H, W = x.shape
    # [B, C, H, W] -> reshape -> [B, C, G, H/G, G, W/G] -> permute -> [B, H/G, W/G, G, G, C]
    x = x.reshape(B, C, G, H // G, G, W // G).permute(0, 3, 5, 2, 4, 1).contiguous()
    # [B, H/G, W/G, G, G, C] -> reshape -> [B*HW/G/G, G*G, C] = [_B, n, C]
    x = x.reshape(-1, G*G, C)
    
    return x


def grid_reverse(x, H, W):
    """The reverse operation about grid partition.
    
    Args:
        x (tensor): input tensor.
        H (int): original H of x.
        W (int): original W of x.
    """
    _B, n, C = x.shape
    G = int(np.sqrt(n))
    # [_B, n, C] -> reshape -> [B, H/G, W/G, G, G, C] -> permute -> [B, C, G, H/G, G, W/G] -> reshape -> [B, C, H, W]
    x = x.reshape(-1, H // G, W // G, G, G, C).permute(0, 5, 3, 1, 4, 2).contiguous().reshape(-1, C, H, W)
    
    return x


def get_relative_position_index(
        win_h: int,
        win_w: int
):
    """ Function to generate pair-wise relative position index for each token inside the window.
        Taken from Timms Swin V1 implementation.

    Args:
        win_h (int): Window/Grid height.
        win_w (int): Window/Grid width.

    Returns:
        relative_coords (torch.Tensor): Pair-wise relative position indexes [height * width, height * width].
    """
    coords = torch.stack(torch.meshgrid([torch.arange(win_h), torch.arange(win_w)], indexing='ij'))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += win_h - 1
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w - 1
    return relative_coords.sum(-1)


class RelativeAttention(nn.Module):
    """ Relative Self-Attention similar to Swin V1. Implementation inspired by Timms Swin V1 implementation.

    Args:
        in_channels (int): Number of input channels.
        num_heads (int, optional): Number of attention heads. Default 32
        grid_window_size (Tuple[int, int], optional): Grid/Window size to be utilized. Default (7, 7)
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, in_channels, num_heads, M, cfg):
        """ Constructor method """
        # Call super constructor
        super(RelativeAttention, self).__init__()
        # Save parameters
        self.in_channels: int = in_channels
        self.num_heads: int = num_heads
        self.grid_window_size = M
        self.scale: float = num_heads ** -0.5
        self.attn_area = M * M
        # Init layers
        self.qkv_mapping = nn.Linear(in_features=in_channels, out_features=3 * in_channels)
        self.attn_drop = nn.Dropout(p=cfg.attn_drop)
        self.proj = nn.Linear(in_features=in_channels, out_features=in_channels)
        self.proj_drop = nn.Dropout(p=cfg.drop)
        self.softmax = nn.Softmax(dim=-1)
        # Define a parameter table of relative position bias, shape: 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * M - 1) * (2 * M- 1), num_heads))

        # Get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index", get_relative_position_index(M, M))
        # Init relative positional bias
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def _get_relative_positional_bias(
            self
    ) -> torch.Tensor:
        """ Returns the relative positional bias.

        Returns:
            relative_position_bias (torch.Tensor): Relative positional bias.
        """
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(self.attn_area, self.attn_area, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)

    def forward(self, input):
        """ Forward pass.

        Args:
            input (torch.Tensor): Input tensor of the shape [B_, N, C].

        Returns:
            output (torch.Tensor): Output tensor of the shape [B_, N, C].
        """
        # Get shape of input
        B_, N, C = input.shape
        # Perform query key value mapping
        qkv = self.qkv_mapping(input).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        # Scale query
        q = q * self.scale
        # Compute attention maps
        attn = self.attn_drop(self.softmax(q @ k.transpose(-2, -1) + self._get_relative_positional_bias()))
        # Map value with attention maps
        output = (attn @ v).transpose(1, 2).reshape(B_, N, -1)
        # Perform final projection and dropout
        output = self.proj(output)
        output = self.proj_drop(output)
        return output
    

class MLP(nn.Module):
    def __init__(self, in_channels, cfg):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, in_channels*cfg.mlp_expansion, 1)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv1d(in_channels*cfg.mlp_expansion, in_channels, 1)
        self.dropout = nn.Dropout(cfg.mlp_dropout)
        
    def forward(self, x):
        # [B, n, d] -> transpose ->[B, d, n] -> conv1 -> [B, 4d, n]
        x = self.conv1(x.transpose(1, 2))
        x = self.gelu(x)
        # [B, 4d, n] -> conv1 -> [B, d, n] -> transpose ->[B, n, d]
        x = self.conv2(x).transpose(1, 2)
        x = self.dropout(x)
        
        return x
    
    
class TransformerBlock(nn.Module):
    """Block Attention or Grid Attention.
    Args:
        dim (int): patch dimension.
        num_heads (int): number of heads.
        drop_path (float): droppath rate after MSA or MLP.
        M (int): side size of attention area.
        partition_function (func): window_partition or grid_partition.
        reverse_function (func): window_reverse or grid_reverse.
        """
    def __init__(self, dim, num_heads, drop_path, attn_size, partition_function, reverse_function, cfg):
        super().__init__()
        self.attn_size = attn_size
        self.partition_function = partition_function
        self.reverse_function = reverse_function
        self.relative_attention = RelativeAttention(
            in_channels=dim,
            num_heads=num_heads,
            M=attn_size,
            cfg=cfg
        )
        self.ln1 = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, cfg)

        
    def forward(self, x):
        _, C, H, W = x.shape
        # Perform partition: [B, C, H, W] -> partition -> [_B, n, C]
        x = self.partition_function(x, self.attn_size)
        x = x + self.drop_path(self.relative_attention(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        # Reverse partition: [_B, n, C] -> reverse -> [B, C, H, W]
        x = self.reverse_function(x, H, W)
        
        return x
    
    
class MaxViTStage(nn.Module):
    """MaxViT stage, consists of some MaxViT blocks.
    
    Args:
        num_blocks (int): number of blocks in this stage.
        num_heads (int): number of heads.
        drop_path (float): droppath rate after MSA or MLP.
        M (int): side size of attention aera.
    """
    def __init__(self, num_blocks, in_channels, out_channels, num_heads, drop_path, cfg):
        super().__init__()
        self.blocks = nn.Sequential(*[MaxViTBlock(
                in_channels=in_channels if index==0 else out_channels,
                out_channels=out_channels,
                downsample=index==0,
                num_heads=num_heads,
                drop_path=drop_path[index] if isinstance(drop_path, list) else drop_path,
                cfg=cfg
            ) 
            for index in range(num_blocks)]
        )
        
    def forward(self, x):
        x = self.blocks(x)
        
        return x