import torch
from torch import nn, einsum
from einops import rearrange, repeat
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CA(nn.Module):
    def __init__(self, input_channels, reduction_ratio=16):
        super(CA, self).__init__()
        self.input_channels = input_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        middle_channel = input_channels // reduction_ratio
        if middle_channel < 10:
            middle_channel = input_channels
        self.MLP1 = nn.Sequential(
            Flatten(),
            nn.Linear(input_channels, middle_channel),
            nn.ReLU(),
            nn.Linear(middle_channel, input_channels)
        )
        self.MLP2 = nn.Sequential(
            Flatten(),
            nn.Linear(input_channels, middle_channel),
            nn.ReLU(),
            nn.Linear(middle_channel, input_channels)
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        avg_values = self.avg_pool(x)
        max_values = self.max_pool(x)
        out = self.MLP1(avg_values) + self.MLP2(max_values)
        scale = x * torch.sigmoid(out).unsqueeze(2).unsqueeze(3).expand_as(x)
        scale = scale.permute(0, 2, 3, 1)
        return scale

class SA(nn.Module):
    def __init__(self, kernel_size=3):
        super(SA, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)
        return out


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x,channel_interaction, **kwargs):
        x_continue=self.fn(x,channel_interaction, **kwargs)
        
        return x_continue

class Residual_mlp(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim,shifted, fn):
        super().__init__()
        if not shifted:
            self.norm = nn.LayerNorm(dim//2)
        else:
            self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x,channel_interaction, **kwargs):
        return self.fn(self.norm(x),channel_interaction, **kwargs)

class PreNorm_mlp(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)

        
class ChannelAttention(nn.Module):

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return y


class WindowAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        self.inner_dim = head_dim * heads
        self.N = window_size * window_size
        self.dim = dim
        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted
        self.to_qkv = nn.Linear(dim//2, self.inner_dim * 3, bias=False)
        self.to_out = nn.Linear(self.inner_dim, dim//2)
        self.ca = CA(dim//2)    

    def forward(self, x,channel_interaction):

        b, n_h, n_w, _, h = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        
        v1 = qkv[2]
            
        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size

        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
                                h=h, w_h=self.window_size, w_w=self.window_size), qkv)

        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale
        
        channel_interaction_sigmoid = F.sigmoid(channel_interaction)
        channel_interaction_sigmoid=channel_interaction_sigmoid.reshape([-1, 1, self.heads, 1,  self.inner_dim // self.heads])
        v = v1.reshape([channel_interaction_sigmoid.shape[0], -1, self.heads, self.N, self.inner_dim // self.heads])
        v = v * channel_interaction_sigmoid
        v = v.permute(0, 2, 1, 3, 4)

        attn = dots.softmax(dim=-1)
        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)',
                        h=h, w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w)

        out = self.to_out(out)

        out = self.ca(out)

        return out

class SwinBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        self.shifted=shifted
        self.cab_norm = nn.LayerNorm(dim)
        self.proj_cnn_norm = nn.BatchNorm2d(dim)
        self.dwconv3x3 = nn.Sequential(
            nn.Conv2d(
                dim, dim,
                kernel_size=3,
                padding=3 // 2,
                groups=dim
            ),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        self.projection = nn.Conv2d(dim, dim // 2, kernel_size=1)
        self.channel_interaction = nn.Sequential(
            nn.Conv2d(dim, dim // 8, kernel_size=1),
            nn.BatchNorm2d(dim // 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 8, head_dim * heads, kernel_size=1),
        )
        self.attention_block = Residual(PreNorm(dim,shifted, WindowAttention(dim=dim,
                                                                     heads=heads,
                                                                     head_dim=head_dim,
                                                                     shifted=shifted,
                                                                     window_size=window_size,
                                                                     relative_pos_embedding=relative_pos_embedding)))                                                                  
        self.spatial_interaction=SA()
        self.conv_norm = nn.BatchNorm2d(dim // 2)    
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout2d(0.5)
        self.conv_cab = ChannelAttention(num_feat=dim, squeeze_factor=16)   
        self.mlp_block = Residual_mlp(PreNorm_mlp(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)))

    def forward(self, x_c,x_att_old):

        #dwc
        x_cnn = x_c.permute(0, 3, 1, 2)
        x_cnn = self.proj_cnn_norm(x_cnn) #     
        x_cnn = self.dwconv3x3(x_cnn)

        #Channel Attention
        channel_interaction = self.channel_interaction(F.adaptive_avg_pool2d(x_cnn, output_size=1))   
        x_cnn = self.projection(x_cnn)     
        x_sa_old = self.attention_block(x_att_old,channel_interaction)
        x_sa_old = x_sa_old.permute(0, 3, 1, 2)
        
        #Spatial Attention
        x_sa=self.spatial_interaction(x_sa_old)
        x = torch.concat([x_sa_old, x_cnn], dim=1)
        x=x_sa *x
        x=x.permute(0, 2, 3, 1)
            
        #cab
        x_cab_o=self.cab_norm(x_c)
        x_cab_o=x_cab_o.permute(0, 3, 1, 2)
        x_cab = self.conv_cab(x_cab_o)
            
        x_att_old1=x_att_old.permute(0, 3, 1, 2)
        x_cab=x_cab*x_cab_o
        x_cab=x_cab.permute(0, 2, 3, 1)
                   
        x=x+x_c+0.1*x_cab
        x = self.proj(x) 
        x = self.dropout(x)
        x = self.mlp_block(x)
        
        return x


class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.linear1 = nn.Linear(in_channels, out_channels//2)
    def forward(self, x):
        x=x.permute(0, 2, 3, 1)
        x_att = self.linear1(x)
        x = self.linear(x)

        return x,x_att


class HCSA(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers, downscaling_factor, num_heads, head_dim, window_size,
                 relative_pos_embedding, h_w):
        super().__init__()

        self.patch_partition = PatchMerging(in_channels=in_channels, out_channels=hidden_dimension)

        self.regular_block=SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding)

    def forward(self, x):
        x,x_att = self.patch_partition(x)
        global_x = self.regular_block(x,x_att)
        global_x = global_x.permute(0, 3, 1, 2)
               
        return global_x
