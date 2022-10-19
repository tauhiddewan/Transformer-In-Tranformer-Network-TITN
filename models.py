# Student 2 : Vit
import torch
import torch.nn as nn
from cutmix_utils import CutMixCriterion
from einops import repeat, rearrange, reduce
from einops.layers.torch import Reduce, Rearrange 
from torch import einsum
def exists(val):
    return val is not None 
def default(val, d):
    return val if exists(val) else d
def get_output_size(image_size, kernel_size, stride, padding):
    return int (((image_size-kernel_size+(2*padding))/stride)+1)

# Student 1 : Vit
# For image_size=32, VIT(img_size=32, patch_size=8, patch_emb_dim=192, depth=12, n_classes=n_classes)
class vit_MLP(nn.Module):
    def __init__(self, patch_emb_dim, mlp_expand=4, mlp_drop=0.0):
        super(vit_MLP, self).__init__()
        self.fc1 = nn.Linear(patch_emb_dim, mlp_expand*patch_emb_dim)
        self.fc2 = nn.Linear(mlp_expand*patch_emb_dim, patch_emb_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(mlp_drop)
    def forward(self, x):
        out = self.dropout(self.act(self.fc1(x)))
        out = self.dropout(self.fc2(out))
        return out
class vit_Attention(nn.Module):
    def __init__(self, patch_emb_dim, n_heads=8, attn_drop=0.0):
        super(vit_Attention, self).__init__()
        self.n_heads = n_heads
        self.patch_emb_dim = patch_emb_dim
        self.head_dim = patch_emb_dim // n_heads
        self.inner_dim = self.n_heads*self.head_dim
        self.keys, self.queries, self.values = (nn.Linear(patch_emb_dim , self.inner_dim), 
                                                nn.Linear(patch_emb_dim, self.inner_dim), 
                                                nn.Linear(patch_emb_dim, self.inner_dim))
        self.to_out = nn.Sequential(nn.Linear(self.inner_dim, self.patch_emb_dim),
                                    nn.Dropout(attn_drop))
    def forward(self, x):
        b, n, d = x.shape
        h = self.n_heads
        scale = (self.head_dim)**(1/2)
        q = rearrange(self.queries(x), 'b n (h d) -> (b h) n d', h=h)
        k = rearrange(self.keys(x), 'b n (h d) -> (b h) n d', h=h)
        v = rearrange(self.values(x), 'b n (h d) -> (b h) n d', h=h)
        qk_dot = einsum('b i d, b j d -> b i j', q, k)/scale
        qk_attention = F.softmax(qk_dot, dim=-1)
        qkv_dot = einsum('b i j, b j d -> b i d', qk_attention, v)
        out = rearrange(qkv_dot, '(b h) n d -> b n (h d)', h = h) 
        out = self.to_out(out)
        return out
class vit_TransformerBlock(nn.Module):
    def __init__(self, patch_emb_dim, n_heads,
                 attn_drop=0.1, mlp_drop=0.0):
        super(vit_TransformerBlock, self).__init__()
        self.norm = nn.LayerNorm(patch_emb_dim)
        self.attn = vit_Attention(patch_emb_dim=patch_emb_dim, 
                              n_heads=8, attn_drop=attn_drop)
        # self.dropout = nn.Dropout(dropout_rate)
        self.mlp = vit_MLP(patch_emb_dim=patch_emb_dim, 
                            mlp_expand=4, mlp_drop=mlp_drop)
    def forward(self, x):
        residual = x
        out = self.norm(x)
        out = self.attn(out)
        out += residual
        residual = out
        out = self.norm(out)
        out = self.mlp(out)
        out += residual
        return out
class vit_TransformerEncoder(nn.Module):
    def __init__(self, num_patches, patch_emb_dim, 
                 depth=12, n_heads=8, attn_drop=0.0, mlp_drop=0.0):
        super(vit_TransformerEncoder, self).__init__()
        # positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, patch_emb_dim))
        # encoder blocks
        self.layers = nn.ModuleList()
        for i in range(depth):
            layer = vit_TransformerBlock(patch_emb_dim=patch_emb_dim, 
                                     n_heads=n_heads, attn_drop=attn_drop, mlp_drop=mlp_drop)
            self.layers.append(layer)
        self.norm = nn.LayerNorm(patch_emb_dim)
    def forward(self, x):
        out = x + self.pos_embedding
        for layer in self.layers:
            out = layer(out)
        out = self.norm(out)
        return out
class VIT(nn.Module):
    def __init__(self,img_size, patch_size, patch_emb_dim, depth, n_classes,
                 attn_drop=0.1, mlp_drop=0.0, channels=3, n_heads=8):
        super(VIT, self).__init__()
        self.num_patches = (img_size//patch_size)**2
        self.channels, self.n_heads = channels, n_heads
        self.img_size, self.patch_size, self.patch_emb_dim = img_size, patch_size, patch_emb_dim
        self.depth, self.n_classes = depth, n_classes
        self.attn_drop, self.mlp_drop = attn_drop, mlp_drop
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.patch_emb_dim))
        self.patch_pos_emb = nn.Parameter(torch.randn(1, self.num_patches+1, self.patch_emb_dim))
        self.to_patch_emb = nn.Sequential(nn.Conv2d(in_channels=self.channels, out_channels=self.patch_emb_dim, kernel_size=self.patch_size, stride=self.patch_size),
                                          Rearrange('b e h w -> b (h w) e'))
        
        self.transformer = vit_TransformerEncoder(num_patches=self.num_patches, patch_emb_dim=self.patch_emb_dim, 
                                              depth=self.depth, n_heads=self.n_heads, 
                                              attn_drop=0.0, mlp_drop=0.0)
        
        self.classification_head = nn.Sequential(Reduce('b n e -> b e', reduction='mean'),
                                                 nn.LayerNorm(self.patch_emb_dim),
                                                 nn.Linear(in_features=self.patch_emb_dim, out_features=n_classes))

    def forward(self, x):
        b, n, h, w = x.shape # [b, c, h, w]
        cls_token = repeat(self.cls_token, '() n e -> b n e', b=b)
        
        patch_emb = self.to_patch_emb(x)
        patch_emb = torch.cat([cls_token, patch_emb], dim=1) # added class tokens
        patch_emb += self.patch_pos_emb # added patch positional embeddings
        
        x = self.transformer(patch_emb) # forward pass to transformer block
        
        # Classfication head
        cls_tok = x[:, :-1]
        out_cls = self.classification_head(cls_tok)
        return out_cls

# Student 2 : DIstillable Vit
# For image_size=32, distillVIT(img_size=32, patch_size=8, patch_emb_dim=192, depth=12, n_classes=n_classes)
class MLP(nn.Module):
    def __init__(self, patch_emb_dim, mlp_expand=4, mlp_drop=0.0):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(patch_emb_dim, mlp_expand*patch_emb_dim)
        self.fc2 = nn.Linear(mlp_expand*patch_emb_dim, patch_emb_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(mlp_drop)
    def forward(self, x):
        out = self.dropout(self.act(self.fc1(x)))
        out = self.dropout(self.fc2(out))
        return out
class Attention(nn.Module):
    def __init__(self, patch_emb_dim, n_heads=8, attn_drop=0.0):
        super(Attention, self).__init__()
        self.n_heads = n_heads
        self.patch_emb_dim = patch_emb_dim
        self.head_dim = patch_emb_dim // n_heads
        self.inner_dim = self.n_heads*self.head_dim
        self.keys, self.queries, self.values = (nn.Linear(patch_emb_dim , self.inner_dim), 
                                                nn.Linear(patch_emb_dim, self.inner_dim), 
                                                nn.Linear(patch_emb_dim, self.inner_dim))
        self.to_out = nn.Sequential(nn.Linear(self.inner_dim, self.patch_emb_dim),
                                    nn.Dropout(attn_drop))
    def forward(self, x):
        b, n, d = x.shape
        h = self.n_heads
        scale = (self.head_dim)**(1/2)
        q = rearrange(self.queries(x), 'b n (h d) -> (b h) n d', h=h)
        k = rearrange(self.keys(x), 'b n (h d) -> (b h) n d', h=h)
        v = rearrange(self.values(x), 'b n (h d) -> (b h) n d', h=h)
        qk_dot = einsum('b i d, b j d -> b i j', q, k)/scale
        qk_attention = F.softmax(qk_dot, dim=-1)
        qkv_dot = einsum('b i j, b j d -> b i d', qk_attention, v)
        out = rearrange(qkv_dot, '(b h) n d -> b n (h d)', h = h) 
        out = self.to_out(out)
        return out
class TransformerBlock(nn.Module):
    def __init__(self, patch_emb_dim, n_heads,
                 attn_drop=0.1, mlp_drop=0.0):
        super(TransformerBlock, self).__init__()
        self.norm = nn.LayerNorm(patch_emb_dim)
        self.attn = Attention(patch_emb_dim=patch_emb_dim, 
                              n_heads=8, attn_drop=attn_drop)
        # self.dropout = nn.Dropout(dropout_rate)
        self.mlp = MLP(patch_emb_dim=patch_emb_dim, 
                            mlp_expand=4, mlp_drop=mlp_drop)
    def forward(self, x):
        residual = x
        out = self.norm(x)
        out = self.attn(out)
        out += residual
        residual = out
        out = self.norm(out)
        out = self.mlp(out)
        out += residual
        return out
class TransformerEncoder(nn.Module):
    def __init__(self, num_patches, patch_emb_dim, 
                 depth=12, n_heads=8, attn_drop=0.0, mlp_drop=0.0):
        super(TransformerEncoder, self).__init__()
        # positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 2, patch_emb_dim))
        self.dis_embedding = nn.Parameter(torch.randn(1, num_patches + 2, patch_emb_dim))
        # encoder blocks
        self.layers = nn.ModuleList()
        for i in range(depth):
            layer = TransformerBlock(patch_emb_dim=patch_emb_dim, 
                                     n_heads=n_heads, attn_drop=attn_drop, mlp_drop=mlp_drop)
            self.layers.append(layer)
        self.norm = nn.LayerNorm(patch_emb_dim)
    def forward(self, x):
        out = x + self.pos_embedding
        out = out + self.dis_embedding
        for layer in self.layers:
            out = layer(out)
        out = self.norm(out)
        return out
class distillVIT(nn.Module):
    def __init__(self,img_size, patch_size, patch_emb_dim, depth, n_classes,
                 attn_drop=0.1, mlp_drop=0.0, channels=3, n_heads=8):
        super(distillVIT, self).__init__()
        self.num_patches = (img_size//patch_size)**2
        self.channels, self.n_heads = channels, n_heads
        self.img_size, self.patch_size, self.patch_emb_dim = img_size, patch_size, patch_emb_dim
        self.depth, self.n_classes = depth, n_classes
        self.attn_drop, self.mlp_drop = attn_drop, mlp_drop
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.patch_emb_dim))
        self.dis_token = nn.Parameter(torch.randn(1, 1, self.patch_emb_dim))
        self.patch_pos_emb = nn.Parameter(torch.randn(1, self.num_patches+2, self.patch_emb_dim))
        self.to_patch_emb = nn.Sequential(nn.Conv2d(in_channels=self.channels, out_channels=self.patch_emb_dim, kernel_size=self.patch_size, stride=self.patch_size),
                                          Rearrange('b e h w -> b (h w) e'))
        
        self.transformer = TransformerEncoder(num_patches=self.num_patches, patch_emb_dim=self.patch_emb_dim, 
                                              depth=self.depth, n_heads=self.n_heads, 
                                              attn_drop=0.0, mlp_drop=0.0)
        
        self.classification_head = nn.Sequential(Reduce('b n e -> b e', reduction='mean'),
                                                 nn.LayerNorm(self.patch_emb_dim),
                                                 nn.Linear(in_features=self.patch_emb_dim, out_features=n_classes))
        self.distil_head = nn.Sequential(nn.LayerNorm(self.patch_emb_dim),
                                         nn.Linear(in_features=self.patch_emb_dim, out_features=n_classes))

    def forward(self, x):
        b, n, h, w = x.shape # [b, c, h, w]
        cls_token = repeat(self.cls_token, '() n e -> b n e', b=b)
        dis_token = repeat(self.dis_token, '() n e -> b n e', b=b)
        
        patch_emb = self.to_patch_emb(x)
        patch_emb = torch.cat([cls_token, patch_emb], dim=1) # added class tokens
        patch_emb = torch.cat([patch_emb, dis_token], dim=1) # added distillation tokens
        patch_emb += self.patch_pos_emb # added patch positional embeddings
        
        x = self.transformer(patch_emb) # forward pass to transformer block
        
        # Classfication head
        cls_tok, dis_tok = x[:, :-1], x[:, -1]
        out_cls, out_dist = self.classification_head(cls_tok), self.distil_head(dis_tok)
        if self.training:
            x = out_cls, out_dist
        else:
            x = (out_cls + out_dist) / 2
        return x
    

# Student 3 : TNT
# For image_size=224, TNT(image_size = 224, patch_dim = 384, pixel_dim = 24, patch_size = 16, pixel_size = 4)
# For image_size=32, TNT(image_size = 32, patch_dim = 192, pixel_dim = 12, patch_size = 8, pixel_size = 2, 
#                        channels=3, n_classes = n_classes, depth = 12, attn_drop = 0.0, mlp_drop = 0.0)
class PreNorm(nn.Module):
    def __init__(self, dim , fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn 
    def forward(self, x, **kwargs):
        x = self.norm(x)
        x = self.fn(x, **kwargs)
        return x
class tnt_MLP(nn.Module):
    def __init__(self, emb_dim, *, expansion=4, mlp_drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features=emb_dim, out_features=expansion*emb_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(in_features=expansion*emb_dim, out_features=emb_dim)
        self.drop = nn.Dropout(mlp_drop)
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.drop(x)
        x = self.drop(self.fc2(x))
        return x   
class tnt_Attention(nn.Module):
    def __init__(self, *, emb_dim=768, head_dim=64, n_heads=8, attn_drop=0.):
        super().__init__()
        inner_dim = n_heads*head_dim
        self.ans = inner_dim
        self.n_heads = n_heads 
        self.head_dim = head_dim
        self.keys, self.queries, self.values = nn.Linear(emb_dim , inner_dim), nn.Linear(emb_dim, inner_dim), nn.Linear(emb_dim, inner_dim)
        self.att_drop = nn.Dropout(attn_drop)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, emb_dim),
            nn.Dropout(attn_drop)
        )
    def forward(self, x):
        b, n, d = x.shape 
        h = self.n_heads
        scale = (self.head_dim)**(1/2)
        q = rearrange(self.queries(x), 'b n (h d) -> (b h) n d', h=h)
        k = rearrange(self.keys(x), 'b n (h d) -> (b h) n d', h=h)
        v = rearrange(self.values(x), 'b n (h d) -> (b h) n d', h=h)
        
        qk_dot = einsum('b i d, b j d -> b i j', q, k)/scale
        qk_attention = F.softmax(qk_dot, dim=-1)
        
        qkv_dot = einsum('b i j, b j d -> b i d', qk_attention, v)
        out = rearrange(qkv_dot, '(b h) n d -> b n (h d)', h = h) 
        out = self.to_out(out)
        return out
class TNT(nn.Module):
        def __init__(self, *, image_size, patch_dim, pixel_dim, patch_size, pixel_size, n_classes, depth, channels=3,
                     inner_n_heads=4, outer_n_heads=8, inner_head_dim=3, outer_head_dim=24, attn_drop=0., 
                     mlp_drop=0., unfold_args=None, **kwargs):
                super().__init__()
                num_patches = (image_size//patch_size)**2 
                self.patch_dim = patch_dim
                self.n_classes = n_classes
                self.image_size = image_size 
                self.patch_size = patch_size 
                self.channels = channels
                # Pixel-Embeddings 
                unfold_args = default(unfold_args, (pixel_size, pixel_size, 0))
                unfold_args = (*unfold_args, 0) if len(unfold_args)==2 else unfold_args
                kernel_size, stride, padding = unfold_args 
                self.to_pixel_tokens = nn.Sequential(
                        Rearrange('b c (h p1) (w p2) -> (b h w) c p1 p2', p1=patch_size, p2=patch_size), 
                        nn.Unfold(kernel_size=kernel_size, stride=stride, padding=padding),
                        Rearrange('... c n -> ... n c'),
                        nn.Linear(self.channels*kernel_size**2, pixel_dim)
                )
                # position Embeddings
                pixel_width = get_output_size(image_size=patch_size,
                                        kernel_size=kernel_size, 
                                        stride=stride, padding=padding)
                num_pixels = pixel_width**2
                self.patch_tokens = nn.Parameter(torch.randn(num_patches+1, patch_dim))
                self.patch_pos_emb = nn.Parameter(torch.randn(num_patches+1, patch_dim)) #[197, 768]
                self.pixel_pos_emb = nn.Parameter(torch.randn(num_pixels, pixel_dim))
                
                # Creating TNT Blocks in depth
                layers = nn.ModuleList([])
                for _ in range(depth):
                        pixel_to_patch = nn.Sequential(
                                nn.LayerNorm(pixel_dim),
                                Rearrange('... n d -> ... (n d)'),
                                nn.Linear(in_features=num_pixels*pixel_dim, out_features=patch_dim)
                                )
                        layers.append(nn.ModuleList([
                                PreNorm(pixel_dim, tnt_Attention(emb_dim=pixel_dim, head_dim=inner_head_dim, n_heads=inner_n_heads, attn_drop=attn_drop)),
                                PreNorm(pixel_dim, tnt_MLP(emb_dim=pixel_dim, mlp_drop=mlp_drop)),
                                pixel_to_patch,
                                PreNorm(patch_dim, tnt_Attention(emb_dim=patch_dim, head_dim=outer_head_dim, n_heads=outer_n_heads, attn_drop=attn_drop)),
                                PreNorm(patch_dim, tnt_MLP(emb_dim=patch_dim, mlp_drop=mlp_drop))
                                ]))
                self.layers = layers
                # Classification Head - Final Layer
                self.classification_head = nn.Sequential(nn.LayerNorm(patch_dim),
                                                         nn.Linear(in_features=patch_dim, out_features=n_classes))
        def forward(self, x):
                b, c, h, w = x.shape
                patch_size = self.patch_size 
                image_size = self.image_size
                num_patch_h = h // patch_size
                num_patch_w = w // patch_size 
                num_patches = num_patch_h*num_patch_w
                pixels = self.to_pixel_tokens(x)
                pixels += rearrange(self.pixel_pos_emb, 'n d -> () n d')
                patches = repeat(self.patch_tokens[:(num_patches+1)], 'n d -> b n d', b=b)
                patches += rearrange(self.patch_pos_emb[:(num_patches+1)], 'n d -> () n d')
                for pixel_attn, pixel_mlp, pixel_to_patch_res, patch_attn, patch_mlp in self.layers:
                        pixels = pixel_attn(pixels) + pixels
                        pixels = pixel_mlp(pixels) + pixels
                        
                        patches_residual = pixel_to_patch_res(pixels)
                        patches_residual = rearrange(patches_residual, '(b h w) d -> b (h w) d', h=num_patch_h, w=num_patch_w)
                        # -> [1, 16, 192] -> [1, 18, 192] 
                        # pad 2nd-to-last dim by 1 on each side -> pads o to first-row(cls_tok) and last-row(dist_tok)
                        patches_residual = F.pad(input=patches_residual, pad=(0, 0, 1, 0), value=0) 
                        patches = patches+patches_residual
                        patches = patch_attn(patches)+patches
                        patches = patch_mlp(patches)+patches
                class_token = patches[:, 0]
                out = self.classification_head(class_token)
                return out
          
# Student 4 : DIstillable TNT
# For image_size=224, DTNT(image_size = 224, patch_dim = 384, pixel_dim = 24, patch_size = 16, pixel_size = 4)
# For image_size=32, DTNT(image_size = 32, patch_dim = 192, pixel_dim = 12, patch_size = 8, pixel_size = 2, 
#                         channels=3, n_classes = n_classes, depth = 12, attn_drop = 0.0, mlp_drop = 0.0)

class DTNT(nn.Module):
        def __init__(self, *, image_size, patch_dim, pixel_dim, patch_size, pixel_size, n_classes, depth, channels=3,
                     inner_n_heads=4, outer_n_heads=8, inner_head_dim=3, outer_head_dim=24, attn_drop=0., 
                     mlp_drop=0., unfold_args=None, **kwargs):
                super().__init__()
                num_patches = (image_size//patch_size)**2 
                self.patch_dim = patch_dim
                self.n_classes = n_classes
                self.image_size = image_size 
                self.patch_size = patch_size 
                self.channels = channels
                # Pixel-Embeddings 
                unfold_args = default(unfold_args, (pixel_size, pixel_size, 0))
                unfold_args = (*unfold_args, 0) if len(unfold_args)==2 else unfold_args
                kernel_size, stride, padding = unfold_args 
                self.to_pixel_tokens = nn.Sequential(
                        Rearrange('b c (h p1) (w p2) -> (b h w) c p1 p2', p1=patch_size, p2=patch_size), 
                        nn.Unfold(kernel_size=kernel_size, stride=stride, padding=padding),
                        Rearrange('... c n -> ... n c'),
                        nn.Linear(self.channels*kernel_size**2, pixel_dim)
                )
                # position Embeddings
                pixel_width = get_output_size(image_size=patch_size,
                                        kernel_size=kernel_size, 
                                        stride=stride, padding=padding)
                num_pixels = pixel_width**2
                self.patch_tokens = nn.Parameter(torch.randn(num_patches+2, patch_dim))
                self.dist_token = nn.Parameter(torch.randn(num_patches+2, patch_dim))
                self.patch_pos_emb = nn.Parameter(torch.randn(num_patches+2, patch_dim)) #[197, 768]
                self.pixel_pos_emb = nn.Parameter(torch.randn(num_pixels, pixel_dim))
                
                # Creating TNT Blocks in depth
                layers = nn.ModuleList([])
                for _ in range(depth):
                        pixel_to_patch = nn.Sequential(
                                nn.LayerNorm(pixel_dim),
                                Rearrange('... n d -> ... (n d)'),
                                nn.Linear(in_features=num_pixels*pixel_dim, out_features=patch_dim)
                                )
                        layers.append(nn.ModuleList([
                                PreNorm(pixel_dim, tnt_Attention(emb_dim=pixel_dim, head_dim=inner_head_dim, n_heads=inner_n_heads, attn_drop=attn_drop)),
                                PreNorm(pixel_dim, tnt_MLP(emb_dim=pixel_dim, mlp_drop=mlp_drop)),
                                pixel_to_patch,
                                PreNorm(patch_dim, tnt_Attention(emb_dim=patch_dim, head_dim=outer_head_dim, n_heads=outer_n_heads, attn_drop=attn_drop)),
                                PreNorm(patch_dim, tnt_MLP(emb_dim=patch_dim, mlp_drop=mlp_drop))
                                ]))
                self.layers = layers
                # Classification Head - Final Layer 
                self.classification_head = nn.Sequential(Reduce('b n e -> b e', reduction='mean'),
                                                         nn.LayerNorm(patch_dim),
                                                         nn.Linear(in_features=patch_dim, out_features=n_classes))
                self.distil_head = nn.Sequential(
                        nn.LayerNorm(patch_dim),
                        nn.Linear(in_features=patch_dim, out_features=n_classes)
                )
        def forward(self, x):
                b, c, h, w = x.shape
                patch_size = self.patch_size 
                image_size = self.image_size
                num_patch_h = h // patch_size
                num_patch_w = w // patch_size 
                num_patches = num_patch_h*num_patch_w
                pixels = self.to_pixel_tokens(x)
                pixels += rearrange(self.pixel_pos_emb, 'n d -> () n d')
                patches = repeat(self.patch_tokens[:(num_patches+2)], 'n d -> b n d', b=b)
                patches += rearrange(self.patch_pos_emb[:(num_patches+2)], 'n d -> () n d')
                patches += rearrange(self.dist_token[:(num_patches+2)], 'n d -> () n d') #extra distill token added
                for pixel_attn, pixel_mlp, pixel_to_patch_res, patch_attn, patch_mlp in self.layers:
                        pixels = pixel_attn(pixels) + pixels
                        pixels = pixel_mlp(pixels) + pixels
                        
                        patches_residual = pixel_to_patch_res(pixels)
                        patches_residual = rearrange(patches_residual, '(b h w) d -> b (h w) d', h=num_patch_h, w=num_patch_w)
                        # -> [1, 16, 192] -> [1, 18, 192] 
                        # pad 2nd-to-last dim by 1 on each side -> pads o to first-row(cls_tok) and last-row(dist_tok)
                        patches_residual = F.pad(input=patches_residual, pad=(0, 0, 1, 1), value=0) 
                        
                        patches = patches+patches_residual
                        patches = patch_attn(patches)+patches
                        patches = patch_mlp(patches)+patches
                        
                class_token, dist_tok = patches[:, :-1], patches[:, -1]
                out_cls, out_dist = self.classification_head(class_token), self.distil_head(dist_tok)
                if self.training:
                    x = out_cls, out_dist
                else:
                    x = (out_cls + out_dist) / 2
                return x    
class HardDistLoss(nn.Module):
    def __init__(self, teacher, using_cutmix=False):
        super().__init__()
        self.teacher = teacher 
        if using_cutmix:
            self.criterion1 = CutMixCriterion(reduction='mean')
        else:
            self.criterion1 = nn.CrossEntropyLoss(reduction='mean')
        self.criterion2 = nn.CrossEntropyLoss(reduction='mean')
    def forward(self, inputs, outputs, labels):
        outputs_cls, outputs_dist = outputs
        base_loss = self.criterion1(outputs_cls, labels)
        with torch.no_grad():
            teacher_outputs = self.teacher(inputs)    
        teacher_labels = torch.argmax(teacher_outputs, dim=1)
        teacher_loss = self.criterion2(outputs_dist, teacher_labels) #cause it works xD
        hdloss = base_loss*0.5+teacher_loss*0.5
        return hdloss