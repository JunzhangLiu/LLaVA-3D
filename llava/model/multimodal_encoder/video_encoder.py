import torch
import torch.nn as nn

from .video_processor import RGBDVideoProcessor
from .spatial_aware_module import SpatialAwareModule
from .unproject import backprojector_dataloader, voxelize
from torch_scatter import scatter_mean
from .position_encodings import PositionEmbeddingLearnedMLP


def kmeans_cluster(features, xyz, K, num_iters=10):
    """
    K-means++ spatial clustering on 3D positions; aggregates features per cluster.

    Args:
        features: (B, N, F) per-patch features
        xyz:      (B, V, H, W, 3) or (B, N, 3) 3D positions
        K:        number of clusters (output tokens per scene)
        num_iters: Lloyd iteration count

    Returns:
        pooled:       (B*K, F) cluster-averaged features
        batch_offset: (B,)  int32 cumulative token counts (each = K)
    """
    B, N, F = features.shape
    device = features.device
    xyz_flat = xyz.reshape(B, -1, 3)  # (B, N, 3)

    pooled_list = []
    for b in range(B):
        pts  = xyz_flat[b]   # (N, 3)
        feat = features[b]   # (N, F)

        # K-means++ initialisation: spread seeds across space
        first = torch.randint(N, (1,), device=device)
        centroids = pts[first]                                            # (1, 3)
        for _ in range(K - 1):
            d2 = torch.cdist(pts, centroids).min(dim=1).values ** 2      # (N,)
            nxt = torch.multinomial(d2 / d2.sum(), 1)
            centroids = torch.cat([centroids, pts[nxt]], dim=0)           # (k+1, 3)

        # Lloyd iterations
        for _ in range(num_iters):
            assign = torch.cdist(pts, centroids).argmin(dim=1)            # (N,)
            centroids = scatter_mean(pts, assign, dim=0, dim_size=K)      # (K, 3)

        assign = torch.cdist(pts, centroids).argmin(dim=1)                # (N,)
        pooled_list.append(scatter_mean(feat, assign, dim=0, dim_size=K)) # (K, F)

    pooled = torch.cat(pooled_list, dim=0)                                # (B*K, F)
    batch_offset = torch.arange(1, B + 1, device=device, dtype=torch.int32) * K
    return pooled, batch_offset


class PromptEncoder(nn.Module):
    
    def __init__(self, latent_dim=4096):
        super(PromptEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.pos_emb3d = PositionEmbeddingLearnedMLP(dim=3, num_pos_feats=latent_dim)

    def encode_pe(self, xyz=None):
        return self.pos_emb3d(xyz)
    
    def forward(self, clicks):
        # (n, 3)
        pos_embed = self.encode_pe(clicks) #  (N, F)
        return pos_embed

class RGBDVideoTower(nn.Module):
    def __init__(self, vision_tower, video_tower, args, delay_load=False):
        super().__init__()
        self.is_loaded = False
        self.num_frames = args.num_frames
        self.num_sample_tokens = args.num_sample_tokens
        self.pooling = 'voxelize'
        self.voxel_size = 0.2
        self.num_clusters = 512
        self.vision_tower_name = vision_tower
        self.video_tower_name = video_tower

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_video_tower', False):
            self.load_model()
        else:
            self.cfg_only = None

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.video_tower_name))
            return

        self.video_processor = RGBDVideoProcessor(self.vision_tower_name, self.num_frames)
        if self.video_tower_name == 'SpatialAwareModule':
            self.video_tower = SpatialAwareModule()
        else:
            raise NotImplementedError

        self.prompt_encoder = PromptEncoder()
        # self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def forward(self, features, depths, poses, intrinsics, lengths=None):
        """
        Compute visual features/position embeddings for each patch.

        Args:
            - features: (B, V, 1024, 336, 336), image token features
            - depths: (B, V, H, W), depth images
            - poses: (B, V, 4, 4) pose information
            - instrinsics: (B, V, 4, 4), intriniscs
            - lengths: (B,)  view number of each scene

        Returns:
            - rgb_feats_pyramid: [(B, ncam, F, H_i, W_i)]
            - pcd_pyramid: [(B, ncam * H_i * W_i, 3)]
        """
        B, V, C, H, W = features.shape
        assert intrinsics.dim() == 4
        # (B, V, 24, 24, 3)
        feat_xyz, xyz = backprojector_dataloader([features.flatten(0, 1)], depths, poses, intrinsics)
        # (B, V*H*W, C)
        video_features = self.video_tower([features.flatten(0, 1)], [feat_xyz.flatten(0, 1)], (B, V))[0]
        video_xyz = feat_xyz.reshape(B, V*H*W, 3)
        if lengths is not None:
            lengths = lengths*H*W

        if self.pooling == 'voxelize':
            p2v = voxelize(feat_xyz, self.voxel_size)  # (B, N)
            pooled_video_features = torch.cat([scatter_mean(video_features[b], p2v[b], dim=0) for b in range(len(video_features))]) # (B*n_voxels, F)
            batch_offset = ((p2v).max(1)[0] + 1).cumsum(0).to(torch.int32)
        elif self.pooling == 'kmeans':
            pooled_video_features, batch_offset = kmeans_cluster(video_features, feat_xyz, self.num_clusters)
        else:
            raise NotImplementedError
        
        return pooled_video_features, batch_offset  # (B, num_token, 1024) or (Bn, 1024)

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size
