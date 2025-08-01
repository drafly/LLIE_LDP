import torch
import torch.nn.functional as F
import numpy as np

def get_local_max_torch(img: torch.Tensor, win: int = 5) -> torch.Tensor:
    """Get local maximum using max pooling."""
    img = img.unsqueeze(0).unsqueeze(0)  # shape: [1,1,H,W]
    max_pooled = F.max_pool2d(img, kernel_size=2*win+1, stride=1, padding=win)
    return max_pooled.squeeze()

def downsample_block(img: torch.Tensor, num: int = 50) -> torch.Tensor:
    """Downsample image to num x num blocks."""
    m, n = img.shape
    blkm = m // num
    blkn = n // num
    return img[:blkm*num:blkm, :blkn*num:blkn]  # shape: [num, num]

def loe_block(epic: torch.Tensor, ipic: torch.Tensor, win: int = 5, num: int = 50) -> float:
    """
    Compute block-wise LOE (Lightness Order Error) between two images.
    epic: enhanced image, shape [H,W,3] or [H,W]
    ipic: input image, same shape
    win: local max window size
    num: block size (LOE_b default: 50x50)
    """
    assert epic.shape == ipic.shape, "Input and enhanced images must be the same shape"

    # Convert to grayscale max if RGB
    if epic.ndim == 3 and epic.shape[2] == 3:
        ipic = ipic.max(dim=2).values
        epic = epic.max(dim=2).values
    elif epic.ndim == 2:
        pass
    else:
        raise ValueError("Unsupported image shape")

    ipic = ipic.double()
    epic = epic.double()

    imax = get_local_max_torch(ipic, win)
    emax = get_local_max_torch(epic, win)

    ipic_ds = downsample_block(imax, num)
    epic_ds = downsample_block(emax, num)

    RD = torch.zeros((num, num), dtype=torch.float64)

    for i in range(num):
        for j in range(num):
            ip_val = ipic_ds[i, j]
            ep_val = epic_ds[i, j]

            ip_temp = (ipic_ds >= ip_val).float()
            ep_temp = (epic_ds >= ep_val).float()

            flag = (ip_temp != ep_temp).float()
            RD[i, j] = flag.sum()
    normalized_value = RD.sum().item() / RD.numel()

    return normalized_value
