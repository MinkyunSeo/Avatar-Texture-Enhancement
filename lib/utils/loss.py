import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_3D_SSIM_error(img1, img2):
# Calculate SSIM error
# self.preds: [B, P, Res, N]
# self.labels: [B, P, Res, N]

        patch = img1
        patch_gt = img2

        # mu1 shape [B, P, Res, 1]
        mu1 = patch.mean(dim=(-1), keepdim=True)
        mu2 = patch_gt.mean(dim=(-1), keepdim=True)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        n = patch.shape[-1]     
        cov_norm = n / (n - 1)  # unbiased covariance
        sigma1_sq = (patch.pow(2).mean(dim=(-1), keepdim=True) - mu1_sq) * cov_norm
        sigma2_sq = (patch_gt.pow(2).mean(dim=(-1), keepdim=True) - mu2_sq) * cov_norm
        sigma12 = ((patch * patch_gt).mean(dim=(-1), keepdim=True) - mu1_mu2) * cov_norm

        C1 = 0.01**2
        C2 = 0.03**2

        # ssim_map shape [B, P, Res, 1]
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        ssim_loss = (1 - ssim_map.mean()).clamp(min=0, max=1)

        return ssim_loss

def get_3D_SSIM_error_Window(img1, img2):
        patch = img1
        patch_gt = img2

        # Todo: change window size to x*y*z (no need for window)
        window_size = 11

        # RGB channel
        channel = 3

        sigma = 1.5
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        gauss = gauss/gauss.sum()

        _1d_window = gauss.unsqueeze(1)
        _2d_window = _1d_window.mm(_1d_window.t())
        _3d_window = _1d_window.mm(_2d_window.reshape(1, -1)).reshape(window_size, window_size, window_size).float().unsqueeze(0).unsqueeze(0)
        window = _3d_window.expand(channel, 1, window_size, window_size, window_size).contiguous()

        mu1 = F.conv3d(patch, window, padding = window_size//2, groups = channel)
        mu2 = F.conv3d(patch_gt, window, padding = window_size//2, groups = channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)

        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv3d(patch*patch, window, padding = window_size//2, groups = channel) - mu1_sq
        sigma2_sq = F.conv3d(patch_gt*patch_gt, window, padding = window_size//2, groups = channel) - mu2_sq
        sigma12 = F.conv3d(patch*patch_gt, window, padding = window_size//2, groups = channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        ssim_loss = 1 - ssim_map.mean()

        return ssim_loss

# # print(img1)
# img2 = img1+3

img1 = torch.rand(2, 16, 3, 27)
img2 = torch.rand(2, 16, 3, 27)
img2 = img1 * 5
print(get_3D_SSIM_error(img1, img2))


