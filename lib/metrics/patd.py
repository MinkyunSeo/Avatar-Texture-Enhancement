import torch
from sklearn.neighbors import NearestNeighbors

from lib.utils.metric_utils import *


def PATD(verts_pred, colors_pred, verts_gt, colors_gt):
    assert type(verts_pred) is torch.Tensor
    assert type(colors_pred) is torch.Tensor
    assert type(verts_gt) is torch.Tensor
    assert type(colors_gt) is torch.Tensor
    # Assert that all inputs are torch.Tensor
    
    assert verts_pred.dim() == 2
    assert colors_pred.dim() == 2
    assert verts_gt.dim() == 2
    assert colors_gt.dim() == 2
    # Assert that all tensors' dimensions are 2D
    
    assert verts_pred.size()[1] == 3
    assert colors_pred.size()[1] == 3
    assert verts_gt.size()[1] == 3
    assert colors_gt.size()[1] == 3
    # Assert that all tensors have 3 units
    
    N = verts_pred.size()[0]
    assert colors_pred.size()[0] == N
    assert N >= 4
    Np = verts_gt.size()[0]
    assert colors_gt.size()[0] == Np
    assert Np >= 4
    # Assert that vertex tensor have same shape with color tensor
    # Mesh should have more than 4 vertices (3D simplicial complex)
    
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(verts_gt)
    distances, indices = nbrs.kneighbors(verts_pred)
    
    patd = 0
    for i in range(N):
        patd += color_distance(colors_pred[i], colors_gt[indices[i][0]])
        
    patd /= N
        
    return patd
