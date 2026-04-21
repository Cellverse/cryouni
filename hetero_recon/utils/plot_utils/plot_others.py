import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation
import torch

from cellverse.math_utils.gmm import GaussianMixture

try:
    from cuml import PCA
    from cuml.manifold.umap import UMAP
except ImportError:
    from sklearn.decomposition import PCA
    from umap import UMAP


def plot_euler_angles_versus(
    euler_ref: np.ndarray,
    euler_aligned: np.ndarray,
):
    fig = plt.figure(dpi=96, figsize=(15, 5))
    plt.subplot(131)
    plt.plot(euler_ref[:, 0], euler_aligned[:, 0], 'ro', alpha=.1)
    plt.xlabel('alpha ref')
    plt.ylabel('alpha aligned')
    plt.subplot(132)
    plt.plot(euler_ref[:, 1], euler_aligned[:, 1], 'go', alpha=.1)
    plt.xlabel('beta ref')
    plt.ylabel('beta aligned')
    plt.subplot(133)
    plt.plot(euler_ref[:, 2], euler_aligned[:, 2], 'bo', alpha=.1)
    plt.xlabel('gamma ref')
    plt.ylabel('gamma aligned')
    return fig


def plot_in_plane_error(in_plane_errors: np.ndarray,):
    fig = plt.figure(dpi=96, figsize=(7, 5))
    plt.hist(in_plane_errors.flatten(), bins=100)
    plt.xlabel('In-Plane Error (deg)')
    return fig


def plot_trans(
    trans_gt: np.ndarray,
    trans_pred: np.ndarray,
):
    fig = plt.figure(dpi=96, figsize=(10, 4))
    plt.subplot(121)
    plt.plot(trans_gt[:, 0], trans_pred[:, 0], 'ko', alpha=.1)
    plt.xlabel('x gt')
    plt.subplot(122)
    plt.plot(trans_gt[:, 1], trans_pred[:, 1], 'ko', alpha=.1)
    plt.xlabel('y gt')
    return fig


def rotmat_to_euler(rotmat: np.ndarray) -> np.ndarray:
    """
    Convert a rotation matrix to Euler angles.

    Args:
        `rotmat` (np.ndarray): Rotation matrix, shape [..., 3, 3].

    Returns:
        (np.ndarray): Euler angles, shape [..., 3].
    """
    return Rotation.from_matrix(rotmat.swapaxes(-2, -1)).as_euler('zxz')


def direction_to_azimuth_elevation(out_of_planes):
    """
    out_of_planes: [..., 3]
    up: Y
    plane: (Z, X)
    output: ([...], [...]) (azimuth, elevation)
    """
    azimuth = np.arctan2(out_of_planes[..., 0], out_of_planes[..., 2])
    elevation = np.arcsin(out_of_planes[..., 1])
    return azimuth, elevation


def s2s2_to_matrix(v1, v2=None):
    """
    Normalize 2 3-vectors. Project second to orthogonal component.
    Take cross product for third. Stack to form SO matrix.
    """
    if v2 is None:
        assert v1.shape[-1] == 6
        v2 = v1[..., 3 :]
        v1 = v1[..., 0 : 3]
    u1 = v1
    e1 = u1 / u1.norm(p=2, dim=-1, keepdim=True).clamp(min=1E-5)
    u2 = v2 - (e1 * v2).sum(-1, keepdim=True) * e1
    e2 = u2 / u2.norm(p=2, dim=-1, keepdim=True).clamp(min=1E-5)
    e3 = torch.linalg.cross(e1, e2)
    return torch.cat([e1[..., None, :], e2[..., None, :], e3[..., None, :]], -2)


def euler_to_rotmat(euler: np.ndarray) -> np.ndarray:
    """
    Convert Euler angles to rotation matrix.

    Args:
        `euler` (np.ndarray): Euler angles, shape [..., 3].

    Returns:
        (np.ndarray): Rotation matrix, shape [..., 3, 3].
    """
    return Rotation.from_euler('zxz', euler).as_matrix().swapaxes(-2, -1)


def select_predicted_latent(pred_full, activated_paths):
    """
    rots_full: [sym_loss_factor * batch_size, ...]
    activated_paths: [batch_size]
    """
    batch_size = activated_paths.shape[0]
    pred_full = pred_full.reshape(-1, batch_size, *pred_full.shape[1 :])
    list_arange = np.arange(batch_size)
    pred = pred_full[activated_paths, list_arange]
    return pred


### Dimensionality reduction ###


def run_pca(z):
    pca = PCA(n_components=z.shape[1])
    pca.fit(z)
    pc = pca.transform(z)
    return pc, pca


def run_umap(z, **kwargs):
    reducer = UMAP(**kwargs)
    z_embedded = reducer.fit_transform(z)
    return z_embedded


def cluster_gmm(z, K, on_data=True, random_state=None, covariance_type='diag', **kwargs):
    '''
    Cluster z by a K-component full covariance Gaussian mixture model

    Inputs:
        z (Ndata x zdim np.array): Latent encodings
        K (int): Number of clusters
        on_data (bool): Compute cluster center as nearest point on the data manifold
        random_state (int or None): Random seed used for GMM clustering
        **kwargs: Additional keyword arguments passed to GaussianMixture

    Returns:
        np.array (Ndata,) of cluster labels
        np.array (K x zdim) of cluster centers
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    clf = GaussianMixture(n_components=K, covariance_type=covariance_type, random_state=random_state, **kwargs)
    labels = clf.fit_predict(torch.from_numpy(z).float().to(device)).detach().cpu().numpy()
    centers = clf.means_.detach().cpu().numpy()
    if on_data:
        centers, centers_ind = get_nearest_point(z, centers)
    return labels, centers


def get_nearest_point(data, query):
    '''
    Find closest point in @data to @query
    Return datapoint, index
    '''
    ind = cdist(query, data).argmin(axis=1)
    return data[ind], ind


def get_pc_traj(
    pca: PCA,
    zdim: int,
    numpoints: int,
    dim: int,
    start: float | None = None,
    end: float | None = None,
    percentiles: np.ndarray | None = None,
) -> np.ndarray:
    """
    Create trajectory along specified principal component

    Inputs:
        pca: sklearn PCA object from run_pca
        zdim (int)
        numpoints (int): number of points between @start and @end
        dim (int): PC dimension for the trajectory (1-based index)
        start (float): Value of PC{dim} to start trajectory
        end (float): Value of PC{dim} to stop trajectory
        percentiles (np.array or None): Define percentile array instead of np.linspace(start,stop,numpoints)

    Returns:
        np.array (numpoints x zdim) of z values along PC
    """
    if percentiles is not None:
        assert len(percentiles) == numpoints
    traj_pca = np.zeros((numpoints, zdim))
    if percentiles is not None:
        traj_pca[:, dim - 1] = percentiles
    else:
        assert start is not None
        assert end is not None
        traj_pca[:, dim - 1] = np.linspace(start, end, numpoints)
    ztraj_pca = pca.inverse_transform(traj_pca)
    return ztraj_pca
