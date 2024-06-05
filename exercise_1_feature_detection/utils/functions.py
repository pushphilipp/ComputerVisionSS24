import cv2
import numpy as np
from typing import Tuple


def compute_harris_response(I: np.array, k: float = 0.06) -> Tuple[np.array]:
    """Determines the Harris Response of an Image.

    Args:
        I: A Gray-level image in float32 format.
        k: A constant changing the trace to determinant ratio.

    Returns:
        A tuple with float images containing the Harris response (R) and other intermediary images. Specifically
        (R, A, B, C, Idx, Idy).
    """
    assert I.dtype == np.float32

    # Step 1: Compute Idx and Idy with cv2.Sobel
    print(I.shape)
    Idx = cv2.Sobel(I, cv2.CV_64F, 1, 0, ksize=3)
    Idy = cv2.Sobel(I, cv2.CV_64F, 0, 1, ksize=3)

    # Step 2: Ixx Iyy Ixy from Idx and Idy
    Ixx = Idx * Idx # Element-wise multiplication
    Iyy = Idy * Idy
    Ixy = Idx * Idy

    # Step 3: compute A, B, C from Ixx, Iyy, Ixy with cv2.GaussianBlur
    # Use sdev = 1 and kernelSize = (3, 3) in cv2.GaussianBlur
    A = cv2.GaussianBlur(Ixx, (3,3), 1)
    B = cv2.GaussianBlur(Iyy, (3,3), 1)
    C = cv2.GaussianBlur(Ixy, (3,3), 1)

    # Step 4: Compute the harris response with the determinant and the trace of T
    det = A * B - C * C
    trace = A + B
    R = det - k * trace * trace 

    return R, A, B, C, Idx, Idy

def detect_corners(R: np.array, threshold: float = 0.1) -> Tuple[np.array, np.array]:
    """Computes key-points from a Harris response image.

    Key points are all points where the harris response is significant and greater than its neighbors.

    Args:
        R: A float image with the harris response
        threshold: A float determining which Harris response values are significant.

    Returns:
        A tuple of two 1D integer arrays containing the x and y coordinates of key-points in the image.
    """
    # Step 1 (recommended): Pad the response image to facilitate vectorization
    I_pad = np.pad(R, 1, mode='constant', constant_values=0)

    # shape of padded image is (R.shape[0] + 2, R.shape[1] + 2) (258x258)

    # Step 2 (recommended): Create one image for every offset in the 3x3 neighborhood
    neighbors = np.zeros((3, 3, *R.shape))
    for i in range(3):
        for j in range(3):
            neighbors[i, j] = I_pad[i:i+R.shape[0], j:j+R.shape[1]]

    # shape of neighbors is (3, 3, 256, 256)

    # Step 3 (recommended): Compute the greatest neighbor of every pixel from the 3x3 neighborhood
    # perfor the max operation along the first two axes (0,1) to get the maximum value of each pixel
    max_neighbors = np.max(neighbors, axis=(0, 1))

    # shape of max_neighbors is (256, 256)

    # Step 4 (recommended): Compute a boolean image with only all key-points set to True
    key_points = R > threshold
    key_points = key_points & (R == max_neighbors)

    # shape of key_points is (256, 256)

    # Step 5 (recommended): Use np.nonzero to compute the locations of the key-points from the boolean image
    y, x = np.nonzero(key_points)

    # shape of x and y is (n,) where n is the number of key-points -> number of corners

    return x, y


def detect_edges(R: np.array, edge_threshold: float = -0.01) -> np.array:
    """Computes a boolean image where edge pixels are set to True.

    Edges are significant pixels of the harris response that are a local minimum along the x or y axis.

    Args:
        R: a float image with the harris response.
        edge_threshold: A constant determining which response pixels are significant

    Returns:
        A boolean image with edge pixels set to True.
    """
    # Step 1 (recommended): Pad the response image to facilitate vectorization
    I_pad = np.pad(R, 1, mode='constant', constant_values=0)

    # Step 2 (recommended): Calculate significant response pixels
    significant = R <= edge_threshold

    # Step 3 (recommended): Create two images with the smaller x-axis and y-axis neighbors respectively
    # check for smaller neighbors along the x-axis
    x_neighbors = np.minimum(I_pad[1:-1, :-2], I_pad[1:-1, 2:])
    # check for smaller neighbors along the y-axis
    y_neighbors = np.minimum(I_pad[:-2, 1:-1], I_pad[2:, 1:-1])

    # Step 4 (recommended): Calculate pixels that are lower than either their x-axis or y-axis neighbors
    x_minimal = significant & (R < x_neighbors)
    y_minimal = significant & (R < y_neighbors)

    # Step 5 (recommended): Calculate valid edge pixels by combining significant and axis_minimal pixels
    edges = significant & (x_minimal | y_minimal)

    return edges
