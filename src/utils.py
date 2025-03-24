import numpy as np


def pb_image_to_numpy(rgbpx, depthpx, segpx, width, height):
    """
    Convert pybullet camera images to numpy arrays. =Convert the raw image data (RGB, depth, and segmentation) into NumPy arrays

    Args:
        rgbpx: RGBA pixel values
        depthpx: Depth map pixel values
        segpx: Segmentation map pixel values
        width: Image width
        height: Image height

    Returns:
        Tuple of:
            rgb: RGBA image as numpy array [height, width, 4]
            depth: Depth map as numpy array [height, width]
            seg: Segmentation map as numpy array [height, width]
    """
    # RGBA - channel Range [0-255]
    rgb = np.reshape(rgbpx, [height, width, 4])
    # Depth Map Range [0.0-1.0]
    depth = np.reshape(depthpx, [height, width])
    # Segmentation Map Range {obj_ids}
    seg = np.reshape(segpx, [height, width])

    return rgb, depth, seg


## Our code below:


def depth_to_point_cloud(depth, projection_matrix, mask=None):
        height, width = depth.shape
        fx = projection_matrix[0, 0]
        fy = projection_matrix[1, 1]
        cx = projection_matrix[0, 2]
        cy = projection_matrix[1, 2]

        x = np.arange(width)
        y = np.arange(height)
        x, y = np.meshgrid(x, y)

        x = (x - cx) * depth / fx
        y = (y - cy) * depth / fy
        z = depth

        points = np.stack((x, y, z), axis=-1)

        if mask is not None:
            points = points[mask]

        return points