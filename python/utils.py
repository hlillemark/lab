import numpy as np


def getRawDepth(depth_buffer_val, proj_matrix):
    """Converts depth from buffer a [0,1] range to view-space linear depth."""
    zNears, zFars = extract_znear_zfar_from_projection(proj_matrix)
    zNears = np.expand_dims(zNears, -1)
    zFars = np.expand_dims(zFars, -1)
    ndc_depth = depth_buffer_val * 2.0 - 1.0  # [0, 1] -> [-1, 1]
    view_space_depth = (2.0 * zNears * zFars) / (zFars + zNears - ndc_depth * (zFars - zNears))
    return view_space_depth

def extract_znear_zfar_from_projection(P):
    """
    Extracts near and far plane distances from a right-handed projection matrix.
    NOTE: The formula B/(A+1) yields the far plane and B/(A-1) yields the near plane.
    This function returns them in that "swapped" order, as required by the
    linearization function for a reversed-Z depth buffer.
    """
    # Using names that reflect the swapped output needed for the next step.

    A = P[2, 2]
    B = P[2, 3]

    # In a right-handed system, B/(A+1) is the far plane distance.
    far_dist = B / (A + 1.0)
    # And B/(A-1) is the near plane distance.
    near_dist = B / (A - 1.0)
    
    return far_dist, near_dist