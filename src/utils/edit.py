import numpy as np
import scipy.ndimage as ndi


def largest_cluster(array:np.ndarray)->np.ndarray:
    """Given a mask array, return a new mask array containing only the largest cluster.

    Args:
        array (np.ndarray): mask array with values 1 (inside) or 0 (outside)

    Returns:
        np.ndarray: mask array with only a single connect cluster of pixels.
    """
    # Label all features in the array
    label_img, cnt = ndi.label(array)
    # Find the label of the largest feature
    labels = range(1,cnt+1)
    size = [np.count_nonzero(label_img==l) for l in labels]
    max_label = labels[size.index(np.amax(size))]
    # Return a mask corresponding to the largest feature
    return label_img==max_label


def largest_cluster_label(array:np.ndarray)->np.ndarray:
    """Given a label image, return a new label image with only the 
    largest cluster for each label.
    """
    output_array = np.zeros(array.shape, dtype=np.int16)
    for label_value in np.unique(array):
        if label_value == 0:
            continue
        mask = np.zeros(array.shape)
        mask[array==label_value] = 1
        mask = largest_cluster(mask)
        output_array[mask] = label_value
    return output_array


