import cv2
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine

# Convert image to size 32x32 and range 0-1
def convert_image(img, size=(32, 32)):

    # Remove nan values
    img_nonan = np.nan_to_num(img)

    # Normalize images to 0-1
    img_norm = cv2.normalize(img_nonan, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img_norm = np.where(np.isnan(img), 0, img_norm)

    # Resize to size
    img_norm = cv2.resize(img_norm, size)

    return img_norm

def get_hybrid_image(reference, img_a, img_b, size=(32, 32)):

    reference = convert_image(reference, size)
    img_a = convert_image(img_a, size)
    img_b = convert_image(img_b, size)

    hybrid_image = np.dstack([reference, img_a, img_b])

    return hybrid_image
