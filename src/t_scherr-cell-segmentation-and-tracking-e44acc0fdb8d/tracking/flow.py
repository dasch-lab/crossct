"""Estimate shift between image crops."""
import numpy as np
from scipy.fftpack import fftn, ifftn


def compute_fft_displacement(img1, img2):
    """
    Estimates shift between two images via phase correlation.
    Args:
        img1: a np.array representing an image crop
        img2: a np.array representing an image crop

    Returns: a tuple containing the estimated shift between the images in pixels.

    """

    # filter images to suppress image border information
    img_filter = [np.hanning(s) for s in img1.shape]
    if len(img1.shape) == 2:
        img_filter = img_filter[0].reshape(-1, 1) * img_filter[1].reshape(1, -1)
    elif len(img1.shape) == 3:
        img_filter = img_filter[0].reshape(-1, 1, 1) * img_filter[1].reshape(1, -1, 1) * \
                     img_filter[2].reshape(1, 1, -1)

    # compute phase correlation
    fft1 = fftn(img1*img_filter)
    fft2 = fftn(img2*img_filter)
    quotient = np.conj(fft1)*fft2 / np.abs(np.conj(fft1)*fft2+1e-12)
    correlation = ifftn(quotient)

    # finding the peak in the phase correlation and computing the image shift in pixels
    peak = np.unravel_index(np.argmax(np.abs(correlation)), correlation.shape)
    peak = np.array(peak)
    # distinct positive/negative shift due to circularity of fft
    # positive shift: displacement==shift, for peak 0...img1.shape//2
    # negative shift: displacement=shape-shift, for img1.shape//2 +1 ... img1.shape
    negative_shift = peak > np.array(img1.shape) // 2
    displacement = peak
    displacement[negative_shift] = -(np.array(img1.shape) - peak)[negative_shift]
    return displacement
