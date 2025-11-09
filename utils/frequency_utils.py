import numpy as np
import cv2

def image_to_magnitude_spectrum(image):
    """
    Convert an image to its magnitude spectrum using FFT.
    """
    # Convert to float32
    image_float = np.float32(image)
    
    # Perform FFT
    dft = cv2.dft(image_float, flags=cv2.DFT_COMPLEX_OUTPUT)
    
    # Shift the zero frequency component to the center
    dft_shift = np.fft.fftshift(dft)
    
    # Compute magnitude spectrum
    magnitude_spectrum = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
    
    # Log scale for better visualization
    magnitude_spectrum = np.log(magnitude_spectrum + 1)
    
    # Normalize to 0-255
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    return magnitude_spectrum
