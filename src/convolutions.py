import cv2
import numpy as np

def ds_convolution(image, kernel, boundary_value=0):
    """
    Performs convolutions with the double summation (ds) formula
    Parameters:
    image: A n x m numpy array containing the RGB values for each pixel in the image
    kernel: A n x m numpy array representing the kernel filter
    boundary_value: The value pixels outside of the image boundaries should take; default to 0.
    Returns:
    An nxm numpy array representing the filtered image
    """

    # Get the image and kernel dimensions
    i_width, i_height = image.shape[0], image.shape[1]
    k_width, k_height = kernel.shape[0], kernel.shape[1]

    # Define the output image
    filtered = np.zeros_like(image)
    kernel_sum = kernel.sum() if kernel.sum() > 0 else 1

    # Iterate each pixel in the image, left to right, top to bottom (x,y)
    for y in range(i_height):
        for x in range(i_width):
            weighted_pixel_sum = 0

            # Iterate over the kernel matrix for each pixel
            for ky in range(-(k_height // 2), (k_height // 2) + 1):
                for kx in range(-(k_width // 2), (k_width // 2) + 1):
                    # Compute image pixel coordinates with respect to the kernel, i.e. the ones under inspection
                    pixel_y = y + ky
                    pixel_x = x + kx

                    # Set default pixel value to passed boundary value
                    pixel = boundary_value
                    # Update pixel value if it lies inside the image boundaries
                    if (pixel_y >= 0) and (pixel_y < i_height) and (pixel_x >= 0) and (pixel_x < i_width):
                        pixel = image[pixel_y, pixel_x]

                    # Transpose local pixel coordinates (-k/2, k/2) back to valid array index (0,k-1)
                    weight = kernel[ky + (k_height // 2), kx + (k_width // 2)]

                    # Add weighted sum of current pixel to total
                    weighted_pixel_sum += pixel * weight

            # Average the collected pixel values
            filtered[y, x] = weighted_pixel_sum
    
    return filtered

def built_in_convolution(image, kernel):
    """
    Performs convolutions with the built in library function
    Parameters:
    image: A n x m numpy array containing the RGB values for each pixel in the image
    kernel: A n x m numpy array representing the kernel filter
    Returns:
    An nxm numpy array representing the filtered image
    """
    return cv2.filter2D(image, -1, kernel)