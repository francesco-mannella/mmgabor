# %%

import glob

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve


# %% GABOR FILTER FUNCTION
def gabor_kernel(
    frequency, orientation, sigma, sigma_y=None, phase_offset=0, size=5
):
    """
    Generate a Gabor filter.

    Args:
    - frequency (float): Spatial frequency of the harmonics.
    - orientation (float): Orientation of the Gabor filter in radians.
    - sigma (float): Standard deviation of the Gaussian envelope.
    - sigma_y (float): Standard deviation of the Gaussian envelope in the 2nd
      dimension.
    - phase_offset (float): Phase offset of the sine wave.
    - size (int): Size of the filter.

    Returns:
    - np.ndarray: The generated Gabor filter.
    """

    if sigma_y is None:
        sigma_y = sigma

    half_size = size // 2
    x_grid, y_grid = np.ogrid[
        -half_size : (half_size + 1), -half_size : (half_size + 1)
    ]
    rotated_x = x_grid * np.cos(orientation) + y_grid * np.sin(orientation)
    rotated_y = -x_grid * np.sin(orientation) + y_grid * np.cos(orientation)
    gabor = np.exp(
        -(rotated_x**2 / (2 * sigma**2) + rotated_y**2 / (2 * sigma_y**2))
    ) * np.cos(2 * np.pi * frequency * rotated_x + phase_offset)

    # Normalize the Gabor filter by dividing it by the sum of its non-negative
    # elements
    non_negative_sum = np.sum(np.maximum(gabor, 0))
    gabor /= non_negative_sum

    return gabor


#####
def apply_channel_gabor_filters(
    image,
    scale_list,
    orientation_list,
    frequency,
    phase_offset,
    kernel_size=21,
    filter_slope=0.8,
    sigma_y_multiplier=5,
    bw_channel_ratio=0.2,
):
    """Apply multi-scale, multi-orientation channel-opponent Gabor filters.

    This function processes an RGB image by applying Gabor filters at various
    scales and orientations, using specially constructed channel-opponent masks
    for comparative color feature detection. The image is filtered through
    masks that isolate and contrast the red, green, and blue channels, as well
    as provide uniform and inverted uniform responses, enabling assessment of
    color- and brightness-dependent spatial features.

    Output channels:
        - Channel 0: Salience of red features relative to green.
        - Channel 1: Salience of green features relative to red.
        - Channel 2: Salience of blue features relative to red and green.
        - Channel 3: Composite channel representing overall brightness
          (uniform response to all channels, normalized).

    Args:
        image (np.ndarray): Input image of shape (H, W, 3), channel order RGB.
        scale_list (list[float]): List of scales (sigma values) for Gabor
            kernels.
        orientation_list (list[float]): List of filter orientations in radians.
        frequency (float): Spatial frequency parameter for Gabor kernels.
        phase_offset (float): Phase offset for Gabor kernels, in radians.
        kernel_size (int): Width/height of square Gabor kernels.
        filter_slope (float): Exponential slope for feature nonlinearity.
        sigma_y_multiplier (float): Elongation factor for the kernel y-axis.
        bw_channel_ratio (float): Fraction of output energy for brightness
            channel.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Filtered RGB channels of shape (H, W, 3), normalized
              to [0, 1].
            - np.ndarray: Brightness channel of shape (H, W), normalized to
              [0, 1].
    """

    # Defines linear combinations of RGB channels to create masks for
    # highlighting or suppressing specific colors in each processing pass.
    mask_channel_weights = (
        # Emphasizes red over green; suppresses green
        (1, -1, 0),
        # Emphasizes green over red; suppresses red
        (-1, 1, 0),
        # Highlights blue against red and green; suppresses red and green
        (-0.5, -0.5, 1),
        # Neutral mix of all color channels
        (0.3, 0.3, 0.3),
        # Inverted mix for detecting dark saliences
        (-0.3, -0.3, -0.3),
    )

    # Number of multi-channel mask types used
    num_masks = len(mask_channel_weights)

    # Prepare output array matching input shape, with an extra dimension
    shape = list(image.shape)
    shape[-1] += 1  # Additional 'brightness' channel
    output = np.zeros(shape)

    if len(image.shape) > 3:
        image = image[:, :, :3]

    # Iterate over all channel-comparison masks
    for weights in mask_channel_weights:
        # Combine color channels using current mask for color-opponent
        # filtering
        masked = image @ weights

        # Apply nonlinear transformation to enhance proactive feature responses
        masked = np.exp(-(filter_slope**-2) * (masked - 1) ** 2)

        # Apply Gabor filtering over all spatial scales and orientations
        for sigma in scale_list:
            for theta in orientation_list:
                # --- Gabor kernel construction for feature extraction ---
                kernel = gabor_kernel(
                    size=kernel_size,
                    sigma=sigma,
                    sigma_y=sigma * sigma_y_multiplier,
                    orientation=theta,
                    frequency=frequency,
                    phase_offset=phase_offset,
                )
                # --- Channel-specific convolution with spatial kernel ---
                filtered = np.abs(convolve(masked, kernel, mode="nearest"))

                # If weights are not balanced, add to specific color channel
                if not all(x == weights[0] for x in weights):
                    # Determine the strongest color channel for this mask
                    output[:, :, np.argmax(weights)] += filtered
                else:
                    # If mask is uniform, treat as brightness and distribute
                    output[:, :, -1] += bw_channel_ratio * filtered / num_masks

    # Normalize entire output to the [0, 1] interval, preserving structure
    output = (output - output.min()) / (output.max() - output.min())

    filtered_rgb, brightness = output[:, :, :3], output[:, :, 3]
    return filtered_rgb, brightness


#####


if __name__ == "__main__":
    # """
    # Demo: Visualizes the effect of multi-scale, multi-orientation Gabor
    # filtering on a set of images. For each image, applies channel-wise Gabor
    # filters and displays the original, filtered channels, and various
    # visualizations of the filter responses for qualitative analysis.
    # """

    # Collect all jpg image file paths from the specified directory
    image_files = glob.glob("photos_no_class/*jpg")
    # image_files = glob.glob("base_imgs/*jpg")

    # Shuffle the image file list for random demo order
    np.random.shuffle(image_files)

    # Process each image in the shuffled list
    for image_file in image_files:

        # Load image and ensure only RGB channels are used
        image = plt.imread(image_file)
        if image.shape[-1] > 3:
            image = image[:, :, :3]
        image = image.astype(float) / 255.0

        # Define Gabor filter parameters
        scales = [1, 10, 20]
        orientations = np.pi * np.linspace(0, 360, 10) / 180.0
        frequency = 0.006
        phase_offset = -np.pi * (0.5 - 9e-4)
        kernel_size = 5
        filter_slope = 0.8
        sigma_y_multiplier = 6
        bw_channel_ratio = 2.0

        # Apply channel-wise Gabor filters to the image
        rgb, brightness = apply_channel_gabor_filters(
            image,
            scales,
            orientations,
            frequency,
            phase_offset,
            kernel_size,
            filter_slope=filter_slope,
            sigma_y_multiplier=sigma_y_multiplier,
            bw_channel_ratio=bw_channel_ratio,
        )

        # Create a 3x4 grid of subplots for visualization
        plt.close("all")
        fig, axes = plt.subplots(3, 4, figsize=(12, 6))
        axes = axes.ravel()

        # Hide axes for cleaner visualization
        for ax in axes:
            ax.set_axis_off()
        fig.tight_layout(pad=0.2)

        # Display original image and filtered channels
        axes[0].imshow(image)
        axes[1].imshow(rgb[:, :, 0], vmin=0, vmax=1, cmap=plt.cm.gray)
        axes[2].imshow(rgb[:, :, 1], vmin=0, vmax=1, cmap=plt.cm.gray)
        axes[3].imshow(rgb[:, :, 2], vmin=0, vmax=1, cmap=plt.cm.gray)
        axes[4].imshow(brightness, vmin=0, vmax=1, cmap=plt.cm.gray)

        # Combine RGB and brightness channels for enhanced visualization
        brightness_expanded = np.expand_dims(brightness, -1)
        adjusted_rgb = (rgb * 0.7) + (brightness_expanded * 0.3)
        adjusted_rgb /= adjusted_rgb.max()
        axes[5].imshow(np.clip(adjusted_rgb, 0, 1))

        # Visualize the maximum response across RGB channels plus brightness
        overall_brightness = rgb[:, :, :3].max(-1) + brightness
        overall_brightness /= overall_brightness.max()
        axes[6].imshow(
            overall_brightness,
            vmin=0,
            vmax=1,
            cmap=plt.cm.gray,
        )

        # Show the figure with all visualizations
        plt.show()
