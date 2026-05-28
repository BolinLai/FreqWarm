"""
This module implements the core idea of FreqWarm: filtering high-frequency
components from images before encoding them into the latent space.

Pipeline:
    1. Low-pass filter the input images to remove high-frequency content.
    2. Encode the filtered images with a pretrained autoencoder (e.g., DC-AE).
    3. Train a diffusion model on the resulting latent representations.

By removing high-frequency detail from the input images, the autoencoder
produces smoother latents that are easier for the diffusion model to learn
during the warm-up stage.
"""

import torch


# ---------------------------------------------------------------------------
# Frequency mask construction
# ---------------------------------------------------------------------------

def make_frequency_mask(H: int, W: int, cutoff: float, filter_type: str = "ideal",
                        order: int = 2, device: str = "cpu") -> torch.Tensor:
    """
    Create a 2D low-pass frequency mask.

    Args:
        H, W: Spatial dimensions of the image.
        cutoff: Normalized cutoff frequency.
        filter_type: One of "ideal" (hard threshold), "butterworth" (smooth
                     roll-off controlled by `order`), or "gaussian" (soft).
        order: Order parameter for Butterworth filter.
        device: Target device.

    Returns:
        A (H, W) float tensor with values in [0, 1] representing the low-pass
        transfer function.
    """
    # Compute sample frequencies (cycles per pixel) along each axis
    fy = torch.fft.fftfreq(H, d=1.0, device=device)  # shape (H,)
    fx = torch.fft.fftfreq(W, d=1.0, device=device)  # shape (W,)

    # Shift so that zero-frequency (DC) is at the center
    fy = torch.fft.fftshift(fy)
    fx = torch.fft.fftshift(fx)

    # Create 2D grid of radial frequencies
    u, v = torch.meshgrid(fy, fx, indexing="ij")  # shape (H, W)
    radius = torch.sqrt(u ** 2 + v ** 2)

    if filter_type == "ideal":
        # Hard cutoff: pass everything below threshold, block everything above
        mask = (radius <= cutoff).float()
    elif filter_type == "butterworth":
        # Smooth roll-off; higher order → sharper transition at cutoff
        mask = 1.0 / (1.0 + (radius / cutoff) ** (2 * order))
    elif filter_type == "gaussian":
        # Gradual Gaussian attenuation centered at DC
        mask = torch.exp(-0.5 * (radius / cutoff) ** 2)
    else:
        raise ValueError(f"Unknown filter_type: {filter_type}")

    return mask


# ---------------------------------------------------------------------------
# Core API: low-pass filter images to remove high-frequency components
# ---------------------------------------------------------------------------

def apply_frequency_filter(batch: torch.Tensor, keep: str = "low",
                           cutoff: float = 0.2, filter_type: str = "ideal",
                           order: int = 2) -> torch.Tensor:
    """
    Apply a frequency-domain filter to a batch of images.

    This is the core operation of FreqWarm. It is applied to input images
    BEFORE encoding them with the autoencoder. By discarding high-frequency
    content, the encoder produces smoother latent representations that the
    diffusion model can learn more easily during the warm-up stage.

    The operation per channel:
        1. Compute the 2D FFT of the spatial slice.
        2. Shift zero-frequency (DC) to center.
        3. Multiply by the frequency mask to zero out unwanted components.
        4. Shift back and compute inverse FFT to return to spatial domain.

    Args:
        batch: Image tensor, shape (B, C, H, W) with values in [-1, 1].
        keep: "low" to keep low frequencies (discard high), or "high" to keep
              high frequencies (discard low).
        cutoff: Normalized cutoff frequency. Lower values mean
                more aggressive filtering. Typical value: 0.2.
        filter_type: "ideal" (hard cutoff), "butterworth" (smooth roll-off),
                     or "gaussian" (soft attenuation).
        order: Butterworth filter order (ignored for other types).

    Returns:
        Filtered image tensor with the same shape (B, C, H, W).
    """
    B, C, H, W = batch.shape
    device = batch.device

    # Build the low-pass mask
    mask = make_frequency_mask(H, W, cutoff, filter_type, order, device)

    # Invert for high-pass if needed
    if keep == "high":
        mask = 1.0 - mask

    # Prepare output tensor
    output = torch.empty_like(batch)

    # Apply filter channel-by-channel via FFT
    for b in range(B):
        for c in range(C):
            img = batch[b, c]  # shape (H, W)

            # Transform to frequency domain
            F = torch.fft.fft2(img)
            F = torch.fft.fftshift(F)  # move DC to center

            # Zero out frequencies outside the passband
            F_filtered = F * mask

            # Transform back to spatial domain
            F_filtered = torch.fft.ifftshift(F_filtered)
            img_filtered = torch.fft.ifft2(F_filtered).real

            output[b, c] = img_filtered

    return output


# ---------------------------------------------------------------------------
# Standalone usage example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Simulate a batch of images (e.g., 512x512 RGB images normalized to [-1, 1])
    B, C, H, W = 2, 3, 512, 512
    images = torch.randn(B, C, H, W)

    print(f"Input image shape: {images.shape}")
    print(f"Input image std:   {images.std():.4f}")

    # Low-pass filter the images (remove high-frequency details)
    cutoff = 0.20
    filtered_images = apply_frequency_filter(images, keep="low", cutoff=cutoff, filter_type="ideal")

    print(f"\nAfter low-pass filtering (cutoff={cutoff}):")
    print(f"  Shape: {filtered_images.shape}")
    print(f"  Std:   {filtered_images.std():.4f}  (reduced — high-freq energy removed)")
