from .process_polygon import construct_sector
import numpy as np
from shapely.affinity import rotate
import rasterio.features as rfeat
import torch

def construct_polar_kernels(radii: list[int], num_sectors: int) -> np.ndarray:
    """ Construct list of filters corresponding to each sector and radius. This will be used to convolve the original dbz map. """
    max_radii = radii[-1]
    origin = (max_radii, max_radii)

    kernels_lst = []
    prev_base_sector = None

    # Idea: travel through each radius, construct the sector for that radius by taking the 
    for r_idx, radius in enumerate(radii):
        new_sector = construct_sector(origin, radius, 0, 360/num_sectors)
        base_sector = new_sector if not prev_base_sector else new_sector.difference(prev_base_sector)
        
        prev_base_sector = new_sector
        
        for s_idx in range(num_sectors):
            sector = rotate(base_sector, s_idx * 360 / num_sectors, origin=origin)
            mask = rfeat.rasterize(shapes=[(sector, 1)], out_shape=(max_radii*2, max_radii*2), fill=0, masked=True, dtype=np.int8)

            kernels_lst.append(mask)

    return np.array(kernels_lst)

def fft_conv2d(img, kernel):
    """
    Performs 2D convolution via FFT to avoid 'im2col' memory explosion 
    with large kernels.
    
    img: (Batch, C, H, W)
    kernel: (Out_C, In_C, KH, KW)
    """
    b, c, h, w = img.shape
    out_c, in_c, kh, kw = kernel.shape

    # 1. Pad dimensions to avoid circular convolution artifacts
    # We pad to (H + KH - 1)
    padded_h = h + kh - 1
    padded_w = w + kw - 1

    # 2. Compute FFT of the Image and the Kernel
    # PyTorch automatically handles padding in the fft2 function if 's' is provided
    # rfft2 is faster for real-valued inputs (exploits symmetry)
    img_fft = torch.fft.rfft2(img, s=(padded_h, padded_w))
    ker_fft = torch.fft.rfft2(kernel, s=(padded_h, padded_w))

    # 3. Multiply in Frequency Domain
    # Broadcasting happens here: 
    # img: (1, 1, H_freq, W_freq)
    # ker: (N, 1, H_freq, W_freq)
    # out: (N, 1, H_freq, W_freq)
    # Note: We sum over the input channel dimension if In_C > 1, but here it's 1.
    res_fft = img_fft * ker_fft

    # 4. Inverse FFT to get back to Spatial Domain
    res = torch.fft.irfft2(res_fft, s=(padded_h, padded_w))

    # 5. Crop to original size (mimicking padding='same')
    # The result starts at 0. We need to center-crop it.
    start_h = (kh - 1) // 2
    start_w = (kw - 1) // 2
    
    sectors_convolved = res[:, :, start_h : start_h + h, start_w : start_w + w]
    return sectors_convolved.squeeze(0).squeeze(1).numpy()