import torch
import torchvision
from skimage import color
from skimage.exposure import match_histograms
from skimage.util import compare_images


def to_numpyArray(img):
    """
    tensor to numpy array of shape (H, W, C)
    Params:
        img: image as Tensor (C, H, W)
    Returns:
        image as numpy_ndarray (H, W, C)
    """
    return img.permute(1, 2, 0).detach().cpu().numpy()


def diff_cmp(img, img2):
    """
    compute absolute diff of both images in greyscale and create comparison image
    Params:
        img: image as numpy_ndarray (H, W, C)
        img2: image as numpy_ndarray (H, W, C)
    Returns:
        numpy_ndarray (H, W, C)
    """
    return compare_images(color.rgb2gray(img), color.rgb2gray(img2), method='diff')


def checkb_cmp(img, img2, n_tiles):
    """
    "makes tiles of dimension n_tiles that display alternatively the first and the second image" (=from skimage doc)
    in greyscale.

    Params:
        img: image as numpy_ndarray (H, W, C)
        img2: image as numpy_ndarray (H, W, C)
    Returns:
        numpy_ndarray (H, W, C)
    """
    return compare_images(color.rgb2gray(img), color.rgb2gray(img2), method='checkerboard', n_tiles=n_tiles)


def vizGridWithCheckboard(fixed, moving, registered):
    """
    make torchvision image grid with [fixed, moving, registered, fixDiffMov, fixDiffReg,fixCheckBMov, fixCheckBReg ]
    Params:
        fixed: image as numpy_ndarray (H, W, C)
        moving: image as numpy_ndarray (H, W, C)
        registrated: image as numpy_ndarray (H, W, C)
    Returns:
        grid of images
    """
    def tt(img):
        # helper to 1 channel grey image to 3 channel grey image
        tmp = torch.from_numpy(img).view(1, len(img), len(img))
        return torch.cat([tmp, tmp, tmp])

    reg = to_numpyArray(registered[0])
    fix = to_numpyArray(fixed[0])
    mov = to_numpyArray(moving[0])
    n_tiles = [50, 50]

    reg = match_histograms(reg, fix, channel_axis=-1)
    mov = match_histograms(mov, fix, channel_axis=-1)  # match histogram for better visualization

    # create diff image
    diff = diff_cmp(fix, mov)
    diff2 = diff_cmp(fix, reg)
    #create checkerboard image
    checkboard = checkb_cmp(fix, mov, n_tiles=n_tiles)
    checkboard2 = checkb_cmp(fix, reg, n_tiles=n_tiles)
    
    return torchvision.utils.make_grid([
        fixed[0].cpu(), moving[0].cpu(), registered[0].cpu(),
        tt(diff), tt(diff2),
        tt(checkboard), tt(checkboard2)
        ])