import torch
import wand
from wand.image import Image
from viz import to_numpyArray


def to_wandImage(img):
    """
    Tensor to wandImage
    Parameters:
        img: Tensor (C, H, W)
    Returns:
        wandImage: wandImage (H, W, C)
    """
    if torch.is_tensor(img):
        return wand.image.Image.from_array(to_numpyArray(img))
    return wand.image.Image.from_array(img)


def createGifAnim(tensorlist, filename, delay=0):
    """
    creates gif animation of a list of Tensor Images
    Parameters:
        tensorlist: List of Tensors with shape (C, H, W)
        filename: filename with .gif suffix
        delay: 1/100 seconds delay applied to each image
    """
    wandimagelist = [to_wandImage(tensor) for tensor in tensorlist]
    with Image() as seq:
        for img in wandimagelist:
            seq.sequence.append(img)
        if delay:
            for frame in seq.sequence:
                frame.delay = delay
        seq.save(filename=filename)
