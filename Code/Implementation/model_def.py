"""
VoxelMorph Determined Training

In the `__init__` method, the model and optimizer are wrapped with `wrap_model`
and `wrap_optimizer`. This model is single-input and single-output.

The methods `train_batch` and `evaluate_batch` define the forward pass
for training and evaluation respectively.
"""

from typing import Any, Dict, Sequence, Tuple, Union, cast

import torch
from determined.pytorch import DataLoader, PyTorchTrial, PyTorchTrialContext
from determined.tensorboard.metric_writers.pytorch import TorchWriter
from torch import nn
from torchvision import transforms

# MyFiles
import data
import losses
import network
import viz

TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]


class VoxelMorphTrial(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext) -> None:
        self.context = context
        self.logger = TorchWriter()
        self.dim = self.context.get_hparam("image_dim")  # for resizing to (dim, dim)

        # transformation initialization
        transformations = [
            transforms.Resize((self.dim, self.dim)),
            transforms.ToTensor(),
        ]

        if self.context.get_data_config()['is_grey']:
            transformations = [
                transforms.Resize((self.dim, self.dim)),
                transforms.Grayscale(),
                transforms.ToTensor(),
            ]

        self.transform = transforms.Compose(transformations)

        # model definition
        net = network.VoxelMorph(imagedim=(self.dim, self.dim), isgrey=False)

        # loss initialization
        self.loss_sim = nn.MSELoss()
        self.loss_smooth = losses.Grad('l2', loss_mult=2).loss
        self.smooth_weight = self.context.get_hparam("grad_loss_weight")

        # determined wrapping of model and optimizer
        self.model = self.context.wrap_model(net)
        self.optimizer = self.context.wrap_optimizer(
            torch.optim.Adam(self.model.parameters(), lr=self.context.get_hparam("learning_rate"))
        )

    def build_training_data_loader(self) -> DataLoader:
        dataset = data.HistologyDataset(csv_file=self.context.get_data_config()['filePath_train'],
                                        transform_he=self.transform,
                                        transform_phh3=self.transform, swap=self.context.get_data_config()['swap'])
        return DataLoader(dataset, batch_size=self.context.get_per_slot_batch_size())

    def build_validation_data_loader(self) -> DataLoader:
        dataset = data.HistologyDataset(csv_file=self.context.get_data_config()['filePath_val'],
                                        transform_he=self.transform,
                                        transform_phh3=self.transform, swap=self.context.get_data_config()['swap'])
        return DataLoader(dataset, batch_size=self.context.get_per_slot_batch_size())

    def train_batch(
            self, batch: TorchData, epoch_idx: int, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        batch = cast(Tuple[torch.Tensor, torch.Tensor], batch)
        inputs, outputs = batch  # inputs = [moved, fixed] outputs=[ground_truth(=fixed)]

        # forward
        registered, displacement_field = self.model(*inputs)

        # loss
        loss_sim = self.loss_sim(registered, outputs[0])
        loss_smooth = self.loss_smooth(displacement_field)
        loss = loss_sim + loss_smooth * self.smooth_weight

        # backward + optimize
        self.context.backward(loss)
        self.context.step_optimizer(self.optimizer)

        # Log sampled images to Tensorboard each 100 batches.
        if batch_idx % 1000 == 0:
            with torch.no_grad():
                grid = viz.vizGridWithCheckboard(fixed=inputs[1], moving=inputs[0], registered=registered)
                self.logger.writer.add_image(f'generated_images_epoch_{epoch_idx}', grid, batch_idx)
        return {"loss": loss, "loss_sim": loss_sim, "loss_smooth": loss_smooth}

    def evaluate_batch(self, batch: TorchData) -> Dict[str, Any]:
        batch = cast(Tuple[torch.Tensor, torch.Tensor], batch)
        inputs, _ = batch

        registered, _ = self.model(*inputs)  # returns [registered , displacement_field]

        mse_loss = torch.nn.functional.mse_loss(inputs[1], registered)
        l1_loss = torch.nn.functional.l1_loss(inputs[1], registered)

        # validate over grey images
        registered_grey = transforms.functional.rgb_to_grayscale(registered)
        fix_grey = transforms.functional.rgb_to_grayscale(inputs[1])

        mse_loss_grey = torch.nn.functional.mse_loss(fix_grey, registered_grey)
        l1_loss_grey = torch.nn.functional.l1_loss(fix_grey, registered_grey)

        return {"mse_loss": mse_loss, "mse_loss_grey": mse_loss_grey, "l1_loss": l1_loss, "l1_loss_grey": l1_loss_grey}
        