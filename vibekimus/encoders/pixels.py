"""Turning images into the pixel tensors the model reads.

open_clip's preprocess is one chain: geometry (resize / crop) on a PIL image, then
ToTensor and Normalize. It is split here so that:
  * CPU workers run only the geometry and hand over small uint8 tensors
    (4x less memory to queue and to copy to the GPU than float tensors),
  * images decoded on the GPU go through the same geometry as tensors,
  * the float conversion and normalisation run once per batch on the GPU.
If the chain does not have the expected shape, the whole chain runs on the CPU
as before and the tensors arrive ready to use.
"""
import numpy as np

UINT8_MAX = 255.0
CHANNELS = 3


class PixelPipeline:
    def __init__(self, preprocess, device):
        import torch
        self._torch = torch
        self._preprocess = preprocess
        self._device = device
        steps = list(getattr(preprocess, "transforms", []))
        names = [type(step).__name__ for step in steps]
        self.split = "ToTensor" in names and names[-1] == "Normalize"
        if self.split:
            to_tensor = names.index("ToTensor")
            self._image_geometry = steps[:to_tensor]
            # Plain functions (e.g. open_clip's convert-to-RGB) only accept PIL images.
            self.tensor_geometry = [step for step in self._image_geometry if isinstance(step, torch.nn.Module)]
            normalize = steps[-1]
            shape = (1, CHANNELS, 1, 1)
            self._mean = torch.tensor(normalize.mean, device=device).view(shape)
            self._std = torch.tensor(normalize.std, device=device).view(shape)
        else:
            self.tensor_geometry = []

    def from_image(self, image):
        """CPU side: one RGB PIL image -> a CHW tensor (uint8 when split)."""
        if not self.split:
            return self._preprocess(image)
        for step in self._image_geometry:
            image = step(image)
        return self._torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).permute(2, 0, 1)

    def batch_on_device(self, tensors, pin_memory):
        """Stack tensors that may live on the CPU or already on the device, keeping
        their order, and return a model-ready float batch on the device."""
        torch = self._torch
        on_cpu = [i for i, tensor in enumerate(tensors) if tensor.device.type == "cpu"]
        if len(on_cpu) == len(tensors):
            batch = torch.stack(tensors)
            batch = batch.pin_memory().to(self._device, non_blocking=True) if pin_memory else batch.to(self._device)
        else:
            batch = torch.empty((len(tensors), *tensors[0].shape), dtype=tensors[0].dtype, device=self._device)
            cpu_positions = set(on_cpu)
            on_device = [i for i in range(len(tensors)) if i not in cpu_positions]
            if on_cpu:
                batch[on_cpu] = torch.stack([tensors[i] for i in on_cpu]).to(self._device)
            batch[on_device] = torch.stack([tensors[i] for i in on_device])
        if batch.dtype == torch.uint8:
            batch = (batch.float() / UINT8_MAX - self._mean) / self._std
        return batch
