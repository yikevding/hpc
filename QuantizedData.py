import torch
import numpy as np
from onnxruntime import quantization

def to_numpy(pt_tensor):
        return pt_tensor.detach().cpu().numpy() if pt_tensor.requires_grad else pt_tensor.cpu().numpy()

class QuantizationDataReader(quantization.CalibrationDataReader):
    def __init__(self, torch_ds, batch_size, input_name):

        self.torch_dl = torch.utils.data.DataLoader(torch_ds, batch_size=batch_size, shuffle=False)

        self.input_name = input_name
        self.datasize = len(self.torch_dl)

        self.enum_data = iter(self.torch_dl)


    def get_next(self):
        batch = next(self.enum_data, None)
        if batch is not None:
          return {self.input_name: to_numpy(batch[0])}
        else:
          return None

    def rewind(self):
        self.enum_data = iter(self.torch_dl)