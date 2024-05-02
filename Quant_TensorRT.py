import torch
import time
from tqdm import tqdm
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import argparse
import pycuda.driver as cuda
# This import causes pycuda to automatically manage CUDA context creation and cleanup.
import pycuda.autoinit

import tensorrt as trt

from trt.utils import build_engine_onnx, build_engine_onnx_int8
from trt import common
from trt.calibrator import ResNetEntropyCalibrator
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--load_from", help = "Path to ONNX model",
                    type=str)
parser.add_argument("--batch_size", help = "Batch size for TensorRT engine (must be fixed, no partial batches)",
                    type = int, default = 64)
args = parser.parse_args()


class ModelData(object):
    INPUT_SHAPE = (3, 32, 32)
    # We can convert TensorRT data types to numpy types with trt.nptype()
    DTYPE = trt.float32


def quantization_main(onnx_model_file, batch_size):
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

    inference_times = 100
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        ])
    #dataset_test = Imagenet1k('./datasets')
    train  = CIFAR10(root = "../CIFAR", train=True, download=True, transform=transform)
    val  = CIFAR10(root = "../CIFAR", train=False, download=True, transform=transform)
    train = torch.utils.data.Subset(train, list(range(5000)))
    val_loader = validation = DataLoader(val, batch_size=batch_size, shuffle = True)
    # ==> trt test

    # ==> trt int8 quantization test
    calibration_cache = './tensorRT/modelInt8.engine'
    training_data = './datasets'
    # get the calibrator for int8 post-quantization
    calib = ResNetEntropyCalibrator(training_data=training_data, cache_file=calibration_cache)

    with build_engine_onnx_int8(TRT_LOGGER, onnx_model_file, calib, batch_size) as engine:

        # Inference is the same regardless of which parser is used to build the engine, since the model architecture is the same.
        # Allocate buffers and create a CUDA stream.
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        
        # Contexts are used to perform inference.
        with engine.create_execution_context() as context:

            # Load a normalized test case into the host input page-locked buffer.
            #dataset_test.get_one_image_trt(inputs[0].host, idx=0)
            imgs = np.zeros((batch_size,3,32,32), dtype=np.float32)
            trgts = np.zeros((batch_size))
            for i in range(batch_size):
                img, trgt = train.__getitem__(i)
                imgs[i,:,:,:] = img.numpy()
                trgts[i] = trgt            
            imgs = np.ravel(imgs)
            np.copyto(inputs[0].host, imgs)
            # Run the engine. The output will be a 1D tensor of length 1000, where each value represents the
            # probability that the image corresponds to that label
            t_begin = time.time()
            for i in tqdm(range(inference_times)):
                trt_int8_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            t_end = time.time()
            trt_int8_time = (t_end - t_begin)/inference_times       
            correct = 0
            tot = 0
            for imgs, targets in val_loader:
                #cannot handle non-constant batch size
                if len(imgs) < 64:
                    break
                imgs = imgs.numpy()
                imgs = np.ravel(imgs)
                np.copyto(inputs[0].host, imgs)
                trt_outputs = common.do_inference(context,bindings=bindings,inputs=inputs,outputs=outputs,stream=stream)
                correct += np.sum(np.where(targets.numpy() == np.argmax(np.reshape(np.array(trt_outputs), (batch_size,10)), axis = 1), 1, 0))
                tot += 64
            print(f"Inference Time: {trt_int8_time:.5f}")
            print(f"Validation Accuracy: {correct/tot:.3f}")

        with open('./model/modelInt8.engine', "wb") as f:
            f.write(engine.serialize())

    #print('==> Torch time: {:.5f} ms'.format(torch_time))
if __name__ =='__main__': 
    quantization_main(args.load_from, args.batch_size)