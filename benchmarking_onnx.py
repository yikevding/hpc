import onnx
import onnxruntime as rt
import numpy as np
from onnx import numpy_helper
import torch
from get_dataloaders import get_dataloaders
import onnxruntime as ort
import time
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--path", help = "Location from which to load pre-trained model",
                    type=str, default = "../quant/quantized_models/20/mk1.onnx")
parser.add_argument("--batch", help = "Batch size used during inference time measurement",
                    type=int, default = 16)
args = parser.parse_args()


def to_numpy(pt_tensor):
        return pt_tensor.detach().cpu().numpy() if pt_tensor.requires_grad else pt_tensor.cpu().numpy()

def get_size(model):
  onnx_weights = model.graph.initializer
  params = 0
  size_bytes = 0
  for onnx_w in onnx_weights:
    try:
        weight = numpy_helper.to_array(onnx_w)
        for w in weight.ravel():
            if isinstance(w, np.int8):
                size_bytes += 1
            elif isinstance(w, np.float32) or isinstance(w, np.int32):
                size_bytes += 4
            elif isinstance(w, np.int64):
                size_bytes += 8
            else:
                print(type(w))
        params += np.prod(weight.shape)
    except Exception as _:
        pass
  print(f"Model contains {params} parameters and takes up roughly {size_bytes/(1024**2):.3f} MB")

def measure_inference_time(model_path, device, batch = 8):
   if device == "cpu":
       ort_provider = ['CPUExecutionProvider']
   elif device == "cuda":
       ort_provider = ['CUDAExecutionProvider'] 
   ort_sess = ort.InferenceSession(model_path, providers = ort_provider)
   data = np.random.randn(8,3,32,32).astype('float32')
   ort_inputs = {ort_sess.get_inputs()[0].name : data}
   _ = ort_sess.run(None, ort_inputs)[0]
   start = time.time()
   _ = ort_sess.run(None, ort_inputs)[0]
   end = time.time()
   print(f"Inference took {end-start} on {device} with batch size of {batch}")

def validate(model_path, validation, device):
    if device == "cpu":
       ort_provider = ['CPUExecutionProvider']
    elif device == "cuda":
       ort_provider = ['CUDAExecutionProvider'] 
    ort_sess  = ort.InferenceSession(model_path, providers = ort_provider)
    correct_onnx = 0

    for img_batch, label_batch in validation:

      ort_inputs = {ort_sess.get_inputs()[0].name: to_numpy(img_batch)}
      ort_outs = ort_sess.run(None, ort_inputs)[0]

      ort_preds = np.argmax(ort_outs, axis=1)
      correct_onnx += np.sum(np.equal(ort_preds, to_numpy(label_batch)))

    print("\n")

    print(f"Model acc = { correct_onnx/(128*len(validation))} with {correct_onnx} correct samples on {device}")

_, validation = get_dataloaders()
model_onnx = onnx.load(args.path)
print(f"Now benchmarking the model save at {args.path}")
get_size(model_onnx)
print("CPU Measurements:")
measure_inference_time(args.path, "cpu", args.batch)
#validate(args.path,validation, "cpu")
print("GPU Measurements:")
measure_inference_time(args.path, "cuda", args.batch)
validate(args.path, validation, "cuda")


