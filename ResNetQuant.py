import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import time
import argparse

import onnx
import onnxruntime as ort
from onnxruntime import quantization
import numpy as np

from QuantizedData import QuantizationDataReader, to_numpy
import sys
sys.path.append("..")
from baselines import resnet

parser = argparse.ArgumentParser()
parser.add_argument("--load_from", help = "Location from which to load pre-trained model",
                    type=str, default = "../baselines/fine-tuned/")
parser.add_argument("--val", help = "Specific ResNet Architecture (e.g resnet20)",
                    type=int, default = 20)
parser.add_argument("--save_fp32_to", help="Location to save the fp32 model in ONNX format",
                    type = str, default="../baselines/fp32-onnx/")
parser.add_argument("--save_preprocessed_to", help="Location to save the preprocessed model in ONNX format",
                    type = str, default="preprocessed_models/")
parser.add_argument("--save_quant_to", help="Location to save the quantized model in ONNX format",
                    type = str, default="quantized_models/")
parser.add_argument("--name", help = "Experiment name",
                    type=str)
args = parser.parse_args()

#load our model
if args.val == 20:
    model_fp32 = resnet.resnet20()
elif args.val == 32:
    model_fp32 = resnet.resnet32()
elif args.val == 44:
    model_fp32 = resnet.resnet44()
elif args.val == 56:
    model_fp32 = resnet.resnet56()
elif args.val == 110:
    model_fp32 = resnet.resnet110()
else:
    raise Exception("Please choose a supported model type")
load_from = args.load_from + f"resnet{args.val}-B.th"
model_fp32.load_state_dict(torch.load(load_from)['state_dict'])

#set up our pytorch dataset, we will need these to calibrate our quantized models and perform inference
transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        ])
train  = CIFAR10(root = "../CIFAR", train=True, download=True, transform=transform)
val = CIFAR10(root = "../CIFAR", train = False, download = True,transform = transform )
validation = DataLoader(val, batch_size=128, shuffle = True)

#load the model and convert to Onnx format (this is only if the model we are quantizing is in PyTorch format, can refactor once we implement for Onnx models)

model_fp32.eval()
model_fp32_path = args.save_fp32_to + str(args.val) + "/" + args.name + ".onnx"
#model_fp32_path = '../baselines/fp32-onnx/resnet20_fp32.onnx'

dummy_in = torch.randn(1,3,32,32, requires_grad = True)
torch.onnx.export(model_fp32,                                       # model
                  dummy_in,                                         # model input
                  model_fp32_path,                                  # path
                  export_params=True,                               # store the trained parameter weights inside the model file
                  opset_version=14,                                 # the ONNX version to export the model to
                  do_constant_folding=False,                        # constant folding for optimization
                  input_names = ['input'],                          # input names
                  output_names = ['output'],                        # output names
                  dynamic_axes={'input' : {0 : 'batch_size'},       # variable length axes
                                'output' : {0 : 'batch_size'}})


#we should pre-process our onnx models, according to the docs.
model_prep_path = 'resnet20_prep.onnx'
model_prep_path = args.save_preprocessed_to + str(args.val) + "/" + args.name + ".onnx"
quantization.shape_inference.quant_pre_process(model_fp32_path, model_prep_path, skip_symbolic_shape=False)

ort_provider = ['CUDAExecutionProvider']
ort_sess = ort.InferenceSession(model_fp32_path, providers=ort_provider)

qdr = QuantizationDataReader(train, batch_size=64, input_name=ort_sess.get_inputs()[0].name)

q_static_opts = {"ActivationSymmetric":True,
                  "WeightSymmetric":True}

model_int8_path = 'resnet20_int8.onnx'
model_int8_path = args.save_quant_to + str(args.val) + "/" + args.name + ".onnx"
quantized_model = quantization.quantize_static(model_input=model_prep_path,
                                               model_output=model_int8_path,
                                               calibration_data_reader=qdr,
                                               extra_options=q_static_opts)
ort_int8_sess = ort.InferenceSession(model_int8_path, providers=ort_provider)
correct_int8 = 0
correct_onnx = 0
tot_abs_error = 0

for img_batch, label_batch in validation:

  ort_inputs = {ort_sess.get_inputs()[0].name: to_numpy(img_batch)}
  ort_outs = ort_sess.run(None, ort_inputs)[0]

  ort_preds = np.argmax(ort_outs, axis=1)
  correct_onnx += np.sum(np.equal(ort_preds, to_numpy(label_batch)))


  ort_int8_outs = ort_int8_sess.run(None, ort_inputs)[0]

  ort_int8_preds = np.argmax(ort_int8_outs, axis=1)
  correct_int8 += np.sum(np.equal(ort_int8_preds, to_numpy(label_batch)))

  tot_abs_error += np.sum(np.abs(ort_int8_outs - ort_outs))


print("\n")

print(f"onnx top-1 acc = {correct_onnx/(len(validation)*128)} with {correct_onnx} correct samples")
print(f"onnx int8 top-1 acc = {100.0 * correct_int8/(128*len(validation))} with {correct_int8} correct samples")

mae = tot_abs_error/(1000*len(val))
print(f"mean abs error = {mae} with total abs error {tot_abs_error}")












'''
model_fp32.qconfig = torch.ao.quantization.get_default_qconfig('x86')

#model_fp32.fuse_model()

model_fp32_prepared = torch.ao.quantization.prepare(model_fp32)

for i, (input, _) in enumerate(loader):
    if i > 5:
        break
    model_fp32_prepared(input)

model_int8 = torch.ao.quantization.convert(model_fp32_prepared)
model_fp32 = resnet.resnet20()
model_fp32.load_state_dict(torch.load("save_temp/model.th")['state_dict'])


count_int8 = 0
count_fp32 = 0
n = 0
for input, target in validation:
    pred_int8 = model_int8(input)
    pred_fp32 = model_fp32(input)

    count_int8 += torch.sum(torch.argmax(pred_int8, dim = 1) == target)
    count_fp32 += torch.sum(torch.argmax(pred_fp32,dim=1) == target)
    n += len(input)
print(f'Accuracy of Quantized Model: {count_int8/n}')
print(f'Accuracy of Full-precision Model: {count_fp32/n}')

torch.save(model_int8.state_dict(), 'save_temp/resnet20_int8.pt')

'''