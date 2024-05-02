import torch
import torch.nn as nn
import torch.nn.functional as F

from nni.compression.utils import auto_set_denpendency_group_ids
from nni.compression import pruning
from nni.compression.speedup import ModelSpeedup

from finetune import train, validate
import sys
sys.path.append("..")
from baselines import resnet 
from utils import benchmarking, get_dataloaders
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--s", help="Level of sparsity in pruning",
                    type=float, default = .4)
parser.add_argument("--load_from", help = "Location from which to load pre-trained model",
                    type=str, default = "../baselines/resnet20.th")
parser.add_argument("--val", help = "Specific ResNet Architecture (e.g resnet20)",
                    type=int, default = 20)
parser.add_argument("--save_to", help="Location to save the pruned model in ONNX format",
                    type = str, default="pruned_models/")
parser.add_argument("--name", help = "Experiment name",
                    type=str)
args = parser.parse_args()


training, validation = get_dataloaders.get_dataloaders()
if args.val == 20:
    model = resnet.resnet20()
    model_unpruned = resnet.resnet20()
elif args.val == 32:
    model = resnet.resnet32()
    model_unpruned = resnet.resnet32()
elif args.val == 44:
    model = resnet.resnet44()
    model_unpruned = resnet.resnet44()
elif args.val == 56:
    model = resnet.resnet56()
    model_unpruned = resnet.resnet56()
elif args.val == 110:
    model = resnet.resnet110()
    model_unpruned = resnet.resnet110()
else:
    raise Exception("Please choose a supported model type")

#state_dict = get_state_dict(args.load_from, args.val)
state_dict = torch.load(args.load_from)
model.load_state_dict(state_dict)
model_unpruned.load_state_dict(state_dict)
config_list =[{
    'op_types' : ['Conv2d'],
    'sparse_ratio' : args.s,
}]
dummy_input = torch.randn(8,3,32,32)
config_list = auto_set_denpendency_group_ids(model, config_list, dummy_input)
pruner = pruning.FPGMPruner(model, config_list )
_, masks = pruner.compress()


pruner.unwrap_model()
ModelSpeedup(model, torch.rand(3, 3, 32, 32), masks).speedup_model()
model.to("cuda")

#Fine-tune our pruned model:
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4)
criterion = nn.CrossEntropyLoss().cuda()
for i in range(10):
    train(training, model, criterion, optimizer, i)
    acc = validate(validation, model, criterion)
    print(f'Validation accuracy @ epoch {i}: {acc}')

model.eval()
model_unpruned.eval()
print("############################################")
print("Uncompressed Model:")
benchmarking.calculate_model_size(model_unpruned)
print(f"CPU inference Time {benchmarking.inference_time(model_unpruned, 'cpu')}")
print(f"GPU inference Time {benchmarking.inference_time(model_unpruned, 'cuda')}")
print(f"Train Accuracy {benchmarking.validate(model_unpruned, training, criterion)}")
print(f"Validation Accuracy {benchmarking.validate(model_unpruned, validation, criterion)}")
print("############################################")
print("Compressed Model:")
benchmarking.calculate_model_size(model)
print(f"CPU inference Time {benchmarking.inference_time(model, 'cpu')}")
print(f"GPU inference Time {benchmarking.inference_time(model, 'cuda')}")
print(f"Train Accuracy {benchmarking.validate(model, training, criterion)}")
print(f"Validation Accuracy {benchmarking.validate(model, validation, criterion)}")

pruned_path = args.save_to + str(args.val) + "/" + args.name + ".onnx"
dummy_input = torch.randn(1,3,32,32, requires_grad=True).to("cuda")
torch.onnx.export(model,                                       # model
                  dummy_input,                                 # model input
                  pruned_path,                                  # path
                  export_params=True,                               # store the trained parameter weights inside the model file
                  opset_version=14,                                 # the ONNX version to export the model to
                  do_constant_folding=False,                        # constant folding for optimization
                  input_names = ['input'],                          # input names
                  output_names = ['output'],                        # output names
                  dynamic_axes={'input' : {0 : 'batch_size'},       # variable length axes
                                'output' : {0 : 'batch_size'}})



