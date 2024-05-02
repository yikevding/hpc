import torch
import sys
sys.path.append("..")
from baselines import resnet

def get_state_dict(path, model_num = 20):
    if model_num == 20:
        model_unpruned = resnet.resnet20()
    elif model_num == 32:
        model_unpruned = resnet.resnet32()
    elif model_num == 44:
        model_unpruned = resnet.resnet44()
    elif model_num == 56:
        model_unpruned = resnet.resnet56()
    elif model_num == 110:
        model_unpruned = resnet.resnet110()
    else:
        print("Choose an appropriate model type")
        return
    pretrained_state_dict = torch.load(path)['state_dict']
    state_dict = model_unpruned.state_dict()
    for key in pretrained_state_dict.keys():
        new_key = key[key.find('.')+1:]
        state_dict[new_key] = pretrained_state_dict[key]
    return state_dict


    
