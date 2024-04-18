from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

def get_dataloaders():
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        ])
    train_set  = CIFAR10(root = "../CIFAR", train=True, download=True, transform=transform)
    val = CIFAR10(root = "../CIFAR", train = False, download = True,transform = transform )
    training = DataLoader(train_set, batch_size = 128, shuffle = True )
    validation = DataLoader(val, batch_size = 128, shuffle = True)
    return training, validation