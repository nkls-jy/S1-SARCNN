import torch
from . import paths
from . import sar_dataset
from torchvision.transforms import Compose

def create_train_realsar_dataloaders(patchsize, batchsize, trainsizeiters):
    transform_train = Compose([
        sar_dataset.RandomCropNy(patchsize),
        sar_dataset.Random8OrientationNy(),
        sar_dataset.NumpyToTensor(),
    ])

    trainset = sar_dataset.PlainSarFolder(dirs=paths.train_dir, transform=transform_train, cache=True)
    trainset = torch.utils.data.ConcatDataset([trainset] * trainsizeiters)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchsize, shuffle=True, num_workers=12)

    return trainloader

def create_valid_realsar_dataloaders(patchsize, batchsize):
    transform_valid = Compose([
        sar_dataset.CenterCropNy(patchsize),
        sar_dataset.NumpyToTensor(),
    ])

    validset = sar_dataset.PlainSarFolder(dirs=paths.valid_dir, transform=transform_valid, cache=True)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batchsize, shuffle=False, num_workers=12)

    return validloader


class PreprocessingInt:
    def __call__(self, batch):
        tl = torch.split(batch, 1, dim=1)

        noisy = tl[0]
        target = tl[1]

        if batch.is_cuda:
            noisy = noisy.cuda()
            target = target.cuda()

        return noisy, target


