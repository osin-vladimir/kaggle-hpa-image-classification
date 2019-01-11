import torch
from torchvision.models import resnet18, resnet34
from torch import nn
from pretrainedmodels.models import bninception


def create_model(model_name, n_classes, device, multi_gpu=False, checkpoint=None, freeze_layers=False):

    if model_name == 'resnet18':
        model    = resnet18(pretrained=True)
        if freeze_layers:
            for i, param in model.named_parameters():
                param.requires_grad = False
        model.fc = nn.Linear(in_features=512, out_features=n_classes, bias=True)
        model.fc.weight.requires_grad = True
        model.fc.bias.requires_grad   = True

    elif model_name == 'resnet34':
        model    = resnet34(pretrained=True)
        if freeze_layers:
            for i, param in model.named_parameters():
                param.requires_grad = False
        model.fc = nn.Linear(in_features=512, out_features=n_classes, bias=True)
        model.fc.weight.requires_grad = True
        model.fc.bias.requires_grad   = True

    elif model_name == 'resnet18_4':
        model    = resnet18(pretrained=True)
        if freeze_layers:
            for i, param in model.named_parameters():
                param.requires_grad = False

        weight             = model.conv1.weight
        model.conv1        = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.conv1.weight = nn.Parameter(torch.cat((weight, weight[:, :1, :, :]), dim=1))
        model.fc           = nn.Linear(in_features=512, out_features=n_classes, bias=True)
        model.fc.weight.requires_grad = True
        model.fc.bias.requires_grad   = True

    elif model_name == 'bninception':
        model = bninception(pretrained="imagenet")
        model.global_pool = nn.AdaptiveAvgPool2d(1)
        model.last_linear = nn.Sequential(nn.Dropout(0.2), nn.Linear(1024, n_classes))

    if checkpoint:
        model.load_state_dict(torch.load(checkpoint))

    if multi_gpu:
        model = nn.DataParallel(model)

    return model.to(device)