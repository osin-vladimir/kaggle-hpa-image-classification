from torch import nn
from torch.optim import Adam, SGD, lr_scheduler
import numpy as np
import torch
from torchvision.models import resnet18, resnet34, resnet50, squeezenet1_1, vgg11

# from apex.fp16_utils import network_to_half

import time
import os
from   datetime import datetime as dt

from tqdm import tqdm


def create_model(model_name, n_classes, device, multi_gpu=False, checkpoint=None, freeze_layers=False, fp16=False):

    if model_name == 'resnet18':
        model    = resnet18(pretrained=True)
        if freeze_layers:
            for i, param in model.named_parameters():
                param.requires_grad = False
        model.fc = nn.Linear(in_features=512, out_features=n_classes, bias=True)
        model.fc.weight.requires_grad = True
        model.fc.bias.requires_grad   = True
        model.avgpool = nn.AdaptiveAvgPool2d(1)

    elif model_name == 'resnet34':
        model    = resnet34(pretrained=True)
        if freeze_layers:
            for i, param in model.named_parameters():
                param.requires_grad = False
        model.fc = nn.Linear(in_features=512, out_features=n_classes, bias=True)
        model.fc.weight.requires_grad = True
        model.fc.bias.requires_grad   = True
        model.avgpool = nn.AdaptiveAvgPool2d(1)

    elif model_name == 'resnet50':
        model    = resnet50(pretrained=True)
        if freeze_layers:
            for i, param in model.named_parameters():
                param.requires_grad = False
        model.fc = nn.Linear(in_features=2048, out_features=n_classes, bias=True)
        model.fc.weight.requires_grad = True
        model.fc.bias.requires_grad   = True

    elif model_name == 'squeezenet':
        model    = squeezenet1_1(pretrained=True)
        if freeze_layers:
            for i, param in model.named_parameters():
                param.requires_grad = False
        model.num_classes   = n_classes
        model.classifier[1] = nn.Conv2d(in_channels=512, out_channels=n_classes, kernel_size=(1, 1), stride=(1, 1))

    elif model_name == 'vgg11':
        model = vgg11(pretrained=True)
        if freeze_layers:
            for i, param in model.named_parameters():
                param.requires_grad = False
        model.fc = nn.Linear(in_features=512, out_features=n_classes, bias=True)

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

    if checkpoint:
        model.load_state_dict(torch.load(checkpoint))

    # if fp16:
    #     model = network_to_half(model)

    if multi_gpu:
        model = nn.DataParallel(model)

    return model.to(device)


def create_optim(model, checkpoint=None, name='adam'):

    if name == 'adam':
        optimizer = Adam(list(filter(lambda p: p.requires_grad, model.parameters())), lr=5e-4, amsgrad=True)

    elif name == 'sgd':
        optimizer = SGD(list(filter(lambda p: p.requires_grad, model.parameters())), momentum=0.9)

    if checkpoint:
        optimizer.load_state_dict(torch.load(checkpoint))

    return optimizer


def create_scheduler(optimizer, checkpoint=None):
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5, mode='max')
    if checkpoint:
        scheduler.load_state_dict(torch.load(checkpoint))
    return scheduler


def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train(model, optimizer, criterion, train_loader):
    """
    general training/validation loop

    :param model
    :param optimizer
    :param criterion
    :param train_loader

    :returns training loss, training time
    """

    # params
    it             = 0
    train_loss     = 0

    # start training timer
    train_timer = time.time()

    # set model into training mode
    model.train()

    for inputs, labels in tqdm(train_loader, ncols=50):

        # zeroed all gradients for parameters of the model
        optimizer.zero_grad()

        # send data to device
        inputs, labels = inputs.float().cuda(non_blocking=True), labels.float().cuda(non_blocking=True)

        # forward pass
        output = model(inputs)

        # loss calculation
        loss = criterion(output, labels)

        # graph is created/backward pass (storing gradients)
        loss.backward()

        # accumulating training loss
        train_loss += float(loss.item())

        # clearing the graph
        optimizer.step()

        # print current loss
        it += 1
        # tqdm.write("{0}".format(train_loss / it))

    train_time = time.time() - train_timer

    return train_loss / len(train_loader), train_time / 60


def val(model, optimizer, criterion, val_loader):
    """
    validation step

    :param model:
    :param optimizer:
    :param criterion:
    :param val_loader:


    :returns validation loss, validation time
    """

    # params
    val_timer = time.time()
    optimizer.zero_grad()
    val_loss = 0

    # turn off gradients for validation, saves memory and computations
    with torch.no_grad():

        # set model into evaluation mode
        model.eval()

        tp, fp, fn = np.zeros(33), np.zeros(33), np.zeros(33)

        for inputs, labels in tqdm(val_loader, ncols=50):
            inputs, labels = inputs.float().cuda(non_blocking=True), labels.float().cuda(non_blocking=True)
            output         = model(inputs)
            loss           = criterion(output, labels)
            val_loss      += float(loss.item())

            y_pred = (output.sigmoid() > 0.5).int()
            y_true = labels.int()
            tp += (y_pred * y_true).float().sum(dim=0).cpu().data.numpy()
            fp += (y_pred > y_true).float().sum(dim=0).cpu().data.numpy()
            fn += (y_pred < y_true).float().sum(dim=0).cpu().data.numpy()

        f1_score = (2.0 * tp / (2.0 * tp + fp + fn + 1e-6)).mean()

    val_time = time.time() - val_timer

    return val_loss / len(val_loader), val_time / 60, f1_score