import os
import sys

sys.path.append(os.path.abspath(".."))

import torch
import yaml
import pandas as pd

from common.torch_info import device
from common.utils      import create_model, train, val, create_optim, create_scheduler, get_learning_rate
from dataset           import ProteinDataset
from common.criterions import F1Loss
from torch.utils.data  import DataLoader, WeightedRandomSampler
from torchvision       import transforms
from datetime import datetime as dt


if __name__ == '__main__':
    # params
    params = yaml.load(open('config.yaml'))

    # data transforms
    main_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(255*params['mean'], 255*params['std'])])

    # defining train/val datasets
    train_dataset = ProteinDataset(img_csv_path=params['img_csv_path_train'],
                                   mode='train',
                                   data_path=params['data_path'],
                                   depth=params['depth'],
                                   img_size=params['image_size'],
                                   transform=main_transforms)

    val_dataset   = ProteinDataset(img_csv_path=params['img_csv_path_val'],
                                   data_path=params['data_path'],
                                   mode='val',
                                   depth=params['depth'],
                                   img_size=params['image_size'],
                                   transform=main_transforms)

    # defining dataloaders
    weights      = pd.read_csv(params['img_csv_path_train']).Weight.values
    train_loader = DataLoader(train_dataset,
                              sampler=WeightedRandomSampler(weights=weights, num_samples=len(weights)),
                              batch_size=params['batch_size'],
                              num_workers=4,
                              pin_memory=True)

    val_loader = DataLoader(val_dataset,
                             shuffle=False,
                             batch_size=params['batch_size'],
                             pin_memory=True,
                             num_workers=4)
    
    # defining training components
    model     = create_model(model_name=params['model_name'],
                              n_classes=params['n_classes'],
                              device=device,
                              multi_gpu=params['multi_gpu'])
    optimizer = create_optim(model=model, name='adam')
    scheduler = create_scheduler(optimizer)
    criterion = F1Loss().cuda()
    
    print('start training...')
    
    # create model/experiment folder
    if not params['debug']:
        # create folder for experiment
        experiment_folder = params['models_dir'] + params['model_name'] + '_' + str(dt.now()).replace(' ', '_')
        checkpoint_folder = experiment_folder + '/checkpoint/'
    
        if not os.path.exists(experiment_folder):
          os.makedirs(experiment_folder)
    
        if not os.path.exists(checkpoint_folder):
          os.makedirs(checkpoint_folder)
    
    for epoch in range(100):
    
        # updating model weights
        train_loss, train_time = train(model, optimizer, criterion, train_loader)
    
        # validation
        val_loss, val_time, f1 = val(model, optimizer, criterion, val_loader)

        # logging message
        log = "Epoch : {} |".format(epoch + 1), \
              "LR    : {} |".format(get_learning_rate(optimizer)), \
              "Train Loss : {:.3f} |".format(train_loss), \
              "Val   Loss : {:.3f} |".format(val_loss), \
              "Val   F1   : {:.3f} |".format(f1), \
              "Time  Train: {:.3f} |".format(train_time), \
              "Time  Val  : {:.3f} |".format(val_time),

        print(log)
    
        if not params['debug']:
            # save model state
            torch.save(model.module.state_dict(), checkpoint_folder + '{0}_model.pth'.format(epoch + 1))

            # save optimizer state
            torch.save(optimizer.state_dict(), checkpoint_folder + '{0}_optimizer.pth'.format(epoch + 1))
    
            # save scheduler state
            torch.save(scheduler.state_dict(), checkpoint_folder + '{0}_scheduler.pth'.format(epoch + 1))

            # save logs
            logs = open(experiment_folder + '/logs.txt', 'a')
            logs.writelines(log)
            logs.writelines('\n')
            logs.close()

        scheduler.step(f1)



