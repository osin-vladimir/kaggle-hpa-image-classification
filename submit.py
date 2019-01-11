import torch
import os
import sys
import yaml
import pandas as pd
import numpy  as np

sys.path.append(os.path.abspath(".."))

from common.torch_info import device
from models            import create_model
from dataset           import ProteinDataset
from torch.utils.data  import DataLoader
from torchvision       import transforms
from tqdm import tqdm


from albumentations    import HorizontalFlip, VerticalFlip, Rotate


if __name__ == '__main__':
    # params
    params = yaml.load(open('config.yaml'))

    # data transforms
    main_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(255*params['mean'], 255*params['std'])])

    # defining test dataset
    test_dataset = ProteinDataset(img_csv_path=params['img_csv_path_test'],
                                  mode='test',
                                  data_path=params['data_path'],
                                  depth=3,
                                  img_size=512,
                                  transform=None)

    # defining dataloader
    test_loader = DataLoader(test_dataset,
                             shuffle=False,
                             batch_size=1,
                             pin_memory=True,
                             num_workers=4)

    # defining training components
    model = create_model(model_name=params['model_name'],
                         n_classes=params['n_classes'],
                         device=device,
                         multi_gpu=params['multi_gpu'],
                         checkpoint=params['checkpoint_m'])

    model.eval()

    submission_df = pd.read_csv(params['img_csv_path_test'])
    labels, submissions, probs = [], [], []
    submit_results = []

    if params['tta']:
        with torch.no_grad():
            for inputs in tqdm(test_loader):
                tta = []
                for aug in [HorizontalFlip(p=1), VerticalFlip(p=1), Rotate(90, p=1), Rotate(180, p=1)]:
                    image = (aug(image=inputs.numpy().reshape((512, 512, 3))).copy())['image']
                    image = main_transforms(image)
                    image_var = image.unsqueeze(0).float().cuda(non_blocking=True)
                    y_pred    = model(image_var)
                    prob      = y_pred.sigmoid().cpu().data.numpy()[:28]
                    tta.append(prob)

                prob_avg = np.stack(tta, axis=0).mean(axis=0)
                probs.append(prob_avg)
                labels.append(prob_avg > 0.5)
    else:
        with torch.no_grad():
          for inputs in tqdm(test_loader):
            image_var = inputs.float().cuda(non_blocking=True)
            y_pred    = model(image_var)
            prob      = y_pred.sigmoid().cpu().data.numpy()[:28]
            probs.append(prob)
            labels.append(prob > 0.5)

    for row in np.concatenate(labels):
        subrow = ' '.join(list([str(i) for i in np.nonzero(row)[0]]))
        submissions.append(subrow)

    submission_df['Predicted'] = submissions

    submission_df.to_csv(params['models_dir'] + params['model_name'] + ".csv", index=None)
    probs_array = np.stack(probs, axis=0)
    np.save(params['models_dir'] + params['model_name'] + "_rgb_512_tta_probs_83.npy", probs_array)

