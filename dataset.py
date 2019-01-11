import cv2
import pandas as pd
import numpy  as np
from torch.utils.data import Dataset


from albumentations import (CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose
)


aug = Compose([
        RandomRotate90(),
        Flip(),
        Transpose(),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=.2),
            MedianBlur(blur_limit=3, p=.1),
            Blur(blur_limit=3, p=.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=.2),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomContrast(),
            RandomBrightness(),
        ], p=0.3),
        HueSaturationValue(p=0.3),
    ], p=0.9)


name_to_label_dict = {
    'nucleoplasm': 0,
    'nuclear membrane': 1,
    'nucleoli': 2,
    'nucleoli fibrillar center': 3,
    'nuclear speckles': 4,
    'nuclear bodies': 5,
    'endoplasmic reticulum': 6,
    'golgi apparatus': 7,
    'peroxisomes': 8,
    'endosomes': 9,
    'lysosomes': 10,
    'intermediate filaments': 11,
    'actin filaments': 12,
    'focal adhesion sites': 13,
    'microtubules': 14,
    'microtubule ends': 15,
    'cytokinetic bridge': 16,
    'mitotic spindle': 17,
    'microtubule organizing center': 18,
    'centrosome': 19,
    'lipid droplets': 20,
    'plasma membrane': 21,
    'cell junctions': 22,
    'mitochondria': 23,
    'aggresome': 24,
    'cytosol': 25,
    'cytoplasmic bodies': 26,
    'rods & rings': 27,
    'vesicles': 28,
    'nucleus': 29,
    'midbody': 30,
    'midbody ring': 31,
    'cleavage furrow': 32
}


class ProteinDataset(Dataset):

    def __init__(self, img_csv_path, data_path, mode, img_size, depth, transform):
        self.data_frame      = pd.read_csv(img_csv_path)
        self.data_path       = data_path
        self.mode            = mode
        self.img_size        = img_size
        self.depth           = depth
        self.transform       = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        file_name = self.data_path + self.data_frame.iloc[idx].Id
        colors  = ['_red', '_green', '_blue']

        if self.depth == 4:
            colors.append('_yellow')

        img   = [cv2.imread(file_name + c + '.png', cv2.IMREAD_GRAYSCALE).astype('uint8') for c in colors]
        image = np.stack(img, axis=-1)
        image = cv2.resize(image, (self.img_size, self.img_size))

        if self.mode == 'train':
            image = aug(image=image)['image']

        if self.transform:
            image = self.transform(image)

        if self.mode != 'test':
            labels         = list(map(int, self.data_frame.iloc[idx].Target.split(" ")))
            labels_one_hot = np.eye(len(name_to_label_dict), dtype=np.float)[labels].sum(axis=0)
            return image, labels_one_hot
        else:
            return image
