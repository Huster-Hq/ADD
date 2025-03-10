import glob
import numpy as np
import random
import imageio
from scipy.ndimage import rotate
from skimage.transform import resize
from torch.utils.data import Dataset


class CPCDataset(Dataset):
    def __init__(self, is_train,split_id, enable_aug=True):
        self.enable_aug = enable_aug
        if is_train==True:
            self.wli_list = open('../split/train_wht_pub%d.txt' % split_id).readlines() 
        else:
            self.wli_list = open('../split/val_wht_pub0%d.txt' % split_id).readlines()

        self.is_train = is_train
        self.wli_list = list(map(lambda x: x.strip(), self.wli_list))
        self.wli_label = list(map(lambda x: 'hyperplastic_lesions' not in x, self.wli_list))
        self.wli_label = np.array(self.wli_label, dtype=np.int8)
        unique, counts = np.unique(self.wli_label, return_counts=True)
        print('wli:', dict(zip(unique, counts)))

    def __getitem__(self, index):
        wli_path = self.wli_list[index]
        nbi_paths = glob.glob(wli_path.replace('White_light', 'NBI'))
        if len(nbi_paths) == 0:
            raise Exception("%s NBI is empty" % wli_path)
        nbi_path = random.choice(nbi_paths)
        wli_img = imageio.imread(wli_path)
        nbi_img = imageio.imread(nbi_path)
        # augmentation
        if self.is_train and self.enable_aug:
            angle = random.randint(-180, 180)
            wli_img = rotate(wli_img, angle)
            nbi_img = rotate(nbi_img, angle)

            if random.random() > 0.5:
                nbi_img = np.flip(nbi_img, 0)
                wli_img = np.flip(wli_img, 0)

            if random.random() > 0.5:
                nbi_img = np.flip(nbi_img, 1)
                wli_img = np.flip(wli_img, 1)

        wli_img = resize(wli_img, (448, 448))
        nbi_img = resize(nbi_img, (448, 448))
        wli_img = np.swapaxes(wli_img, 0,-1)
        nbi_img = np.swapaxes(nbi_img, 0,-1)

        label = self.wli_label[index]
        return wli_img, nbi_img, label, wli_path,nbi_path

    def __len__(self):
        return len(self.wli_list)
