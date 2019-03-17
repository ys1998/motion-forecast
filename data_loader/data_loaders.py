from base import BaseDataLoader
from data_loader.preprocess import to_euler_angle, get_motion_bands
from torch.utils.data import Dataset
import numpy as np
import os


# class MnistDataLoader(BaseDataLoader):
#     """
#     MNIST data loading demo using BaseDataLoader
#     """
#     def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):
#         trsfm = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.1307,), (0.3081,))
#             ])
#         self.data_dir = data_dir
#         self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
#         super(MnistDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        

class Human36DataLoader(BaseDataLoader):
    """
    Data loader for Human3.6M dataset.
    """
    def __init__(self, file, batch_size, max_seq_len, num_bands, shuffle, validation_split, num_workers):
        self.dataset = Human36Dataset(file, num_bands, max_seq_len)
        super(Human36DataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class Human36Dataset(Dataset):
    """
    Human 3.6M dataset.
    """

    def __init__(self, file, num_bands, max_seq_len):
        # load data in quaternion format
        quaternion_data = np.load(file)['rotations']
        # convert data to euler angle representation
        euler_data = [to_euler_angle(q) for q in quaternion_data]
        # find length of longest sequence for normalization
        # max_len = max([x.shape[0] for x in euler_data])
        # extract smoothed 'motion bands' from data
        long_data = []
        for x in euler_data:
            smoothed_x = []
            for i in range(3):
                smoothed_x.append(get_motion_bands(x[:,:,i].transpose(), num_bands))
            # (joints, k, time) -> (angles, joints, k, time) -> (time, k, joints, angles)
            long_data.append(np.stack(smoothed_x).transpose(3, 2, 1, 0))
        # split long sequences into independent, shorter sequences
        # chop off trailing values when length not divisible by max_seq_len
        self.data = []
        for x in long_data:
            for i in range(x.shape[0] // max_seq_len):
                self.data.append(x[i*max_seq_len:(i+1)*max_seq_len])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        smoothed, org = x[:,:-1,:,:], x[:,-1,:,:]
        # return smoothed.reshape(smoothed.shape[0], -1), org.reshape(org.shape[0], -1)
        return smoothed.reshape(smoothed.shape[0], -1), smoothed.reshape(*smoothed.shape[0:2], -1)