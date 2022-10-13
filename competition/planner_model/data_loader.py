import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf

from tqdm import tqdm


class Dataset:
    def __init__(self, directory, training=True, debug=False):

        self.training = training
        self.dataset = None
        self.files = []
        for root, dirs, files in os.walk(directory, topdown=True, followlinks=True):
            for dir_ in dirs:

                if dir_.startswith("train" if training else "val"):
                    for fname in os.listdir(os.path.join(root, dir_)):
                        if fname.startswith("env"):
                            env_file = os.path.join(root, dir_, fname)
                            self.files.append(os.path.abspath(env_file))

        self.num_files = len(self.files)

    def _recursive_list(self, subpath):
        return fnmatch.filter(os.listdir(subpath), "*.dat")

    def build_dataset(self):
        self._build_dataset()

    def _build_dataset(self):
        raise NotImplementedError

    def _decode_experiment_dir(self, dir_subpath):
        raise NotImplementedError


class PlanDataset(Dataset):
    def __init__(self, directory, training=True):
        super(PlanDataset, self).__init__(directory, training=training)
        self.df = pd.DataFrame()
        self.build_dataset()

    def _dataset_map(self, tensor):

        features = tensor[:36]
        features = tf.stack([features, features, features])

        labels = [[tensor[36:66], tensor[66:96], tensor[96:126]]]
        labels = tf.stack(labels)[0]

        return features, labels

    def build_dataset(self):

        for fname in self.files:
            self.df = pd.concat([self.df, pd.read_csv(fname, index_col=False)])

        self.dataset = tf.convert_to_tensor(self.df)
        self.dataset = self.dataset[:, 1:]
        self.dataset = tf.data.Dataset.from_tensor_slices(self.dataset)
        self.dataset = self.dataset.shuffle(self.dataset.cardinality().numpy())
        self.dataset = self.dataset.map(self._dataset_map, num_parallel_calls=4)
        self.dataset = self.dataset.batch(8, drop_remainder=False)
        self.dataset = self.dataset.prefetch(buffer_size=4)


if __name__ == "__main__":

    dataset = PlanDataset("planner_model/")
    dataset.test_pulling()
