import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

__all__ = ["LoadDataset", "tensor_data_create", "inf_train_gen"]


def tensor_data_create(features, labels):
    tensor_x = torch.stack(
        [torch.FloatTensor(i) for i in features]
    )  # transform to torch tensors
    tensor_y = torch.stack([torch.LongTensor([i]) for i in labels])[:, 0]
    dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
    return dataset


def inf_train_gen(trainloader):
    while True:
        for data, targets in trainloader:
            yield (data, targets)


class LoadDataset(Dataset):
    def __init__(self, data_X, data_y, preprocess="none", **kwargs):
        self.preprocess = preprocess

        self.dset = data_X.values
        self.column_names = data_X.columns

        ### load labels
        self.anno = data_y.values.tolist()
        self.label_name = data_y.columns.tolist()

        ### encode labels
        self.label_encoder = LabelEncoder()
        self.anno = self.label_encoder.fit_transform(self.anno)
        self.label_map = self.label_encoder.classes_
        self.label_dict = {}
        self.inverse_label_dict = {}
        for ind, ll in enumerate(self.label_map):
            self.label_dict[ll] = ind
            self.inverse_label_dict[ind] = ll

        ### pre-process the features
        if preprocess == "standard":
            self.dset = self.to_standard(self.dset)
        elif preprocess == "minmax":
            self.dset = self.to_minmax(self.dset)
        elif preprocess == "discretize":
            self.dset = self.to_discretize(float(kwargs["alpha"]))

    def get_dim(self):
        x_dim = self.dset.shape[-1]
        y_dim = len(self.label_map)
        return x_dim, y_dim

    def train_test(self, k=1000, test_fraction=0.2):
        Train_x, Test_x, Train_y, Test_y = train_test_split(
            self.dset, self.anno, test_size=test_fraction, random_state=k
        )
        self.dset, self.anno = Train_x, Train_y
        return Train_x, Train_y, Test_x, Test_y

    def to_discretize(self, alpha=0.25):
        assert alpha < 0.5, "the alpha (quantile) should be smaller than 0.5"
        alphas = [
            alpha,
            0.5,
            1 - alpha,
        ]  # quantiles for the discretization (let num_active = num_inactive)
        bin_number = len(alphas) + 1
        data_quantile = np.quantile(self.dset, alphas, axis=0)
        x_dim, _ = self.get_dim()

        data_discrete = []
        statistic_dict = {}  # storing the discritization results
        mean_dict = {}  # storing the mean for each bin
        for idx in range(x_dim):
            gene_name = self.column_names[idx]
            discrete_col = np.digitize(self.dset[:, idx], data_quantile[:, idx])
            data_discrete.append(discrete_col)

            # store the results (for inverse_transform)
            statistic_dict[gene_name] = []
            mean_dict[gene_name] = []
            for bin_idx in range(bin_number):
                curr_col = self.dset[:, idx]
                bin_arr = curr_col[discrete_col == bin_idx]
                statistic_dict[gene_name].append(bin_arr)
                mean_dict[gene_name].append(np.mean(bin_arr))
        data_discrete = np.array(data_discrete).T
        self.dset = data_discrete
        self.statistic_dict = statistic_dict
        self.mean_dict = mean_dict
        self.bin_number = bin_number
        return self.dset

    def inverse_discretize(self, dset):
        x_dim, _ = self.get_dim()
        data_processed = []
        if isinstance(dset, np.ndarray):
            for idx in range(x_dim):
                gene_name = self.column_names[idx]
                mean_labels = np.array(self.mean_dict[gene_name])
                data_processed.append(mean_labels[dset[:, idx]])
        elif isinstance(dset, pd.DataFrame):
            input_column_names = dset.columns
            dset = dset.values
            for idx, gene_name in enumerate(input_column_names):
                if gene_name == self.label_name:
                    continue
                mean_labels = np.array(self.mean_dict[gene_name])
                data_processed.append(mean_labels[dset[:, idx]])
        else:
            raise TypeError("Dset must be a np.ndarray or pd.DataFrame")
        data_processed = np.array(data_processed).T
        return data_processed

    def to_dataframe(self):
        """
        merge the features and labels, covert to a dataframe
        """
        # inverse_label_map={j:i for i,j in self.label_dict.items()}
        inverse_anno = [self.inverse_label_dict[i] for i in self.anno]
        merged = np.hstack((self.dset, np.expand_dims(inverse_anno, -1)))
        dset = pd.DataFrame(
            merged, columns=np.append(self.column_names, self.label_name)
        )
        return dset

    def xy_to_dataframe(self, x, y, merge_xy=False):
        """
        merge the features and labels, covert to a dataframe
        """
        inverse_anno = [self.inverse_label_dict[i] for i in y]
        X_df = pd.DataFrame(self._inverse_transform(x), columns=self.column_names)
        y_df = pd.DataFrame(np.expand_dims(inverse_anno, -1), columns=self.label_name)
        dset = pd.concat([X_df, y_df], axis=1)
        if merge_xy:
            return dset
        else:
            return X_df, y_df

    def to_minmax(self, dset):
        self._transform = MinMaxScaler().fit(dset)
        return self._transform.transform(dset)

    def to_standard(self, dset):
        self._transform = StandardScaler().fit(dset)
        return self._transform.transform(dset)

    def _inverse_transform(self, dset):
        if self.preprocess == "none":
            return dset
        elif self.preprocess == "discretize":
            return self.inverse_discretize(dset)
        else:
            return self._transform.inverse_transform(dset)

    def __getitem__(self, index):
        return self.dset[index].astype(np.float32), self.anno[index]

    def __len__(self):
        return len(self.anno)


def test_preprocess():
    preprocess = "standard"
    data_X = pd.read_csv("./data_splits/TCGA-BRCA/real/X_train_real_split_1.csv")
    data_y = pd.read_csv("./data_splits/TCGA-BRCA/real/y_train_real_split_1.csv")

    data_X_test = pd.read_csv("./data_splits/TCGA-BRCA/real/X_test_real_split_1.csv")
    data_y_test = pd.read_csv("./data_splits/TCGA-BRCA/real/y_test_real_split_1.csv")

    dset = LoadDataset(data_X=data_X, data_y=data_y, preprocess=preprocess)
    print(dset.dset[:10, 0])

    if preprocess == "none":
        assert np.equal(dset.dset[:10, 0], data_X.values[:10, 0]).all()
    elif preprocess == "standard" or preprocess == "minmax":
        inverse_dset = dset._inverse_transform(dset.dset)
        assert np.isclose(data_X.values[:10, 0], inverse_dset[:10, 0]).all()

    print(dset._transform.transform(data_X_test))
    print(dset.label_encoder.transform(data_y_test))


# test_preprocess()
