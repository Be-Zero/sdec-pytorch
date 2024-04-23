import errno
import os
import os.path

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split


class MNIST(data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']
    class_to_idx = {_class: i for i, _class in enumerate(classes)}

    @property
    def targets(self):
        if self.train:
            return self.train_labels
        else:
            return self.test_labels

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.use_cuda = torch.cuda.is_available()

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
            self.train_data = self.train_data.view(
                self.train_data.size(0), -1).float()*0.02
            # self.train_data = self.train_data.view(self.train_data.size(0), -1).float()/255
            self.train_labels = self.train_labels.int()
            if self.use_cuda:
                self.train_data = self.train_data.cuda()
                self.train_labels = self.train_labels.cuda()
        else:
            self.test_data, self.test_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))
            self.test_data = self.test_data.view(
                self.test_data.size(0), -1).float()*0.02
            # self.test_data = self.test_data.view(self.test_data.size(0), -1).float()/255
            self.test_labels = self.test_labels.int()
            if self.use_cuda:
                self.test_data = self.test_data.cuda()
                self.test_labels = self.test_labels.cuda()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(
                self.root, self.processed_folder, self.test_file))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""
        import gzip

        from six.moves import urllib

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        training_set = (
            self.read_image_file(os.path.join(
                self.root, self.raw_folder, 'train-images-idx3-ubyte')),
            self.read_label_file(os.path.join(
                self.root, self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            self.read_image_file(os.path.join(
                self.root, self.raw_folder, 't10k-images-idx3-ubyte')),
            self.read_label_file(os.path.join(
                self.root, self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(
            tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def read_label_file(path):
        with open(path, 'rb') as f:
            data = f.read()
            assert int.from_bytes(
                data[:4], byteorder='big', signed=False) == 2049
            length = int.from_bytes(data[4:8], byteorder='big', signed=False)
            parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
            return torch.from_numpy(parsed).view(length).long()

    def read_image_file(path):
        with open(path, 'rb') as f:
            data = f.read()
            assert int.from_bytes(
                data[:4], byteorder='big', signed=False) == 2051
            length = int.from_bytes(data[4:8], byteorder='big', signed=False)
            num_rows = int.from_bytes(
                data[8:12], byteorder='big', signed=False)
            num_cols = int.from_bytes(
                data[12:16], byteorder='big', signed=False)
            images = []
            parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
            return torch.from_numpy(parsed).view(length, num_rows, num_cols)


class PaviaU(data.Dataset):
    # data: (610, 340, 103)
    # label (610, 340)
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = ['https://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat',
            'https://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat']
    csv_file = 'PaviaU.csv'
    # csv_file = 'PaviaUAllData.csv'
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'
    dataset_file = 'dataset.pt'
    classes = ['1 - Asphalt', '2 - Meadows', '3 - Gravel', '4 - Trees', '5 - Painted metal sheets',
               '6 - Bare Soil', '7 - Bitumen', '8 - Self-Blocking Bricks', '9 - Shadows']  # '0 - background', 
    class_to_idx = {_class: i for i, _class in enumerate(classes)}

    @property
    def targets(self):
        if self.train:
            return self.train_labels
        else:
            return self.test_labels

    def __init__(self, root, download=False, predict=False, train=True, transform=None, target_transform=None, split=False, dataset=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.use_cuda = torch.cuda.is_available()
        self.split = split
        self.dataset = dataset
        self.predict = predict
        self.down = download

        if self.down:
            self.download()

        if self.split:
            self.split_data()

        if self.dataset:
            self.get_data()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

        if self.train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
            self.train_data = self.train_data.view(
                self.train_data.size(0), -1).float()
            # self.train_data = self.train_data.view(self.train_data.size(0), -1).float()/255
            self.train_labels = self.train_labels.int()
            if self.use_cuda:
                self.train_data = self.train_data.cuda()
                self.train_labels = self.train_labels.cuda()
        elif self.predict:
            self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.dataset_file))
            self.train_data = self.train_data.view(
                self.train_data.size(0), -1).float()
            # self.train_data = self.train_data.view(self.train_data.size(0), -1).float()/255
            self.train_labels = self.train_labels.int()
            if self.use_cuda:
                self.train_data = self.train_data.cuda()
                self.train_labels = self.train_labels.cuda()
        else:
            self.test_data, self.test_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))
            self.test_data = self.test_data.view(
                self.test_data.size(0), -1).float()
            # self.test_data = self.test_data.view(self.test_data.size(0), -1).float()/255
            self.test_labels = self.test_labels.int()
            if self.use_cuda:
                self.test_data = self.test_data.cuda()
                self.test_labels = self.test_labels.cuda()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        elif self.predict:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        elif self.predict:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(
                self.root, self.processed_folder, self.test_file))

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(
            tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(
            tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def download(self):
        """Download the PaviaU data if it doesn't exist in processed_folder already."""
        import gzip

        import scipy
        from six.moves import urllib
        from sklearn import preprocessing

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        # data preprocessing
        dataPath = os.path.join(self.root, self.raw_folder, 'PaviaU.mat')
        data = scipy.io.loadmat(dataPath)['paviaU']

        labelPath = os.path.join(self.root, self.raw_folder, 'PaviaU_gt.mat')
        label = scipy.io.loadmat(labelPath)['paviaU_gt']

        # count the number of samples contained in each type
        dict_k = {}
        for i in range(label.shape[0]):
            for j in range(label.shape[1]):
                if label[i][j] not in dict_k:
                    dict_k[label[i][j]] = 0
                dict_k[label[i][j]] += 1

        # 除掉 0 这个非分类的类，把所有需要分类的元素提取出来
        data_label = []
        for i in range(label.shape[0]):
            for j in range(label.shape[1]):
                if label[i][j] != 0:
                    c2l = list(data[i][j])
                    c2l.append(label[i][j])
                    data_label.append(c2l)

        data_label = np.array(data_label)

        # 标准化与存储
        data = preprocessing.StandardScaler().fit_transform(data_label[:, :-1])
        label = data_label[:, -1]

        # 将结果存档后续处理
        data_label = np.column_stack((data, label))
        data_label = pd.DataFrame(data_label)
        data_label.to_csv(os.path.join(self.root, self.raw_folder,
                          self.csv_file), header=False, index=False)

        print('Done!')

    def split_data(self):
        if not os.path.exists(os.path.join(self.root, self.processed_folder)):
            os.makedirs(os.path.join(self.root, self.processed_folder))

        # process and save as torch files
        print('Processing...')

        data = pd.read_csv(os.path.join(
            self.root, self.raw_folder, self.csv_file), header=None)
        data = data.values
        label = data[:, -1].astype(int)
        data = data[:, :-1]

        data_train, data_test, label_train, label_test = train_test_split(
            data, label, test_size=0.3)
        training_set = (torch.from_numpy(data_train),
                        torch.from_numpy(label_train))
        test_set = (torch.from_numpy(data_test), torch.from_numpy(label_test))

        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def get_data(self):
        if not os.path.exists(os.path.join(self.root, self.processed_folder)):
            os.makedirs(os.path.join(self.root, self.processed_folder))

        # process and save as torch files
        print('Processing...')

        data = pd.read_csv(os.path.join(
            self.root, self.raw_folder, self.csv_file), header=None)
        data = data.values
        label = data[:, -1].astype(int)
        data = data[:, :-1]
        data_set = (torch.from_numpy(data), torch.from_numpy(label))

        with open(os.path.join(self.root, self.processed_folder, self.dataset_file), 'wb') as f:
            torch.save(data_set, f)

        print('Done!')
