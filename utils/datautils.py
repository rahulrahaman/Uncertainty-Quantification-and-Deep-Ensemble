import numpy as np
import torch
import torchvision
import torch.utils.data as Data
from torchvision import transforms
from PIL import Image


class CutOut(object):
    """
    CutOut augmentation -- replace a box by its mean (coordinate-wise)
    """

    def __init__(self, side):
        self.side = side

    def __call__(self, image):
        xx = np.random.randint(0, 32-self.side)
        yy = np.random.randint(0, 32-self.side)
        for c in range(3):
            image[c,xx:(xx+self.side), yy:(yy+self.side)]=torch.mean(image[c,xx:(xx+self.side), yy:(yy+self.side)])
        return image


class PixelJitter(object):
    """
    Use the augmentation pixel jitter on images in a sample
    """

    def __init__(self, power_min=0.4, power_max=1.4):
        assert 0 < power_min
        assert power_min < power_max
        self.power_min = power_min
        self.power_max = power_max

    def __call__(self, image):
        p = self.power_min + (self.power_max - self.power_min) * np.random.rand()
        image = image ** p
        for c in range(3):
            alpha = -0.1 + 0.2 * np.random.rand()
            strech = 0.8 + 0.4 * np.random.rand()
            image[c, :, :] = alpha + strech * image[c, :, :]
        image = torch.clamp(image, min=0., max=1.)
        return image


TRAIN_AUGMENT = transforms.Compose([
    transforms.Pad(5, padding_mode='edge'),
    transforms.RandomCrop((32, 32), padding_mode='constant'),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    CutOut(side=12),
    PixelJitter(),
])

TEST_AUGMENT = transforms.Compose([
    transforms.ToTensor(),
])


class Mnist(torchvision.datasets.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, start=0, end=-1):
        """
        :param root: (str) path to data directory
        :param train: whether training dataset (True) or test (False)
        :param transform: (torchvision.transforms) transformation for the images
        :param target_transform: target_transform: transformation of the targets
        :param download: (bool) whether to download the dataset
        :param start: (int) number of examples chosen in the training subset
        :param end: (int) number of training + validation examples chosen
        """
        super(Mnist, self).__init__(root=root, train=train, transform=transform,
                                    target_transform=target_transform, download=download)
        end = len(self.targets) if end < 0 else end
        self.targets = self.targets[start:end]
        self.data = self.data[start:end]


class TruncatedImageFolder(torchvision.datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, start=0, length=-1):
        """

        :param root: (str) path to data directory
        :param transform: (torchvision.transforms) transformation for the images
        :param target_transform: transformation of the targets
        :param start: (int) the index of the first example
        :param length: (int) how many examples to consider in the dataset from 'start'
        """
        super(TruncatedImageFolder, self).__init__(root, transform=transform,
                                                   target_transform=target_transform)

        classes, counts = np.unique([item[1] for item in self.samples], return_counts=True)
        length = min(counts) if length < 0 else length
        samples = []
        curr_index = 0
        for i in range(len(classes)):
            samples += self.samples[curr_index+start: curr_index+start+length]
            curr_index += counts[i]
        self.samples = samples


class RetinopathyDataset(Data.Dataset):
    """
    DIABETIC RETINOPATHY dataset downloaded from
    kaggle link : https://www.kaggle.com/c/diabetic-retinopathy-detection/data
    root : location of data files
    train : whether training dataset (True) or test (False)
    transform : torch image transformations
    binary : whether healthy '0, 1' vs damaged '2,3,4' binary detection (True) 
             or multiclass (False)
    """

    def __init__(self, root, train, transform, binary=True, balance=True, size=None):
        # root += 'DIABETIC_RETINOPATHY_RESZ/'  # DIABETIC_RETINOPATHY_CLAHE_RESZ
        if train:
            self.img_dir = os.path.join(root, 'train/')
            label_csv = os.path.join(root, 'trainLabels.csv')
            with open(label_csv, 'r') as label_file:
                label_tuple = [line.strip().split(',')[:2] for line in label_file.readlines()[1:]]
            self.imgs = [item[0] for item in label_tuple]
            self.labels = [int(item[1]) for item in label_tuple]

            with open(label_csv.replace('train', 'test'), 'r') as label_file:
                label_tuple = [line.strip().split(',')[:2] for line in label_file.readlines()[1:]]
            self.imgs += [item[0] for item in label_tuple[:30000]]
            self.labels += [int(item[1]) for item in label_tuple[:30000]]

        else:
            self.img_dir = os.path.join(root, 'test/')
            label_csv = os.path.join(root, 'testLabels.csv')
            with open(label_csv, 'r') as label_file:
                label_tuple = [line.strip().split(',')[:2] for line in label_file.readlines()[1:]]
            self.imgs = [item[0] for item in label_tuple[30000:]]
            self.labels = [int(item[1]) for item in label_tuple[30000:]]

        self.transform = transform
        self.binary = binary
        if self.binary:
            self.labels = [int(label > 1) for label in self.labels]

        # Discard bad images
        bad_images = ['10_left']
        for img in bad_images:
            if img in self.imgs:
                index = self.imgs.index(img)
                self.imgs = self.imgs[:index] + self.imgs[index+1:]
                self.labels = self.labels[:index] + self.labels[index+1:]

        # Make all these better
        if size is not None:
            classes, counts = np.unique(np.array(self.labels), return_counts=True)
            weights = 1. / (len(classes) * counts)
            indices = np.random.choice(a=np.arange(len(self.labels)), size=size, replace=True, p=weights[self.labels])
            self.indices = indices
            print(indices[:10])
            self.imgs = np.array(self.imgs)[indices].tolist()
            self.labels = np.array(self.labels)[indices].tolist()

        classes, counts = np.unique(np.array(self.labels), return_counts=True)
        weights = 1. / torch.tensor(counts, dtype=torch.float)
        weights = weights / weights.sum()
        deviation = np.std(counts) / np.mean(counts)
        if deviation > 0.05 and train and balance:
            self.weights = weights.numpy().tolist()
            # self.sample_weights = weights[self.labels]
            print('Class weights calculated as ', dict(zip(classes, weights.numpy())))
        
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        try:
            image = Image.open(self.img_dir + img_name + '.jpeg')
        except:
            image = Image.open(self.img_dir.replace('train', 'test') + img_name + '.jpeg')
        image = self.transform(image)
        label = self.labels[idx]

        return image, label


class ReformedCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root, train, download, transform, start=0, end=-1, index=False):
        """

        :param root: (str) path to data directory
        :param train: (bool) train if true, test if false
        :param download: (bool) whether to download the dataset
        :param transform: (torchvision.transforms) transformation for the images
        :param start: (int) number of examples chosen in the training subset
        :param end: (int) number of training + validation examples chosen
        :param index: (bool) whether to return index of the example
        """
        super(ReformedCIFAR10, self).__init__(root=root, train=train, download=download, transform=transform)
        if end == -1:
            end = len(self.targets)
        self.data = self.data[start: end]
        self.targets = self.targets[start: end]
        self.get_index = index
        
    def __getitem__(self, idx):
        img, target = super(ReformedCIFAR10, self).__getitem__(idx)
        if self.get_index:
            return img, target, idx
        else:
            return img, target


class ReformedCIFAR100(torchvision.datasets.CIFAR100):
    def __init__(self, root, train, download, transform, start=0, end=-1):
        """

        :param root: (str) path to data directory
        :param train: (bool) train if true, test if false
        :param download: (bool) whether to download the dataset
        :param transform: (torchvision.transforms) transformation for the images
        :param start: (int) number of examples chosen in the training subset
        :param end: (int) number of training + validation examples chosen
        """
        super(ReformedCIFAR100, self).__init__(root=root, train=train, download=download, transform=transform)
        if end == -1:
            end = len(self.targets)
        self.data = self.data[start: end]
        self.targets = self.targets[start: end]


def return_loaders(dataset, base, start=None, end=None, batch_size=500,
                   train_shuffle=True, valid_shuffle=True, **kwargs):
    """

    :param dataset: (str) name of the dataset (CIFAR10,CIFAR100,DIABETIC_RETINOPATHY,IMAGEWOOF,IMAGENETTE)
    :param base: (str) name of the base directory
    :param start: (int) number of examples chosen in the training subset
    :param end: (int) number of training + validation examples chosen
    :param batch_size: (int) batch size
    :param train_shuffle: (bool) shuffle the train loader or not (do not in case infering Ensembles)
    :param valid_shuffle: (bool) shuffle the validation loader or not (do not in case infering Ensembles)
    :param kwargs: (dict) additional keywords such as num_workers
    :return:
    """
    return_dict = dict()
    num_workers = kwargs.get('num_workers', 5)
    train_batchsize = batch_size

    if 'cifar' in dataset.lower():
        index = kwargs.get('index', False)
        cifar10 = (dataset.lower() == 'cifar10')
        if cifar10:
            dataset_fn = ReformedCIFAR10
            num_class = 10
        else:
            dataset_fn = ReformedCIFAR100
            num_class = 100
        
        # DEFINE DATASET
        cifar_train_dataset = dataset_fn(base, train=True, transform=TRAIN_AUGMENT, download=True,
                                         end=start, index=index)
        return_dict['train'] = torch.utils.data.DataLoader(cifar_train_dataset, batch_size=train_batchsize,
                                                           shuffle=train_shuffle, num_workers=num_workers)

        cifar_train_dataset_noaugment = dataset_fn(base, train=True, transform=TEST_AUGMENT, download=True, end=start)
        return_dict['no-augment_train'] = torch.utils.data.DataLoader(cifar_train_dataset_noaugment,
                                                                      batch_size=train_batchsize, shuffle=train_shuffle,
                                                                      num_workers=num_workers)

        if start != end:
            cifar_valid_dataset = dataset_fn(base, train=True, transform=TRAIN_AUGMENT,
                                             download=True, start=start, end=end)

            cifar_valid_dataset_noaugment = dataset_fn(base, train=True, transform=TEST_AUGMENT,
                                                       download=True, start=start, end=end)
            return_dict['valid'] = torch.utils.data.DataLoader(cifar_valid_dataset, batch_size=train_batchsize,
                                                               shuffle=valid_shuffle, num_workers=num_workers)

            return_dict['no-augment_valid'] = torch.utils.data.DataLoader(cifar_valid_dataset_noaugment,
                                                                          batch_size=train_batchsize,
                                                                          shuffle=valid_shuffle, num_workers=num_workers)

        cifar_test_dataset = dataset_fn(base, train=False, transform=TEST_AUGMENT, download=True)

        return_dict['test'] = torch.utils.data.DataLoader(cifar_test_dataset, batch_size=train_batchsize,
                                                          shuffle=False, num_workers=num_workers)
        
    elif 'diabetic' in dataset.lower():
        # directory = '/data02/DIABETIC_RETINOPATHY/DIABETIC_RETINOPATHY_NOHISTEQ_RESZ/'
        binary = kwargs.get('binary', False)
        size = kwargs.get('size', None)
        val = kwargs.get('val', None)
        if size is None:
            print("Please return valid 'size' and 'val' parameters")
            return
        directory = '/data02/DIABETIC_RETINOPATHY/DIABETIC_RETINOPATHY_HISTEQ_RESZ/'
        transform = torchvision.transforms.Compose([transforms.ColorJitter(brightness=(0.7, 1.3), 
                                                                           contrast=(0.7, 1.3)),
                                                    transforms.ToTensor()])
        test_transform = torchvision.transforms.Compose([transforms.ToTensor()])

        train_data = RetinopathyDataset(root=directory, train=True, transform=transform,
                                        binary=binary, balance=True, size=size)
        test_data = RetinopathyDataset(root=directory, train=False, transform=test_transform, binary=binary,
                                       balance=True, size=5000)
        valid_data = RetinopathyDataset(root=directory, train=True, transform=test_transform,
                                        binary=binary, balance=True, size=val)
        
        return_dict['train'] = Data.DataLoader(train_data, batch_size=batch_size, shuffle=train_shuffle,
                                               num_workers=num_workers)
        return_dict['test'] = Data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return_dict['no-augment_valid'] = Data.DataLoader(valid_data, batch_size=batch_size, shuffle=valid_shuffle,
                                                          num_workers=num_workers)
        num_class = 5 if not binary else 2
        print('Overlap count is', len([item for item in valid_data.indices if item in train_data.indices]))

    elif 'mnist' in dataset.lower():
        return_dict, num_class = {}, 10
        return_dict['train'] = torch.utils.data.DataLoader(
            Mnist('../../data', end=start, train=True, download=True,
                  transform=transforms.Compose([
                      transforms.ToTensor(),
                      transforms.Normalize((0.1307,), (0.3081,))
                  ])),
            batch_size=200, shuffle=train_shuffle, num_workers=num_workers)
        return_dict['no-augment_train'] = return_dict['train']

        if start != end:
            return_dict['no-augment_valid'] = torch.utils.data.DataLoader(
                Mnist('../../data', start=start, end=end, train=True, download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))
                      ])),
                batch_size=200, shuffle=valid_shuffle, num_workers=num_workers)
            return_dict['valid'] = return_dict['no-augment_valid']

        return_dict['test'] = torch.utils.data.DataLoader(
            Mnist('../../data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=200, shuffle=False, num_workers=num_workers)

    elif ('imagewoof' in dataset.lower()) or ('imagenette' in dataset.lower()):
        num_class = 10
        directory = 'imagewoof-320' if 'imagewoof' in dataset.lower() else 'imagenette-320'
        train_augment = transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor()
                        ])
        test_augment = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        ])
        train_dataset = TruncatedImageFolder(
            '/data02/datasets/'+directory+'/train/',
            transform=train_augment, length=start)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=train_shuffle,
            num_workers=num_workers, pin_memory=True)  # , sampler=train_sampler)

        no_aug_train_dataset = TruncatedImageFolder(
            '/data02/datasets/'+directory+'/train/',
            transform=test_augment, length=start)

        no_aug_train_loader = torch.utils.data.DataLoader(
            no_aug_train_dataset, batch_size=batch_size, shuffle=train_shuffle,
            num_workers=num_workers, pin_memory=True)  # , sampler=train_sampler)

        test_dataset = TruncatedImageFolder(
            '/data02/datasets/'+directory+'/train/',
            transform=test_augment)

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)  # , sampler=train_sampler)

        if start != end:
            validation_dataset = TruncatedImageFolder(
                                '/data02/datasets/'+directory+'/train/',
                                transform=test_augment, start=start, length=end-start)
            no_aug_valid_loader = torch.utils.data.DataLoader(
                validation_dataset, batch_size=batch_size, shuffle=valid_shuffle,
                num_workers=num_workers, pin_memory=True)  # , sampler=train_sampler)
            return_dict['no-augment_valid'] = no_aug_valid_loader

        return_dict['train'] = train_loader
        return_dict['no-augment_train'] = no_aug_train_loader
        return_dict['test'] = test_loader

    return return_dict, num_class

