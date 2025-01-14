from typing import Optional
import os
from .imagelist import ImageList
from ._util import download as download_data, check_exits


class PACS(ImageList):
    """`PACS Dataset <https://domaingeneralization.github.io/#data>`_.

    Args:
        root (str): Root directory of dataset
        task (str): The task (domain) to create dataset. Choices include ``'A'``: amazon, \
            ``'D'``: dslr and ``'W'``: webcam.
        download (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, :class:`torchvision.transforms.RandomCrop`.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: In `root`, there will exist following files after downloading.
        ::
            art_painting/
                dog/
                    *.jpg
                    ...
            cartoon/
            photo/
            sketch
            image_list/
                art_painting.txt
                cartoon.txt
                photo.txt
                sketch.txt
    """
    download_list = [
        ("image_list", "image_list.zip", "https://drive.usercontent.google.com/download?id=1pFH6Ro5al5KdgnON8uhA_JllQ3B_9n_g&export=download&authuser=0"),
        ("art_painting", "art_painting.tgz", "https://drive.usercontent.google.com/download?id=15MCPsU-9zBrimi1FzGSoB7ZG1Kg4jzPW&export=download&authuser=0"),
        ("cartoon", "cartoon.tgz", "https://drive.usercontent.google.com/download?id=10XxReJt281dLbKqJJD5Zs7E0mYNXrZu8&export=download&authuser=0"),
        ("photo", "photo.tgz", "https://drive.usercontent.google.com/download?id=1BnXh9SuJzUQAVMYEpdLqizVxSGkmovHj&export=download&authuser=0"),
        ("sketch", "sketch.tgz", "https://drive.usercontent.google.com/download?id=1_yIGPjqhwSPjdz749YjmZAJo8B3Vv-cl&export=download&authuser=0"),
    ]
    image_list = {
        "A": "image_list/art_painting_{}.txt",
        "C": "image_list/cartoon_{}.txt",
        "P": "image_list/photo_{}.txt",
        "S": "image_list/sketch_{}.txt"
    }
    CLASSES = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']

    def __init__(self, root: str, task: str, split='all', download: Optional[bool] = True, **kwargs):
        assert task in self.image_list
        assert split in ["train", "val", "all", "test"]
        if split == "test":
            split = "all"
        data_list_file = os.path.join(root, self.image_list[task].format(split))

        if download:
            list(map(lambda args: download_data(root, *args), self.download_list))
        else:
            list(map(lambda file_name, _: check_exits(root, file_name), self.download_list))

        super(PACS, self).__init__(root, PACS.CLASSES, data_list_file=data_list_file, target_transform=lambda x: x - 1,
                                   **kwargs)

    @classmethod
    def domains(cls):
        return list(cls.image_list.keys())
