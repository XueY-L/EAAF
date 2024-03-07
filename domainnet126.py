import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

def load_image(img_path):
    img = Image.open(img_path)
    img = img.convert("RGB")
    return img


class ImageList(Dataset):
    def __init__(
        self,
        image_root: str,
        label_file: str,
        transform=None,
        batch_idx=None,
        pseudo_item_list=None,
    ):
        self.image_root = image_root
        self._label_file = label_file
        self.transform = transform

        assert (
            label_file or pseudo_item_list
        ), f"Must provide either label file or pseudo labels."
        self.item_list = (
            self.build_index(label_file) if label_file else pseudo_item_list
        )
        if batch_idx != None: 
            self.item_list = self.item_list[batch_idx*50 : (batch_idx+1)*50]

    def build_index(self, label_file):
        """Build a list of <image path, class label> items.

        Args:
            label_file: path to the domain-net label file

        Returns:
            item_list: a list of <image path, class label> items.
        """
        # read in items; each item takes one line
        with open(label_file, "r") as fd:
            lines = fd.readlines()
        lines = [line.strip() for line in lines if line]

        item_list = []
        for item in lines:
            img_file, label = item.split()
            img_path = os.path.join(self.image_root, img_file)
            label = int(label)
            item_list.append((img_path, label, img_file))

        return item_list

    def __getitem__(self, idx):
        """Retrieve data for one item.

        Args:
            idx: index of the dataset item.
        Returns:
            img: <C, H, W> tensor of an image
            label: int or <C, > tensor, the corresponding class label. when using raw label
                file return int, when using pseudo label list return <C, > tensor.
        """
        img_path, label, _ = self.item_list[idx]
        img = load_image(img_path)
        if self.transform:
            img = self.transform(img)

        return img, label, idx

    def __len__(self):
        return len(self.item_list)
    

def get_domainnet126(image_root, src_domain, bs, phase, shuffle, batch_idx=None):
    if batch_idx != None:  # 训练源模型用的
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else: # new_aug
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    label_file = os.path.join(image_root, f"{src_domain}_list.txt")
    if phase == 'test':
        if batch_idx != None:
            data_dataset = ImageList(image_root, label_file, test_transform, batch_idx)
        else:  # 不指定某个batch，返回所有图片
            data_dataset = ImageList(image_root, label_file, test_transform)
    elif phase == 'train':
        if batch_idx != None: 
            data_dataset = ImageList(image_root, label_file, train_transform, batch_idx)
        else:  # 不指定某个batch，返回所有图片
            data_dataset = ImageList(image_root, label_file, train_transform)

    loader = DataLoader(
        data_dataset,
        batch_size=bs,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=16,
    )
    return loader