from itertools import islice

import cv2
import detectron2
import webdataset as wds
import torch
from torch.utils.data import IterableDataset
from torchvision import transforms

url = "http://storage.googleapis.com/nvdata-openimages/openimages-train-000000.tar"
url = f"pipe:curl -L -s {url} || true"


# Visualise for fun

dataset = (
    wds.WebDataset(url)
    .shuffle(100)
    .decode("rgb8")
    .to_tuple("jpg;png", "json")
)

for i, res in enumerate(islice(dataset, 0, 3)):
    image, data = res
    print(image.shape, image.dtype, type(data))
    cv2.imwrite(f'{i}.jpg', image[:,:,::-1])

# get Torch tensors
print('Getting Torch dataset')

def identity(x):
    return x

def identity2(x):
    return x

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])

preproc = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

dataset = (
    wds.WebDataset(url)
    .shuffle(100)
    .decode("pil")
    .to_tuple("jpg;png", "json")
    .map_tuple(preproc, identity, identity2)
)

for image, data in islice(dataset, 0, 3):
    print(image.shape, image.dtype, type(data))

# batch_size = 20
# dataloader = torch.utils.data.DataLoader(dataset.batched(batch_size), num_workers=4, batch_size=None)
# images, targets = next(iter(dataloader))
# images.shape

from detectron2.data.build import build_detection_train_loader
loader = build_detection_train_loader(dataset.batched(2), mapper=None, total_batch_size=2)