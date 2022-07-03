#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
import torch.utils.data
import torchvision
from PIL import Image
from pycocotools.coco import COCO
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from engine import train_one_epoch, evaluate


# In[2]:


class CoCoDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]["file_name"]
        # open the input image
        img = Image.open(os.path.join(self.root, path))
        # number of objects in the image
        num_objs = len(coco_annotation)
        # Size of bbox (Rectangular)
        areas = 0

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        for i in range(num_objs):
            xmin = coco_annotation[i]["bbox"][0]
            ymin = coco_annotation[i]["bbox"][1]
            xmax = xmin + coco_annotation[i]["bbox"][2]
            ymax = ymin + coco_annotation[i]["bbox"][3]
            
            if xmin == xmax or ymin == ymax:                 
                continue             
            else:                 
                boxes.append([xmin, ymin, xmax, ymax])
            areas += (xmax-xmin) * (ymax-ymin)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        # Handle empty bounding boxes
        if num_objs == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
        
        # Labels (In my case, I only one class: target class or background)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        # Tensorise img_id
        img_id = torch.tensor([img_id])
            
        areas = torch.as_tensor(areas, dtype=torch.float32)
        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, my_annotation

    def __len__(self):
        return len(self.ids)


# In[3]:


# Only added ToTensor
def get_transform():
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(custom_transforms)


# collate_fn needs for batch
def collate_fn(batch):
    return tuple(zip(*batch))


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# In[4]:


train_dataset = CoCoDataset(
    root="./train2017/", 
    annotation='annotations/instances_train2017.json', 
    transforms=get_transform()
)
print(len(train_dataset))

valid_dataset = CoCoDataset(
    root="./val2017/", 
    annotation='annotations/instances_val2017.json', 
    transforms=get_transform()
)


# In[5]:


train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=3,
    collate_fn=collate_fn,
    pin_memory=True,
)
print(len(train_loader))

valid_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=3,
    collate_fn=collate_fn,
    pin_memory=True,
)


# In[6]:


num_epochs = 1
num_classes = 100
MODEL_PATH = 'E6.pth'
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = get_model_instance_segmentation(num_classes)
model.load_state_dict(torch.load(MODEL_PATH))
model.to(device)
print(device)


# In[7]:


params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params, 
    lr=0.005, 
    momentum=0.9, 
    weight_decay=0.005
)

lr_scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=3,
    gamma=0.1
)


# In[8]:


for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=1667)
    lr_scheduler.step()
    evaluate(model, valid_loader, device=device)
    
print("That's it!")


# In[9]:


torch.save(model.state_dict(), 'E7.pth')

input("Press Enter to continue...")

