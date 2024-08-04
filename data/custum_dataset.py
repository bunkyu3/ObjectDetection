import torch
from torchvision import transforms
from torchvision.datasets import VOCDetection

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

class CustomVOCDetection(VOCDetection):
    def __init__(self, root="./data/raw", year="2012", image_set=None, download=False):
        super().__init__(root=root, year=year, image_set=image_set, download=download)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        org_img, org_target = super().__getitem__(index)
        if self.transform == None:
            return org_img, org_target
        
        # imgに対する前処理
        new_img = self.transform(org_img)
        # targetに対する前処理
        # 1. バウンディングボックスのリサイズ、リスト化
        new_bboxes = []
        org_w, org_h = org_img.size
        _, new_h, new_w = new_img.shape
        for org_obj in org_target["annotation"]["object"]:
            org_bbox = org_obj["bndbox"]
            new_xmin = int(int(org_bbox["xmin"]) * new_w / org_w)
            new_xmax = int(int(org_bbox["xmax"]) * new_w / org_w)
            new_ymin = int(int(org_bbox["ymin"]) * new_h / org_h)
            new_ymax = int(int(org_bbox["ymax"]) * new_h / org_h)
            new_bboxes.append([new_xmin, new_ymin, new_xmax, new_ymax])
        # 2. クラスのインデックス取得、リスト化
        labels = []
        for org_obj in org_target["annotation"]["object"]:
            class_name = org_obj["name"]
            class_index = VOC_CLASSES.index(class_name)
            labels.append(class_index)
        # 3. 座標とラベルをテンソル化、辞書格納
        target = {
            "boxes": torch.tensor(new_bboxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }
        return new_img, target