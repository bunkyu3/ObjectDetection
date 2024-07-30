
import os, sys
import numpy as np
import torchvision
from PIL import Image, ImageDraw
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from data.custum_dataset import CustomVOCDetection, VOC_CLASSES


def draw_bbox(img, xmin, xmax, ymin, ymax, class_name):
    draw = ImageDraw.Draw(img)
    draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)
    text_size = draw.textbbox((xmin, ymin), class_name)
    text_width = text_size[2] - text_size[0]
    text_height = text_size[3] - text_size[1]
    draw.rectangle([xmin, ymin - text_height, xmin + text_width, ymin], fill="red")
    draw.text((xmin, ymin - text_height), class_name, fill="white")    
    return img


def draw_bboxes_on_pil(img, target):
    for obj in target["annotation"]["object"]:
        bbox = obj["bndbox"]
        xmin = int(bbox["xmin"])
        xmax = int(bbox["xmax"])
        ymin = int(bbox["ymin"])
        ymax = int(bbox["ymax"])
        class_name = obj["name"]
        img = draw_bbox(img, xmin, xmax, ymin, ymax, class_name)
    return img


def draw_bboxes_on_tensor(t_img, target):
    p_img = torchvision.transforms.functional.to_pil_image(t_img)
    for t_bbox in target["boxes"]:
        np_bbox = t_bbox.numpy()
        xmin = int(np_bbox[0])
        ymin = int(np_bbox[1])
        xmax = int(np_bbox[2])
        ymax = int(np_bbox[3])
        p_img = draw_bbox(p_img, xmin, xmax, ymin, ymax, "a")

    # img_pil = Image.fromarray(img_np)
    # print(target[0]["boxes"].shape)

    # for obj in target["annotation"]["object"]:
    #     bbox = obj["bndbox"]
    #     xmin = int(bbox["xmin"])
    #     xmax = int(bbox["xmax"])
    #     ymin = int(bbox["ymin"])
    #     ymax = int(bbox["ymax"])
    #     class_name = obj["name"]
    #     img = draw_bbox(img, xmin, xmax, ymin, ymax, class_name)
    return p_img


def main():
    dataset = CustomVOCDetection(root='./data/raw', year='2012', image_set='train',
                                      download=False, transform=None)
    # 画像数
    dataset_no = len(dataset)
    print(len(dataset))

    for i, (img, target) in enumerate(dataset):
        img = draw_bboxes_on_pil(img, target)
        img.save(f"./data/raw/pngs/{i}.png")
        if i == 2:
            break


if __name__ == '__main__':
    main()
