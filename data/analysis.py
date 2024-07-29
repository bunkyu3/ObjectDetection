import os
import numpy as np
from PIL import ImageDraw
from custum_dataset import CustomVOCDetection, VOC_CLASSES


def draw_bboxes(img, target):
    draw = ImageDraw.Draw(img)

    for obj in target["annotation"]["object"]:
        bbox = obj["bndbox"]
        xmin = int(bbox["xmin"])
        xmax = int(bbox["xmax"])
        ymin = int(bbox["ymin"])
        ymax = int(bbox["ymax"])
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)
        class_name = obj["name"]
        text_size = draw.textbbox((xmin, ymin), class_name)
        text_width = text_size[2] - text_size[0]
        text_height = text_size[3] - text_size[1]
        draw.rectangle([xmin, ymin - text_height, xmin + text_width, ymin], fill="red")
        draw.text((xmin, ymin - text_height), class_name, fill="white")    
    return img


def main():
    dataset = CustomVOCDetection(root='./data/raw', year='2012', image_set='train',
                                      download=False, transform=None)
    # 画像数
    dataset_no = len(dataset)
    print(len(dataset))

    for i, (img, target) in enumerate(dataset):
        img = draw_bboxes(img, target)
        img.save(f"./data/raw/pngs/{i}.png")
        # if i == 2:
        #     break


if __name__ == '__main__':
    main()
