from PIL import ImageDraw
from torchvision import transforms
from data.custum_dataset import *

def save_image_with_bboxes(image, target, output_path='output_image.png'):
    image = transforms.functional.to_pil_image(image)
    draw = ImageDraw.Draw(image)

    boxes = target["boxes"]
    boxes = boxes.numpy()
    boxes = boxes.astype(int)
    labels = target["labels"]
    labels = labels.numpy()
    labels = labels.astype(int)

    # print(boxes)
    for i in range(boxes.shape[0]):
        box = boxes[i,:]
        xmin, ymin, xmax, ymax = box
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)
        label = labels[i]

        category = VOC_CLASSES[label]
        print(category)
        text_size = draw.textbbox((xmin, ymin), category)
        text_width = text_size[2] - text_size[0]
        text_height = text_size[3] - text_size[1]
        draw.rectangle([xmin, ymin - text_height, xmin + text_width, ymin], fill="red")
        draw.text((xmin, ymin - text_height), category, fill="white")

    # for obj in target["boxes"]["object"]:
    #     bbox = obj["bndbox"]
    #     xmin = int(bbox["xmin"])
    #     ymin = int(bbox["ymin"])
    #     xmax = int(bbox["xmax"])
    #     ymax = int(bbox["ymax"])
    #     draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)

    #     category = obj["name"]        
    #     text_size = draw.textbbox((xmin, ymin), category)
    #     text_width = text_size[2] - text_size[0]
    #     text_height = text_size[3] - text_size[1]
    #     draw.rectangle([xmin, ymin - text_height, xmin + text_width, ymin], fill="red")
    #     draw.text((xmin, ymin - text_height), category, fill="white")

    # 画像を保存
    image.save(output_path)