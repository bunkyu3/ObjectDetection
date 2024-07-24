from PIL import ImageDraw
from torchvision import transforms


def save_image_with_bboxes(image, target, output_path='output_image.png'):
    image = transforms.functional.to_pil_image(image)
    draw = ImageDraw.Draw(image)

    for obj in target["annotation"]["object"]:
        bbox = obj["bndbox"]
        xmin = int(bbox["xmin"])
        ymin = int(bbox["ymin"])
        xmax = int(bbox["xmax"])
        ymax = int(bbox["ymax"])
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=2)

        category = obj["name"]        
        text_size = draw.textbbox((xmin, ymin), category)
        text_width = text_size[2] - text_size[0]
        text_height = text_size[3] - text_size[1]
        draw.rectangle([xmin, ymin - text_height, xmin + text_width, ymin], fill="red")
        draw.text((xmin, ymin - text_height), category, fill="white")

    # 画像を保存
    image.save(output_path)