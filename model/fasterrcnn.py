from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def set_fasterrcnn_model():
    num_classes = 21  # VOCデータセットのクラス数 (20クラス + 背景)
    # Pre-trained Faster R-CNN model
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
