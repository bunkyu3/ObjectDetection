import os
import hydra
from omegaconf import OmegaConf
from utils.log import *
from utils.utils import *
from utils.metrics import calculate_metrics
from data.custum_dataset import CustomVOCDetection
from model.fasterrcnn import set_fasterrcnn_model


def evaluate(model, dataloader, device, epoch=None):
    total_loss = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = batch_to_device(images, targets, device)
            # ロスの算出
            model.train()       
            loss_dict = model(images, targets)
            loss = sum(l for l in loss_dict.values())
            total_loss += loss.item()
            # 混合行列の算出
            model.eval()
            outputs = model(images, targets)
            for output, target in zip(outputs, targets):
                pred_boxes = output["boxes"]
                pred_labels = output["labels"]
                true_boxes = target["boxes"]
                true_labels = target["labels"]
                tp, fp, fn = calculate_metrics(pred_boxes, pred_labels, true_boxes, true_labels)
                total_tp += tp
                total_fp += fp
                total_fn += fn
    val_loss = total_loss / len(dataloader)
    print("val_loss: ", val_loss)
    print("tp, fp, fn: ", tp, fp, fn)

def train_one_epoch(epoch, model, dataloader, device, optimizer):
    total_loss = 0
    model.train()
    for images, targets in dataloader:
        images, targets = batch_to_device(images, targets, device)            
        loss_dict = model(images, targets)
        loss = sum(l for l in loss_dict.values())
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss = total_loss / len(dataloader)
    print("train_loss: ", train_loss)

def train(cfg):
    # データの取得
    full_dataset = CustomVOCDetection(image_set="train")
    small_dataset = create_subset(full_dataset, cfg.dataset.subset_size_ratio)
    train_dataset, val_dataset = split_dataset(small_dataset, cfg.dataset.train_size_ratio)
    batch_size = cfg.train_param.batch_size
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                            shuffle=False, collate_fn=collate_fn)
    # 学習設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = set_fasterrcnn_model()
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, 
                                lr=cfg.train_param.learning_rate, 
                                momentum=cfg.train_param.momentum, 
                                weight_decay=cfg.train_param.weight_decay)
    # 学習ループ
    for epoch in range(cfg.train_param.num_epochs):
        print(epoch)
        train_one_epoch(epoch, model, train_loader, device, optimizer)
        evaluate(model, val_loader, device, epoch)
    manager.run(BestModelLogger(cfg, model))


if __name__ == '__main__':
    # ワーキングディレクトリの設定
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)
    # 乱数設定、ログ設定  
    set_seed(42)
    # config読み込み
    cfg = OmegaConf.load("./config/config.yaml")
    # Log制御インスタンス
    manager = LoggerManager(enable_mlflow=False)
    # 学習
    train(cfg)