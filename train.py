import os
import hydra
from omegaconf import OmegaConf
from utils.log import *
from utils.utils import *
from data.custum_dataset import CustomVOCDetection
from model.fasterrcnn import set_fasterrcnn_model


def train_one_epoch(epoch, model, dataloader, device, optimizer):
    model.train()
    for images, targets in dataloader:
        images, targets = batch_to_device(images, targets, device)            
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()


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
        # evaluate(epoch, model, val_loader, criterion)
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