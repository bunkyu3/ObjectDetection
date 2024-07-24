import os
import hydra
from omegaconf import DictConfig
from utils import *
from custum_dataset import *
from custum_dataloader import *
from optimizer import *
from model import *


def set_cwd():
    script_directory = os.path.dirname(os.path.abspath(__file__))
    parent_directory = os.path.dirname(script_directory)
    os.chdir(parent_directory)


def train_one_epoch(epoch, model, dataloader, device, optimizer):
    model.train()
    for images, targets in dataloader:
        images, targets = batch_to_device(images, targets, device)            
        loss_dict = model(images, targets)
        print(loss_dict)
        # losses = sum(loss for loss in loss_dict.values())
        # optimizer.zero_grad()
        # losses.backward()
        # optimizer.step()


@hydra.main(config_name="config", version_base=None, config_path="../config")
def train(cfg: DictConfig) -> None:
    # ワーキングディレクトリの設定、乱数固定
    set_cwd()
    set_seed(42)

    # データの取得
    train_loader, val_loader = create_train_val_dataloader(cfg)

    # モデル、デバイス、オプティマイザの設定
    device = torch.device(cfg.train.device if torch.cuda.is_available() else 'cpu')
    model = set_fasterrcnn_model()
    model.to(device)
    optimizer = set_optimizer(model, cfg)

    # 学習ループ
    for epoch in range(cfg.train.num_epochs):
        train_one_epoch(epoch, model, train_loader, device, optimizer)
        # evaluate(epoch, model, val_loader, criterion)


if __name__ == '__main__':
    train()
