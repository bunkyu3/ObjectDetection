from omegaconf import OmegaConf
import os
import torch.optim as optim
from utils.log import *
from utils.utils import *
from data.custum_dataset import *
from data.custum_dataloader import *
from model.fasterrcnn import *


def evaluate(model, dataloader, device):
    model.eval()
    results = []
    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = batch_to_device(images, targets, device)            
            outputs = model(images)
            print(outputs)
            results.append(outputs)
    return results


def test(cfg):
    # データの取得
    test_loader = create_test_dataloader(cfg)
    # ネットワークと学習設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = set_fasterrcnn_model()
    model.load_state_dict(torch.load(cfg.save_dir.local.best_model))
    model.to(device)
    evaluate(model, test_loader, device)


if __name__ == '__main__':
    # ワーキングディレクトリの設定
    script_directory = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_directory)
    #乱数設定、ログ設定  
    set_seed(42)
    # config読み込み
    cfg = OmegaConf.load("./config/config.yaml")
    # Log制御インスタンス
    manager = LoggerManager(enable_mlflow=False)
    test(cfg)
