from omegaconf import OmegaConf
import os
from utils.log import *
from utils.utils import *
from data.custum_dataloader import *
from model.fasterrcnn import *
from analysis.visualization import *


def evaluate(model, dataloader, device):
    model.eval()
    results = []
    with torch.no_grad():
        i = 0
        for imgs, targets in dataloader:
            imgs, targets = batch_to_device(imgs, targets, device)            
            outputs = model(imgs)
            for img, target, output in zip(imgs, targets, outputs):
                img_with_bbox = draw_bboxes_on_tensor(img, target)
                img_with_bbox.save(f"./data/raw/pngs/{i}.png")
                print(output)
                img_with_bbox = draw_bboxes_on_tensor(img, output)
                i = i + 1
            
            # print(target[0])
            # print("-----------")
            # print(target[0]["boxes"])
            # print(target[0]["boxes"].shape)
            # print(output)
            # break
            results.append(output)
    return results


def test(cfg):
    # データの取得
    test_loader = create_test_dataloader(cfg)
    # ネットワークと学習設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(cfg.save_dir.local.best_model, map_location=device)
    model = set_fasterrcnn_model()
    model.load_state_dict(state_dict)
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
