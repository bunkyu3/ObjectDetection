from omegaconf import OmegaConf
import os
from utils.log import *
from utils.utils import *
from data.custum_dataloader import *
from model.fasterrcnn import *
from voc_data_visualization import *


def evaluate(model, dataloader, device):
    model.eval()
    results = []
    with torch.no_grad():
        for image, target in dataloader:
            image, target = batch_to_device(image, target, device)            
            output = model(image)
            print("***********")
            print(target[0])
            # print("-----------")
            # print(target[0]["boxes"])
            # print(target[0]["boxes"].shape)
            # print(output)
            save_image_with_bboxes(image[0], target[0])
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
