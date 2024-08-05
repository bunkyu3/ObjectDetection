from omegaconf import OmegaConf
import os
from utils.log import *
from utils.utils import *
from model.fasterrcnn import *
from analysis.visualization import *




def test(cfg):
    # データの取得
    # test_loader = create_test_dataloader(cfg)
    # # ネットワークと学習設定
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # state_dict = torch.load(cfg.save_dir.local.best_model, map_location=device)
    # model = set_fasterrcnn_model()
    # model.load_state_dict(state_dict)
    # model.to(device)
    # evaluate(model, test_loader, device)
    pass


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
