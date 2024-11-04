import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), ".")))

from run_me import run
from src.data.processor.medical_data.prepare_data import processed_dir

import pandas as pd


if __name__ == '__main__':
    # model_name = "breast-network_reactome-layer_3-gcn_no-know_no-cnt_0"
    model_names = [
        "bnet-dense",
        "breast-network_reactome-layer_3-gcn_no-know_yes-cnt_0"
    ]

    for i in range(5):
        for model_name in model_names:
            res = run(
                model_name=model_name,
                dataset_name='breast-trainset0' # breast
                # dataset_name='prostate'
            )
    

    # 跑完去./results/目录下找对应的model_name+loss_accuracy.csv文件，然后去画图里面画出来。