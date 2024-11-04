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

    dataset_names = [
        'breast-trainset0',
        'breast-trainset1',
        'breast-trainset2',
        'breast-trainset3',
        'breast-trainset4',
        'breast-trainset5',
        'breast-trainset6',
        'breast-trainset7',
        'breast-trainset8',
        'breast-trainset9',
        'breast-trainset10',
        'breast-trainset11',
        'breast-trainset12',
        'breast-trainset13',
        'breast-trainset14',
        'breast-trainset15',
        'breast-trainset16',
        'breast-trainset17',
        'breast-trainset18',
        'breast-trainset19',
    ]

    results = pd.DataFrame()
    for model_name in model_names:
        for data_name in dataset_names:
            res = run(
                model_name=model_name,
                dataset_name=data_name
                # dataset_name='prostate'
            )
            results = results.append(res, ignore_index=True)
    results.to_csv(os.path.join(processed_dir, 'results-different-number-of-input.csv'))
    print(results)