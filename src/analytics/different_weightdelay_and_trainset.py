import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), ".")))

from run_me import run
from src.data.processor.medical_data.prepare_data import processed_dir

import pandas as pd


if __name__ == '__main__':
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
    
    model_names = [
        'bnet-weightdelay0.01',
        'bnet-weightdelay0.008',
        'bnet-weightdelay0.004',
        'bnet-weightdelay0.002',
        'bnet-weightdelay0.001',
        'bnet-weightdelay0.0008',
        'bnet-weightdelay0.0004',
        'bnet-weightdelay0.0002',
        'bnet-weightdelay0.0001',
    ]
    results = pd.DataFrame()
    for dataset_name in dataset_names:
        res = run(
            model_name='bnet',
            dataset_name=dataset_name
            # dataset_name='prostate'
        )
        results = results.append(res, ignore_index=True)
    for model_names in model_names:
        res = run(
            model_name=model_names,
            dataset_name='breast'
            # dataset_name='prostate'
        )
        results = results.append(res, ignore_index=True)
    results.to_csv(os.path.join(processed_dir, 'results-different-weightdelay.csv'))
    

    print(results)