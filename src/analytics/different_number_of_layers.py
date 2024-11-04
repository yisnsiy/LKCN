import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), ".")))

from run_me import run
from src.data.processor.medical_data.prepare_data import processed_dir

import pandas as pd


if __name__ == '__main__':
    model_names = [
        'bnet-layer1',
        'bnet-layer2',
        'bnet-layer3',
        'bnet-layer4',
        'bnet',
        'bnet-layer6',
    ]
    results = pd.DataFrame()
    for model_names in model_names:

        res = run(
            model_name=model_names,
            dataset_name='breast'
            # dataset_name='prostate'
        )
        results = results.append(res, ignore_index=True)
    results.to_csv(os.path.join(processed_dir, 'results-different-number-of-layer.csv'))
    print(results)