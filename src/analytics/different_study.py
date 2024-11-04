import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), ".")))

from src.data.processor.medical_data.split_data import split_data
from src.data.processor.medical_data.prepare_data import prepare_data, processed_dir
from run_me import run

import pandas as pd





if __name__ == '__main__':
    success_task = ['brca_igr_2015', 'brca_mbcproject_wagle_2017', 'brca_mbcproject_2022', 'brca_tcga_pan_can_atlas_2018']
    task0 = ['pnet']
    previous = ['brca_igr_2015', 'brca_mbcproject_wagle_2017', 'brca_tcga_pan_can_atlas_2018']
    task1 = ['brca_igr_2015', 'brca_tcga_pan_can_atlas_2018']
    task2 = ['brca_igr_2015', 'brca_tcga_pub2015']
    task3 = ['brca_igr_2015', 'brca_mbcproject_wagle_2017', 'brca_mbcproject_2022', 'brca_tcga_pub2015']
    task4 = ['brca_igr_2015', 'brca_mbcproject_wagle_2017', 'brca_mbcproject_2022', 'brca_tcga_pan_can_atlas_2018']
    task5 = ['brca_igr_2015', 'brca_mbcproject_wagle_2017', 'brca_mbcproject_2022', 'brca_tcga_pub2015', 'brca_tcga', 'brca_tcga_pan_can_atlas_2018']

    task6 = ['brca_igr_2015', 'brca_mbcproject_wagle_2017', 'brca_mbcproject_2022', 'brca_tcga']

    tasks = [task6]
    results = pd.DataFrame()
    for study_name in tasks:
        print(study_name)

        prepare_data(study_name)
        split_data()

        res = run(
            model_name='bnet',
            # model_name='test',
            dataset_name='breast'
            # dataset_name='prostate'
        )
        res['study_name'] = study_name
        results = results.append(res, ignore_index=True)
        
        # results.set_index('model_name', drop=True, inplace=True)

        if isinstance(study_name, list) and len(study_name) == 1:
            statistic = pd.read_csv(os.path.join(processed_dir, "statistic.csv"), index_col=0)
            statistic.loc[study_name[0], "result_300epochs"] = str(res)
            statistic.to_csv(os.path.join(processed_dir, "statistic.csv"))
    # results.to_csv(os.path.join(processed_dir, 'results.csv'))
    print(results)