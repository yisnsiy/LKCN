import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), ".")))

import pandas as pd

from run_me import run
from src.data.processor.medical_data.prepare_data import prepare_data, processed_dir
from src.data.processor.medical_data.split_data  import split_data


sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), ".")))



if __name__ == '__main__':
    results = pd.DataFrame()
    
    # breast cancer
    study_name = [
        'brca_igr_2015', 
        'brca_mbcproject_wagle_2017', 
        'brca_mbcproject_2022', 
        'brca_tcga_pan_can_atlas_2018'
    ]
    prepare_data(study_name)
    split_data()
    res = run(
        model_name='breast-network_reactome-layer_3-gcn_no-know_yes-cnt_0',
        dataset_name='breast'
    )
    results = results.append(res, ignore_index=True)  # cof = 24
    # breast-network_reactome-layer_5-gcn_no-know_yes-cnt_0 end, result is {'model_name': 'breast-network_reactome-layer_5-gcn_no-know_yes-cnt_0', 'accuracy': 0.879746835443038, 'precision': 0.7916666666666666, 'f1': 0.7999999999999999, 'recall': 0.8085106382978723, 'aupr': 0.8204522982271077, 'auc': 0.8920835729346367
    # 0.879747   0.791667  0.8  0.808511  0.820452  0.892084         [890, 367]       breast

    res = run(
        model_name='breast-network_reactome-layer_3-gcn_no-know_no-cnt_0',
        dataset_name='breast'
    )
    results = results.append(res, ignore_index=True)

    # prostate
    study_name = ['pnet']
    prepare_data(study_name)
    split_data()
    res = run(
        model_name='prostate-network_reactome-layer_5-gcn_no-know_yes-cnt_0',
        dataset_name='prostate'
    )
    results = results.append(res, ignore_index=True)  # cof = 24
    # prostate-network_reactome-layer_5-gcn_no-know_...  0.882353    0.84375  0.818182  0.794118  0.872865  0.937284         [541, 266]     prostate
    # prostate-network_reactome-layer_5-gcn_no-know_yes-cnt_0 end, result is {'model_name': 'prostate-network_reactome-layer_5-gcn_no-know_yes-cnt_0', 'accuracy': 0.8823529411764706, 'precision': 0.84375, 'f1': 0.8181818181818182, 'recall': 0.7941176470588235, 'aupr': 0.8728648034353089, 'auc': 0.9372837370242215}

    res = run(
        model_name='prostate-network_reactome-layer_5-gcn_no-know_no-cnt_0',
        dataset_name='breast'
    )
    results = results.append(res, ignore_index=True)

    # lung
    study_name = [
        'luad_mskcc_2023_met_organotropism', # lung
    ]
    prepare_data(study_name)
    split_data()
    res = run(
        model_name='prostate-network_reactome-layer_5-gcn_no-know_yes-cnt_0',
        dataset_name='lung'
    )
    results = results.append(res, ignore_index=True)  # cof 14
    # prostate-network_reactome-layer_5-gcn_no-know_yes-cnt_0 end, result is {'model_name': 'prostate-network_reactome-layer_5-gcn_no-know_yes-cnt_0', 'accuracy': 0.602510460251046, 'precision': 0.744, 'f1': 0.6619217081850535, 'recall': 0.5961538461538461, 'aupr': 0.7919500776645524, 'auc': 0.6566265060240963}
    # 0.589958   0.741667  0.644928  0.570513  0.792877  0.659909 

    res = run(
        model_name='prostate-network_reactome-layer_5-gcn_no-know_no-cnt_0',
        dataset_name='lung'
    )
    results = results.append(res, ignore_index=True)

    res = run(
        model_name='breast-network_reactome-layer_3-gcn_no-know_yes-cnt_0',
        dataset_name='lung'
    )
    results = results.append(res, ignore_index=True)

    res = run(
        model_name='breast-network_reactome-layer_3-gcn_no-know_no-cnt_0',
        dataset_name='lung'
    )
    results = results.append(res, ignore_index=True)

    results.to_csv(os.path.join(processed_dir, 'results-different-cancer.csv'))
    print(results)
    
