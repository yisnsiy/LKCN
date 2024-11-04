import os
import sys
from os.path import dirname, join, exists
from copy import deepcopy
sys.path.insert(0, os.path.abspath(join(os.getcwd(), ".")))

from run_me import run
from src.utils.utils import read_json_data, write_json_data
from src.data.processor.medical_data.prepare_data import prepare_data, processed_dir
from src.data.processor.medical_data.split_data  import split_data
from src.data.dataset import set_cached_data_to_empty

import pandas as pd


if __name__ == '__main__':

    base_model_config_dir = join(dirname(dirname(__file__)), 'config/model')
    base_model_config = read_json_data(join(base_model_config_dir, 'bnet.json')),
    base_model_config = base_model_config[0]
    tests = []
    loss_weights = [2, 7, 20, 54, 148, 400, 900, 2000]
    protein_heir = join(dirname(dirname(processed_dir)), 'protein/selected_links.txt')

    # index
    cancers = [
        ('prostate', ['pnet']), 
        ('breast', [
            'brca_igr_2015', 
            'brca_mbcproject_wagle_2017', 
            'brca_mbcproject_2022', 
            'brca_tcga_pan_can_atlas_2018'
        ])
    ]
    networks = ['reactome', 'protein']
    number_of_layers = [i for i in range(2, 6)]
    gcn = [True, False]
    knowledge_trainable = [True, False]

    results = pd.DataFrame()
    for (cancer, study_name) in cancers:
        # change trainset, validset and testset by cancer
        prepare_data(study_name)
        split_data()  
        set_cached_data_to_empty()
        # different cancers have specific genes, so corresponding protein networks have to change
        if exists(protein_heir):
            os.remove(protein_heir)
        for network in networks:
            for i in number_of_layers:
                for g in gcn:
                    for k in knowledge_trainable:
                        for cnt in range(3):
                            if cnt < 2:
                                continue
                            config_file_name = f'{cancer}-network_{network}-' + \
                                            f'layer_{i}-gcn_{"yes" if g else "no"}-' + \
                                            f'know_{"yes" if k else "no"}-cnt_{cnt}'
                            model_config = deepcopy(base_model_config)

                            # change parameters in config file.
                            model_config['network'] = network
                            model_config['n_hidden_layers'] = i
                            model_config['loss_weights'] = loss_weights[0: i+1]
                            model_config['gcn'] = g
                            model_config['trainable_mask'] = k

                            # save config file
                            if not os.path.exists(
                                join(base_model_config_dir, config_file_name + '.json')
                            ):
                                write_json_data(
                                    model_config, 
                                    join(base_model_config_dir, config_file_name + '.json')
                                )

                            # based on config run the model
                            res = run(
                                model_name = config_file_name,
                                dataset_name = cancer
                            )

                            # record result
                            res['cancer'] = cancer
                            res['network'] = network
                            res['n_hidden_layers'] = i
                            res['gcn'] = g
                            res['trainable_mask'] = k
                            results = results.append(res, ignore_index=True)
    results.to_csv(os.path.join(processed_dir, 'joint_test.csv'))
    print(results)