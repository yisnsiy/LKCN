import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), ".")))

from src.data.processor.medical_data.prepare_data import prepare_data, processed_dir
from src.data.processor.medical_data.split_data  import split_data
from src.config.configuration import Config
from src.data.dataset import BnetData

import pandas as pd

def get_breast_data():
    study_name = [
        'brca_igr_2015',
        'brca_mbcproject_wagle_2017',
        'brca_mbcproject_2022',
        'brca_tcga_pan_can_atlas_2018'
    ]

    prepare_data(study_name)
    split_data()
    breast_config = Config(
        model_name="prostate-network_reactome-layer_5-gcn_no-know_yes-cnt_0",
        dataset_name="breast"
    )

    breast_dataset = BnetData(breast_config)
    return pd.DataFrame(breast_dataset.x, columns=breast_dataset.columns, index=breast_dataset.info)

def get_cancer_data(study_name, dataset_name):

    prepare_data(study_name)
    split_data()
    prostate_config = Config(
        model_name="prostate-network_reactome-layer_5-gcn_no-know_yes-cnt_0",
        dataset_name=dataset_name
    )

    prostate_dataset = BnetData(prostate_config)
    return pd.DataFrame(prostate_dataset.x, columns=prostate_dataset.columns, index=prostate_dataset.info)

if __name__ == '__main__':
    breast_pd = get_cancer_data(
        [
            'brca_igr_2015',
            'brca_mbcproject_wagle_2017',
            'brca_mbcproject_2022',
            'brca_tcga_pan_can_atlas_2018'
        ],
        "breast"
    )
    prostate_pd = get_cancer_data(
        ['pnet'],
        "prostate"
    )
    lung_pd = get_cancer_data(
        ["luad_mskcc_2023_met_organotropism"],
        "lung"
    )

    # 1. Find the intersection of genes
    gene_set_a = set(breast_pd.columns.get_level_values(0))
    gene_set_b = set(prostate_pd.columns.get_level_values(0))
    gene_set_c = set(lung_pd.columns.get_level_values(0))

    gene_set = gene_set_a.intersection(gene_set_b).intersection(gene_set_c)
    gene_set = sorted(list(gene_set))  # Convert to sorted list for consistent order


    # 2. Crop the DataFrames based on the intersection
    alterations = ['mut_important', 'cnv_del', 'cnv_amp']
    a_crop = breast_pd[[(gene, alteration) for gene in gene_set for alteration in alterations]]
    b_crop = prostate_pd[[(gene, alteration) for gene in gene_set for alteration in alterations]]
    c_crop = lung_pd[[(gene, alteration) for gene in gene_set for alteration in alterations]]


    # 3. Flatten the MultiIndex of a_crop
    a_crop.columns = ['_'.join(col) for col in a_crop.columns]


    # Print results (optional)
    print("Intersection of genes:", gene_set)
    print("\na_crop:\n", a_crop)
    print("\nb_crop:\n", b_crop)
    print("\nc_crop:\n", c_crop)
