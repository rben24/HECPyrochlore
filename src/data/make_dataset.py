"""

HEADER STUFF

"""

import pandas as pd
import numpy as np
import logging
from pymatgen.core import Composition
import os

# from ..utils.globals import config_vars

# DATA_PATH = config_vars["DATA_PATH"]

# class Data:
#     def __init__(self, fname, from_DATA = False):
#         self.fname = fname
#         if ".csv" in fname:
#             if from_DATA:
#                 self.df = pd.read_csv(os.path.join(DATA_PATH, fname))

def standardize_composition(comp_string):
    '''

    :param comp_string:
    :return:
    '''

    try:
        comp = Composition(comp_string)
        return comp.reduced_formula
    except :
        pass

def config_id(df):
    '''

    :param df:
    :return:
    '''

def combine_df(df1, df2):
    first_col = "Composition"
    df1.rename(columns={df1.columns[0]: first_col}, inplace=True)
    df2.rename(columns={df2.columns[0]: first_col}, inplace=True)

    df = pd.concat([df1, df2], axis=0)

    return df

def create_intial_dataset():
    exp_file_path = '../../data/raw/Sample_Properties_Safin_Feb_2026.csv'
    exp_df = pd.read_csv(exp_file_path)
    data_file_path = '../../data/raw/notebookLM_dataset.csv'
    data_df = pd.read_csv(data_file_path)

    exp_df.drop_duplicates(inplace=True)
    data_df.drop_duplicates(inplace=True)

    df = combine_df(exp_df, data_df)

    df.to_csv('../../data/processed/pyrochlore_df.csv', index=False)
    logging.info('Saved pyrochlore_df.csv to data/processed\n')

if __name__ == "__main__":
    create_intial_dataset()