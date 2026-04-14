import os, sys
from monty.serialization import loadfn

DEFAULT_PATH_RAW = "data/raw"

DATA_PATH = "data/processed/pyrochlore_df.csv"
TARGET_COL = "target"
RANDOM_STATE = 42
TEST_SIZE = 0.2

try:
    config_vars = loadfn(os.path.join(os.path.expanduser('~'), 'myml4hea.yaml'))
except:
    sys.exit('No myml4hea.yaml file was found. Please configure the '
             ' myml4hea.yaml and put it in your home directory.')

