"""

HEADER STUFF

"""

import pandas as pd
import numpy as np
import os

from m

DATA_PATH = config_vars["DATA_PATH"]

class Data:
    def __init__(self, fname, from_DATA = False):
        self.fname = fname
        if ".csv" in fname:
            if from_DATA:
                self.df = pd.read_csv(os.path.join(DATA_PATH, fname))
