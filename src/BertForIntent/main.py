import os
#import torch
import os
#from oauth2client.client import GoogleCredentials

import pandas as pd
import sys
sys.setrecursionlimit(10000)
import wandb

from Settings import *
from Sheet import *
from ClassifierSuite import ClassifierSuite
if __name__ == "__main__":
    os.environ["WANDB_WATCH"] = "false"
    wandb.login(key = '6175fddbcbeb3088964faa63261303bafff5995c')
    wandb.init()
    settings = Settings('settings.json')
    hh_sheet = Sheet('hh')
    ha_sheet = Sheet('ha')
    hh_dataframe = hh_sheet.getAsDF()
    ha_dataframe = ha_sheet.getAsDF()
    print(hh_dataframe)
    CUDA_VISIBLE_DEVICES = 0
    suite = ClassifierSuite()
    suite.run(hh_dataframe, ha_dataframe)