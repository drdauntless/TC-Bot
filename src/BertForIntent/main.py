import os
import torch
import gspread
import os
from oauth2client.client import GoogleCredentials
import wandb
import pandas as pd
import sys
sys.setrecursionlimit(10000)

from Settings import Settings
from Sheet import Sheet
from ClassifierSuite import ClassifierSuite
if __name__ == "__main__":
    #settings = Settings('settings.json')
    hh_sheet = Sheet('hh')
    ha_sheet = Sheet('ha')
    hh_dataframe = hh_sheet.getAsDF()
    ha_dataframe = ha_sheet.getAsDF()
    CUDA_VISIBLE_DEVICES = 0
    suite = ClassifierSuite()
    suite.run(hh_dataframe, ha_dataframe)'''