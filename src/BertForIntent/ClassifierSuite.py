import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import torch
import ssl
from Settings import Settings
from Classifiers import *
import warnings

warnings.filterwarnings('ignore')
settings = Settings("settings.json")

class ClassifierSuite:

    def __init__(self):
        # self.bert = BertClassifier()
        # self.gpt2 = Gpt2()
        # self.albert = Albert()

        pass

    def __preprocess(self, dataframe):
        # Select only the dialogue and DA columns
        dataframe = dataframe[settings.columns_list].copy()
        # dataframe = dataframe[[settings.dialogue_column, settings.intent_column]].copy()

        # return dataframe
        # Encode the DA column
        # print(dataframe[settings.intent_column])

        dataframe[settings.dialogue_column].replace('', np.nan, inplace=True)
        dataframe[settings.intent_column].replace('', np.nan, inplace=True)
        dataframe = dataframe.dropna(axis=0)
        # print(dataframe[settings.intent_column])

        dataframe[settings.intent_column] = dataframe[settings.intent_column].astype(int, errors='ignore')

        return dataframe

    def split_dataset(self, hh_dataframe, ha_dataframe):
        train_dataframe, test_dataframe = train_test_split(ha_dataframe)
        # print("train_dataframe: " + str(train_dataframe.shape))
        # print("test_dataframe: " + str(test_dataframe.shape))

        train_dataframe = train_dataframe.append(hh_dataframe)
        # print("train_dataframe + hh: " + str(train_dataframe.shape))

        train_dataset = Dataset.from_pandas(train_dataframe)
        test_dataset = Dataset.from_pandas(test_dataframe)

        return train_dataset, test_dataset

    def split_dataset_ha(self, hh_dataframe, ha_dataframe):
        train_dataframe, test_dataframe = train_test_split(ha_dataframe)

        train_dataset = Dataset.from_pandas(train_dataframe)
        test_dataset = Dataset.from_pandas(test_dataframe)

        return train_dataset, test_dataset

    def split_separate(self, hh_dataframe, ha_dataframe):
        train_dataframe, test_dataframe = train_test_split(ha_dataframe)
        # print("train_dataframe: " + str(train_dataframe.shape))
        # print("test_dataframe: " + str(test_dataframe.shape))

        # train_dataframe = train_dataframe.append(hh_dataframe)
        # print("train_dataframe + hh: " + str(train_dataframe.shape))

        hh_dataset = Dataset.from_pandas(hh_dataframe)
        ha_dataset = Dataset.from_pandas(ha_dataframe)
        train_dataset = Dataset.from_pandas(train_dataframe)
        test_dataset = Dataset.from_pandas(test_dataframe)

        return train_dataset, test_dataset, ha_dataset, hh_dataset

    def run(self, hh_dataframe, ha_dataframe):
        hh_dataframe = self.__preprocess(hh_dataframe)
        ha_dataframe = self.__preprocess(ha_dataframe)
        print("HH Dataframe initial shape: ", hh_dataframe.shape)
        print("HA Dataframe initial shape: ", ha_dataframe.shape)
        # gpt2 = Gpt2(hh_dataframe, ha_dataframe)
        if (settings.stagger_training):
            train_dataset = []
            test_dataset = []
            for i in range(settings.num_runs):
                temp_train_dataset, temp_test_dataset, ha_dataset, hh_dataset = self.split_separate(hh_dataframe,
                                                                                                    ha_dataframe)
                print("Temp Train Dataset[", i, "] initial shape: ", temp_train_dataset.shape)
                print("Temp Test Dataset[", i, "] initial shape: ", temp_test_dataset.shape)
                train_dataset.append(temp_train_dataset)
                test_dataset.append(temp_test_dataset)
            print("HH Dataset shape: ", hh_dataset.shape)
            print("HA Dataset initial shape: ", ha_dataset.shape)
            BertClassifier(train_dataset, test_dataset, ha_dataset, hh_dataset)
        else:
            train_dataset = []
            test_dataset = []
            for i in range(settings.num_runs):
                temp_train_dataset, temp_test_dataset = self.split_dataset_ha(hh_dataframe, ha_dataframe)
                print("Temp Train Dataset[", i, "] initial shape: ", temp_train_dataset.shape)
                print("Temp Test Dataset[", i, "] initial shape: ", temp_test_dataset.shape)
                train_dataset.append(temp_train_dataset)
                test_dataset.append(temp_test_dataset)
            BertClassifier(train_dataset, test_dataset)
        #
        print("Bert Train")
        # bert =
        # gpt2 = Gpt2Classifier(train_dataset, test_dataset)
        # self.bert.train(hh_dataframe, ha_dataframe)
        # print("BERT Evaluate")
        # self.bert.evaluate(dataframe)
        # print("GPT2 Train")
        # self.gpt2.train(dataframe.copy())
        # print("GPT2 Evaluate")
        # self.gpt2.evaluate(dataframe)
        # print("Albert Train")
        # self.albert.train(hh_dataframe.copy(), ha_dataframe.copy())
        # print("Albert Evaluate")
        # self.albert.evaluate(dataframe)