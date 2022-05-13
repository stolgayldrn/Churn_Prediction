# -*- coding: utf-8 -*-

__info__ = 'Delivery Hero - Case Study-- S.Tolga Yildiran'
__since__ = '01-05-2022'

import argparse
import logging
import os.path

import pandas as pd
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #Read all data
    labeled_data_dir = os.path.join('data', 'machine_learning_challenge_labeled_data.csv')
    labeled_data = pd.read_csv(labeled_data_dir, delimiter=',')

    order_data_dir = os.path.join('data', 'machine_learning_challenge_order_data.csv')
    order_data = pd.read_csv(order_data_dir, delimiter=',')

    #logging.INFO('read data success')
    print(order_data.head(1))
    ##PreProcessing with data
    item_counts_for_customer = order_data['customer_id'].value_counts()
    print(item_counts_for_customer.head(10))
    print('=' * 30)
    print(item_counts_for_customer.tail(10))
    print(item_counts_for_customer[1].mean())


    #Data Transformation


    #Create eval dataset


    #Define Model


    #Train Model

    #Load best model

    #Test model

