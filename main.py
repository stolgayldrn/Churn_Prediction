# -*- coding: utf-8 -*-

__info__ = 'Delivery Hero - Case Study-- S.Tolga Yildiran'
__since__ = '15-05-2022'

import argparse
import os.path
from datetime import date
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier



def is_lth_6month(date_1, date_2):
    days = (date_1 - date_2).days
    return 1 if days <= 180 else 0

#Performance evaluation
def print_scores(alg_name, y_true, y_pred):
    print(alg_name)
    acc_score = accuracy_score(y_true, y_pred)
    print("accuracy: ",acc_score)
    pre_score = precision_score(y_true, y_pred)
    print("precision: ",pre_score)
    rec_score = recall_score(y_true, y_pred)
    print("recall: ",rec_score)
    f_score = f1_score(y_true, y_pred, average='weighted')
    print("f1_score: ",f_score)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_reprocess', type=bool, default=False)
    args, unknown = parser.parse_known_args()

    # 1 - Reading Data
    # 1.1. Read all data
    if args.data_reprocess:
        labeled_data_dir = os.path.join('data', 'machine_learning_challenge_labeled_data.csv')
        labeled_data = pd.read_csv(labeled_data_dir, delimiter=',')

        order_data_dir = os.path.join('data', 'machine_learning_challenge_order_data.csv')
        order_data = pd.read_csv(order_data_dir, delimiter=',')

        order_data.info()


        ## Data Preprocessing

        # 1.2. Feature Engineering

        # I preferred splitting the date information into year, month, and day and creating total days since
        # the first date of data collection.
        # The reason for splitting is seasonal information such as rainy or cold months may affect custormer behaviour.
        # The other reason is that the day of the month may also affect customer behavior due to salary date etc.
        print('*' * 30)
        order_data['order_date'] = pd.to_datetime(order_data['order_date'])
        base_date = pd.to_datetime('2015/03/01 00:00:00')
        order_data['date_from_begining'] = (order_data['order_date'] - base_date).dt.days
        order_data['order_year'] = order_data['order_date'].dt.year
        order_data['order_month'] = order_data['order_date'].dt.month
        order_data['order_day'] = order_data['order_date'].dt.day
        print(f"{order_data['date_from_begining'][0]}----{order_data['order_date'][0]} -->{order_data['order_date'].dtype}")
        print(f"year#{order_data['order_year'][0]}#month#{order_data['order_month'][0]}#day#{order_data['order_day'][0]}")
        print(f"hour{order_data['order_hour'][0]}")
        print(order_data.dtypes)

        # I add an "is_returning" column to use as a target label in the ordered_data data frame by mapping customer_id
        # from the labeled_data frame.
        # However, the initial mapping is not enough to carry information about "is_returning_customer."
        # Therefore, I updated the is_returning value to 1 if the next purchase exists in the last 6 months, otherwise to 0.
        # If other purchases do not exist or if the purchase is the last one, then keep information from labeled_data.
        order_data['is_returning'] = order_data['customer_id'].map(
            labeled_data.set_index('customer_id')['is_returning_customer'])
        for index, row in order_data.iterrows():
            if index == len(order_data.index) - 1: break
            next_row = order_data.iloc[index + 1]
            if row['customer_id'] == next_row['customer_id'] and is_lth_6month(next_row['order_date'], row['order_date']):
                row['is_returning'] = 1
            else:
                row['is_returning'] = 0

        order_data = order_data.drop(['order_date'], axis=1)
        print(order_data.head(50))

        # save data frame so that it does not repeat at each debug
        order_data.to_pickle('order_data_w_is_returning.pkl')
        print("processed data read is done")
    else:
        order_data = pd.read_pickle('order_data_w_is_returning.pkl')

    # 1.3. Data Transformation

    ##Categorical Features
    sparse_features = ['is_failed', 'restaurant_id', 'city_id', 'payment_id', 'platform_id', 'transmission_id']


    ##Integer Features
    dense_features = ['order_year', 'order_month', 'order_day', 'order_hour', 'customer_order_rank', 'voucher_amount',
                      'delivery_fee', 'amount_paid']

    target = ['is_returning']

    order_data[sparse_features] = order_data[sparse_features].fillna('-1', )
    order_data[dense_features] = order_data[dense_features].fillna(0, )

    # 1.3.1. Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        order_data[feat] = lbe.fit_transform(order_data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    order_data[dense_features] = mms.fit_transform(order_data[dense_features])
    print('+' * 30)
    print('info')
    print('+' * 30)
    print(order_data.info())

    print('-' * 30)
    print('describe')
    print('-' * 30)
    print(order_data.describe().T)

    # 2. Data Visualization
    # 2.1 Distribution of target
    f, ax = plt.subplots(1, 2, figsize=(18, 8))
    order_data['is_returning'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
    ax[0].set_title('distribution')
    ax[0].set_ylabel('')
    sns.countplot('is_returning', data=order_data, ax=ax[1])
    ax[1].set_title('is_returning')
    # Order_data is a balanced dataset #
    # The distribution of the target label, "is returning," is almost the same, respectively %48 and %52.
    # Then we can say our dataset is in balance.
    # There is no need for resampling or other imbalanced dataset handling techniques.

    # 2.2 Plotted the categorical variables on the basis of the graph of the column according to the dependent variable.
    fig, axarr = plt.subplots(2, 2, figsize=(20, 12))
    sns.countplot(x='order_hour', hue='is_returning', data=order_data, ax=axarr[0][0])
    sns.countplot(x='is_failed', hue='is_returning', data=order_data, ax=axarr[0][1])
    sns.countplot(x='platform_id', hue='is_returning', data=order_data, ax=axarr[1][0])
    sns.countplot(x='payment_id', hue='is_returning', data=order_data, ax=axarr[1][1])
    plt.show()

    # 2.3 Correlation Matrix
    f, ax = plt.subplots(figsize=[20, 15])
    sns.heatmap(order_data.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
    ax.set_title("Correlation Matrix", fontsize=20)
    plt.show()

    #I dropped these features due to Correalation Matrix result
    #order_data = order_data.drop(['order_year', 'date_from_begining'], axis=1)

    # 3 Train-Test Splitting
    random_state = 2022
    Y = order_data[target]
    X = order_data.drop(['is_returning', 'customer_id'], axis = 1)
    cols = X.columns
    print(X.head())
    print(f'X shape: {X.shape} , Y shape: {Y.shape}')
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=random_state)
    # 4 Define Model
    order_data = order_data.reset_index()
    models = []
    models.append(('LR', LogisticRegression(random_state=random_state)))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier(random_state=random_state)))
    models.append(('RF', RandomForestClassifier(random_state=random_state)))
    models.append(('XGB', GradientBoostingClassifier(random_state=random_state)))
    models.append(("LightGBM", LGBMClassifier(random_state=random_state)))
    models.append(("CatBoost", CatBoostClassifier(random_state=random_state, verbose=False)))
    models.append(('SVM', SVC(gamma='auto', random_state=random_state)))

    # evaluate each model in turn
    results = []
    names = []
    # Train Model
    for name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print_scores(name, y_test, y_pred)
    # Load best model

    # Test model
