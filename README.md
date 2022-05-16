# delivery_hero_casestudy S.Tolga Yildiran

## How to test the python file
requriments.txt is added to the repository

There are two options to execute python file:

1: (Default) Do not repeat Data Processing part to save time

- python main.py

2: Repeat Data Processing

- python main.py --data_reprocess True

## Feature Engineering

I preferred splitting the date information into year, month, and day and creating total days since the first date of data collection.
The reason for splitting is seasonal information such as rainy or cold months may affect customer behaviour.
The other reason is that the day of the month may also affect customer behavior due to salary date etc.

"is_returning" column is added to use as a target label in the ordered_data data frame by mapping customer_id from the labeled_data frame.
However, the initial mapping is not enough to carry information about "is_returning_customer."
Therefore, I updated the is_returning value to 1 if the next purchase exists in the last 6 months, otherwise to 0. 
If other purchases do not exist or if the purchase is the last one, then keep information from labeled_data.

Features are evaluated in two categories, and a scaling method is applied towards these categories.
- Sparse Feature: 'is_failed', 'restaurant_id', 'city_id', 'payment_id', 'platform_id', 'transmission_id'
- Dense Feature: 'order_year', 'order_month', 'order_day', 'order_hour', 'customer_order_rank', 'voucher_amount', 'delivery_fee', 'amount_paid'

## Data Visualization
Order_data is a balanced dataset 

The distribution of the target label, "is returning," is almost the same, respectively %48 and %52.
Then we can say our dataset is in balance.

There is no need for resampling or other imbalanced dataset handling techniques.

There is no need for dropping features due to correlation matrix result.
## Models

### LogisticRegression
- accuracy:  0.6098080345792016
- precision:  0.6066961078476947
- recall:  0.7038843876788057
- f1_score:  0.60585361194275


### KNeighborsClassifier
- accuracy:  0.61849097381134
- precision:  0.6254216783379476
- recall:  0.659022100192442
- f1_score:  0.6177969385657235


###DecisionTreeClassifier
- accuracy:  0.651118738876176
- precision:  0.662633868196937
- recall:  0.6666462375740044
- f1_score:  0.6510745175378062


###RandomForestClassifier
- accuracy:  0.7370645817442156
- precision:  0.7585634932880728
- recall:  0.723128592966672
- f1_score:  0.7371469874440894


###GradientBoostingClassifier
- accuracy:  0.7273201118738876
- precision:  0.7424874948286886
- recall:  0.7259600652096638
- f1_score:  0.7274007965766665


###LGBMClassifier
- accuracy:  0.7298627002288329
- precision:  0.745056240360139
- recall:  0.7282889817731635
- f1_score:  0.7299431113257417


###CatBoostClassifier
- accuracy:  **0.7376366641240784**
- precision:  **0.7554243130893238**
- recall:  **0.7306178983366632**
- f1_score: **0.7377266412426771**


###SVC
- Not enough time to train the SVM model.




##Outputs

The best model with baseline parameters is the Catboost algorithm. 

It is a gradient boosting framework which, among other features, attempts to solve for categorical features using a permutation-driven alternative compared to the classical algorithm.
(2019-01-20) (Prokhorenkova, Liudmila; Gusev, Gleb; Vorobev, Aleksandr; Dorogush, Anna Veronika; Gulin, Andrey)."CatBoost: unbiased boosting with categorical features"

In all measurement methods (accuracy, precision, recall, and f1 score), it produces the best results.

I was expecting the best results from Support Vector Classification, but time is not enough to see the result of it.

The success of gradient boosting is not surprising; it is one of the most popular machine learning techniques in recent years, dominating many Kaggle competitions with heterogeneous tabular data.
    
