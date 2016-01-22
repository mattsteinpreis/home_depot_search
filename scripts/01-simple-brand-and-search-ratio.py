import pandas as pd
import numpy as np
from sklearn import linear_model

def percentMatch(row):
    """
    This function finds the percent of search terms that are in the
    product title or description
    """

    count = 0.
    for word in row['search_term']:
        if word in row['product_title'] or word in row['product_description']:
            count += 1.
    return count / len(row['search_term'])

def transformer(df):

    # transform features
    df['product_title'] = df['product_title'].map(lambda x: x.lower())
    df['search_term'] = df['search_term'].map(lambda x: x.lower().split())
    df['product_description'] = df['product_description'].map(lambda x: x.lower())

    # define features
    df['brand_name'] = df['product_title'].map(lambda x: x.split()[0])
    df['has_brand'] = df.apply(lambda row: row['brand_name'] in row['search_term'],
                                                   axis=1)
    df['percent_match'] = df.apply(percentMatch, axis=1)
    return df

# load data
train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data/test.csv")
attributes = pd.read_csv("../data/attributes.csv")
descriptions = pd.read_csv("../data/product_descriptions.csv")

# join descriptions to data (skip attributes for now)
train_merged = pd.merge(train, descriptions, how='left', on='product_uid')
test_merged = pd.merge(test, descriptions, how='left', on='product_uid')

train_transformed = transformer(train_merged.copy())
test_transformed = transformer(test_merged.copy())

# get training data X, y
X_train = train_transformed[['has_brand','percent_match']]
y_train = train_transformed['relevance']

# get test data
X_test = test_transformed[['has_brand', 'percent_match']]

# create model
mod = linear_model.LinearRegression()
mod.fit(X_train, y_train)

# make predictions
predictions = mod.predict(X_test)
results = pd.DataFrame()
results['id'] = test['id']
results['relevance'] = predictions
results.to_csv('../submissions/submission01.csv', index=False)