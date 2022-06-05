import pandas as pd
import numpy as np
import os
from env import host, user, password
from sklearn.model_selection import train_test_split

def get_connection(db, user=user, host=host, password=password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def zillow_data():
    #we save first before editing so that we still have the original on hand
    filename = "zillow.csv"
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        return get_zillow_data()

def get_zillow_data():
    #this sql code answers Q1 E2 from regressions lesson
    sql_query = """SELECT
    bedroomcnt,
    bathroomcnt,
    calculatedfinishedsquarefeet,
    taxvaluedollarcnt,
    yearbuilt,
    taxamount,
    fips,
    propertylandusetypeid
FROM
    properties_2017
        JOIN
    propertylandusetype USING (propertylandusetypeid)
WHERE
    propertylandusetypeid = 261
        OR propertylandusetypeid = 279;"""
    df = pd.read_sql(sql_query, get_connection('zillow'))
    return df

def drop_propertylandusetypeid(df):
    df = df.drop('propertylandusetypeid', axis=1)
    return df

def drop_nulls(df):
    df = df.dropna()
    return df

def drop_nobed_nobath(df):
    df = df[(df.bedroomcnt != 0) & (df.bathroomcnt != 0) & (df.calculatedfinishedsquarefeet >= 70)]
    return df

def squish_outliers(df):
    #take out properties that are beyond a reasonable price
    df = df[df.bathroomcnt <= 6]
    
    df = df[df.bedroomcnt <= 6]

    df = df[df.taxvaluedollarcnt < 2_000_000]

    return df

def wrangled_zillow(df):
    # df = drop_propertylandusetypeid(df)

    df = drop_nulls(df)

    df = drop_nobed_nobath(df)

    df = squish_outliers(df)

    df.to_csv("zillow.csv", index=False)

    return df

def traintestsplit(df):
    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.fips)
    #stratify is used for categorical data
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123, stratify=train_validate.fips)
    #I can stratify on county if i want an even distribution of it in my train validate test models
    return train, validate, test
    #go back and stratify properly on fips