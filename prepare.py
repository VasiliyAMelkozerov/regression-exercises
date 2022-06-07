import pandas as pd
import numpy as np
import os
from env import host, user, password
from sklearn.model_selection import train_test_split


def drop_propertylandusetypeid(df):
    #after we have used propertylandusetypeid to find our single family residency we can drop it as it serves no further purpose
    df = df.drop('propertylandusetypeid', axis=1)
    return df

def drop_nulls(df):
    #dropping null data which is around 1% of all data to start giving us cleaner data to work with
    df = df.dropna()
    return df

def drop_nobed_nobath(df):
    #dropping listings of 0 bed and 0 bath because they do not help us find homes for our customers
    #drawing a limit at 70 sqft because anythin less then that would be unliviable 
    df = df[(df.bedroomcnt != 0) & (df.bathroomcnt != 0) & (df.calculatedfinishedsquarefeet >= 70)]
    return df

def squish_outliers(df):
    #take out properties that are beyond a reasonable price
    #these houses err towards the 1% which is not our target demographic
    df = df[df.bathroomcnt <= 6]
    #drawing a limit at 6 because it becomes unreasonable for one family ot afford
    df = df[df.bedroomcnt <= 6]
    #same with the amount of bathrooms, not applicable to our customer base
    df = df[df.taxvaluedollarcnt < 2_000_000]
    return df

def wrangled_zillow(df):
    #we bring in all of the edits in one nice small package
    # df = drop_propertylandusetypeid(df)
    df = drop_nulls(df)
    df = drop_nobed_nobath(df)
    df = squish_outliers(df)
    df.to_csv("zillow.csv", index=False)
    #locally we now save adjusted csv file for this exploration
    return df

def traintestsplit(df):
    #this function gives us train validate test variabels to model with
    train_validate, test = train_test_split(df, test_size=.2, random_state=123, stratify=df.fips)
    #80% of value
    #I choose to stratify here to get an even mix of counties to not miss trends and clearly see if the taxvaluedollarcount is in fact evenly set on  counties
    #TL:DR  keeping counties even in sample
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123, stratify=train_validate.fips)
    # 30% of 80% for validate
    return train, validate, test