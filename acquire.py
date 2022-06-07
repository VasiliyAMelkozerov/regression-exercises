import pandas as pd
import numpy as np
import os
from env import host, user, password

def get_connection(db, user=user, host=host, password=password):
    '''
    establishes connection to SQL server
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def zillow_data():
    #this functions runs saves our import as csv to save time
    filename = "zillow.csv"
    #filename kept simple
    if os.path.isfile(filename):
        #if there is a local copy run it or use the following function to import
        return pd.read_csv(filename)
    else:
        return get_zillow_data()

def get_zillow_data():
    #bring in a short list of features
    sql_query = """SELECT * FROM properties_2017
    join predictions_2017
    USING (parcelid)
    WHERE
    propertylandusetypeid = 261
        OR propertylandusetypeid = 279;"""
    #the land use type represents if it is registered as a single family home
    df = pd.read_sql(sql_query, get_connection('zillow'))
    return df