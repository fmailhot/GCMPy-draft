import pandas as pd


def HzToBark(df_cloud, formant_cols):
    '''
    Convert selected columns from Hz to Bark scale. Renames the formants as z.
    Returns the data frame with additional columns: the value of the formant
    converted from Hz to Bark
    
    Required parameters:
    
    df_cloud = pd.DataFrame of exemplars
    
    formant_cols = list of formants to be converted 
    '''
    # For each formant listed, make a copy of the column prefixed with z
    for formant in formant_cols:
        # convert column name to Bark-name
        bark = formant.replace('F', 'z')
        # Convert each value from Hz to Bark
        df_cloud[bark] = (26.81 / (1 + 1960/df_cloud[formant])) - 0.53
    # Return the dataframe with the changes
    return df_cloud
