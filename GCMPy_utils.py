import pandas as pd
import numpy as np

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


def datasummary(dataset, catslist, dimslist):
    '''
    Creates dataframe of mean values grouped by catgories
    
    Required parameters: 
    
    dataset = A dataframe to be analyzed, where each row is an observation
        Requires at least one category and one dimension
        
    catslist = List of categories to group by. Also accepts string.
    
    dimslist = List of dimensions to get values for. Also accepts dict
        with dimensions as keys.
    '''
    # Convert cat to list (e.g. if only one term is given)
    if type(catslist) != list:
        catslist = [catslist]
    # If the weights dictionary is given instead of the dimlist,
    ## take just the keys as a list
    if type(dimslist) == dict:
        dimslist=list(dimslist.keys())

    # group by categories: cats[0] will be used to group first, then cats[1]
    # i.e., if cats = ["vowel","type"], vowel1-type1, vowel1-type2, vowel2-type1, vowel2-type2...
    # get the mean of values for each dimension grouped by categories
    df = dataset.groupby(catslist,as_index=False)[dimslist].mean()
    return df


def continuum (data, start, end, dimslist, steps=7, stimdetails=False):
    '''
    Returns a continuum dataframe with interpolated values
    from a start to end value with a given number of steps
    * Users should be sure to specify any and all parameters they want
    start and end to match for. That is, say there are 2 repetitions of
    a stimulus. If it doesn't matter whether start and end are from the same
    repetition, you do not need to specify repetition number; one row will
    be chosen randomly. If it *does* matter that they're the same repetition,
    be sure to include repetition number in the dictionary.
    
    Required parameters:
    
    data = DataFrame to draw start and end stimuli from
    
    start = Dictionary indicating properties of the desired start
        with category types as keys, and their desired category as values.
        e.g., {"vowel":"i","speaker"="LB"}
    
    end = Dictionary indicating properties of the desired start
        with category types as keys, and their desired category as values
        
    dimslist = list containing the names of dimensions to be interpolated
    
    Optional parameters: 
    
    steps = integer indicating the total number of continuum steps. Defaults to 7.
    
    stimdetails = Boolean, defaults to False. Debugging/auditing tool to
        get details of the stimulus that aren't preserved in the returned
        dataframe (e.g., speaker ID)
    '''
    # create a copy of the entire df to subset according to conditions
    # match category to value from dictionary, subset
    # repeat subsetting until all conditions are satisfied
    st=data.copy()
    for i in range(0,len(start)):
        cat = list(start.keys())[i]
        val = list(start.values())[i]
        condition = st[cat]==val
        st = st.loc[condition]
    # reset index has to be outside of the loop to work with >2 conditions
    # sample(1) is there to just pick an observation if the conditions don't point
    ## a unique row in the dataframe
    st = st.sample(1).reset_index()
    
    en=data.copy()
    for i in range(0,len(end)):
        cat = list(end.keys())[i]
        val = list(end.values())[i]
        condition = en[cat]==val
        en = pd.DataFrame(en.loc[condition])
    en = en.sample(1).reset_index()
    
    # remember start & end values if needed
    if stimdetails == True:
        print("Start: " , st.iloc[0])
        print("End: " , en.iloc[0])

    norms = {}
    for dim in dimslist:                      # Calculate the difference between start and end for each dim
        norms[dim] = en[dim] - st[dim] 

    vals={}
    rowlist = []
    for i in range (0,steps):
        for dim in dimslist: 
            vals[dim] = st[dim] + (norms[dim] * i/(steps-1))    # the values for each dim = start val + diff by step
            row = pd.DataFrame(vals)
        rowlist.append(row)

    contdf = pd.concat(rowlist,ignore_index=True)

    return contdf



def checkaccuracy(choices,cats):
    '''
    Check rather the choices made by the model match the 'intended' label for each category.
    Returns a copy of the testset dataframe with column added indicating whether the choice for
    each category was correct (y) or incorrect (n)
    
    Required parameters:
    
    choices = output of choose() function: the test/stimulus dataframe with added columns showing what was 
        chosen for a category and with what probability.
    
    cats = a list of strings containing at least one item, indicating which
        category's probability was calculated for (e.g. ['vowel','gender']).
        Items should match the name of columns in the data frame
    '''
    if type(cats) != list:
        cats = [cats]
    
    acc = choices.copy()                     # Make a copy of choices to muck around with
    
    for cat in cats:                     # Iterate over your list of cats
        accname = cat + 'Acc'            # Get the right column names
        choicename = cat + 'Choice'
        
        # If choice is the same as intended, acc=y, else n
        acc[accname] = np.where(acc[cat]==acc[choicename], 'y', 'n')      
    
    return acc


def propcorr(acc,cat):
    '''
    Calculates the proportion of stimuli under each label which were categorized correctly
    Returns a dataframe with keys as labels and values as proportions between 0 and 1.
    
    Required parameters:
    
    acc = output of checkaccuracy() function: a copy of the testset dataframe with column
        added indicating whether the choice for each category was correct (y) or incorrect (n)
        
    cat = string ndicating which category accuracy should be assessed for. String should match
        column in acc.
    '''
    perc = dict(acc.groupby(cat)[cat+'Acc'].value_counts(normalize=True).drop(labels='n',level=1).reset_index(level=1,drop=True))
    pc=pd.DataFrame.from_dict(perc, orient='index').reset_index()
    pc.columns=[cat,'propcorr']
    return pc


def overallacc(acc,cat):
    '''
    Calculates accuracy for categorization overall, across all labels. Returns a 
    proportion between 0 and 1. 
    
    Required parameters: 
    
    acc = output of checkaccuracy() function: a copy of the testset dataframe with column
        added indicating whether the choice for each category was correct (y) or incorrect (n)
        
    cat = string ndicating which category accuracy should be assessed for. String should match
        column in acc.
    '''
    
    totalcorrect = acc[cat+'Acc'].value_counts(normalize=True)['y']
    return totalcorrect


def confusion(choices, cats):
    '''
    Returns a confusion matrix comparing intended category with categorization.
    
    Required parameters:
    
    choices = output of choose() function: the test/stimulus dataframe with added columns showing what was 
        chosen for a category and with what probability.
    
    cats = a list of strings containing at least one item, indicating which
        categories probability was calculated for (e.g. ['vowel','gender']).
        Items should match the name of columns in the data frame
    '''
    if type(cats) != list:
        cats = [cats]
    
    matrices={}
    for cat in cats:
        matrices[cat]=pd.crosstab(choices[cat],choices[cat+'Choice'], normalize='index').round(2).rename_axis(None)
    return matrices
