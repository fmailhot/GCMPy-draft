"""
Created on Wed January 20, 2021
@author: Emily Remirez (eremirez@berkeley.edu)
@version: 0.1
"""

import math
import random
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.optimize import minimize
import seaborn as sns
sns.set(style='ticks', context='paper')
colors=["#e3c934","#68c4bf","#c51000","#287271"]
sns.set_palette(colors)

def HzToBark(cloud,formants):
    '''
    Convert selected columns from Hz to Bark scale. Renames the formants as z.
    Returns the data frame with additional columns: the value of the formant
    converted from Hz to Bark
    
    Required parameters:
    
    cloud = dataframe of exemplars 
    
    formants = list of formants to be converted 
    '''
    # Make a copy of the cloud
    newcloud=cloud.copy()
    
    # For each formant listed, make a copy of the column prefixed with z
    for formant in formants:
        for ch in formant:
        if ch.isnumeric():
            num=ch
        formantchar = (formant.split(num)[0])
        name = str(formant).replace(formantchar,'z')
        # Convert each value from Hz to Bark
        newcloud[name] = 26.81/ (1+ 1960/newcloud[formant]) - 0.53
    # Return the dataframe with the changes
    return newcloud

def activation(testset,cloud,dims,c=0.01):
    '''
    Calculate activation for all exemplars stored in the cloud
    with respect to some stimulus, referred to as test. Returns
    a data frame with column 'a' added for each row.
    
    Required parameters:
    
    testset = a dataframe with one or more rows, each a stimulus to be categorized
        must have columns matching those given in the 'dims' dict. These columns
        should be dimensions of the stimulus (e.g., formants)
        
    cloud = A dataframe of stored exemplars which every stimulus is compared to. 
        Each row is an exemplar, which, like testset should have columns matching
        those in the dims dict
    
    dims = a dictionary with dimensions as keys and weights, w, as values. 
    
    c = an integer representing exemplar sensitivity. Defaults to .01. 
        
    '''
    # Get stuff ready                                                   
    dims.update((x, (y/sum(dims.values()))) for x, y in dims.items())   # Normalize weights to sum to 1
    
    # If the testset happens to have N in it, remove it before joining dfs 
    test=testset.copy()
    if 'N' in test.columns:
        test = test.drop(columns='N', axis=1,inplace=True)
    
    exemplars=cloud.copy()

    # Merge test and exemplars
    bigdf = pd.merge(
        test.assign(key=1),         # Add column named 'key' with all values == 1
        exemplars.assign(key=1),    # Add column named 'key' with all values == 1
        on='key',                   # Match on 'key' to get cross join (cartesian product)
        suffixes=['_t', '_ex']
    ).drop('key', axis=1)           # Drop 'key' column
    
    
    dimensions=list(dims.keys())                # Get dimensions from dictionary
    weights=list(dims.values())                 # Get weights from dictionary
    tcols = [f'{d}_t' for d in dimensions]      # Get names of all test columns
    excols = [f'{d}_ex' for d in dimensions]    # Get names of all exemplar columns
    
    
    # Multiply each dimension by weights
    i = bigdf.loc[:, tcols].values.astype(float)     # Get all the test columns
    i *= weights                                     # Multiply test columns by weight
    j = bigdf.loc[:, excols].values.astype(float)    # Get all the exemplar columns
    j *= weights                                     # Multiply exemplar columns by weights
    
    # Get Euclidean distance
    bigdf['dist'] = np.sqrt(np.sum((i-j)**2, axis=1))
    
    # get activation: exponent of negative distance * sensitivity c, multiplied by N_j
    bigdf['a'] = np.exp(-bigdf.dist*c)*bigdf.N
    
    return bigdf
    
    
def exclude(cloud, test, exclude_self=True, alsoexclude=None): 
    '''
    Removes specific rows from the cloud of exemplars, to be used
    prior to calculating activation. Prevents activation from being
    overpowered by stimuli that are too similar to particular exemplars.
    E.g., prevents comparison of a stimulus to itself, or to exemplars from same speaker
    Returns dataframe containing a subset of rows from the cloud.
    
    Required parameters:
    
    cloud = A dataframe of stored exemplars which every stimulus is compared to. 
        Each row is an exemplar
    
    test = single row dataframe containing the stimulus to be categorized
    
    exclude_self = boolean. If True, stimulus will be removed from exemplar cloud
        so that it isn't compared to itself. Defaults to True 
    
    Optional parameters:
    
    alsoexclude = a list of strings matching columns in the cloud (categories) to exclude 
        if value is the same as that of the test. (E.g., to exclude all exemplars from
        the speaker to simulate categorization of novel speaker)
    
    
    '''
    # cloud = exemplars, dataframe
    # test = exemplar to be categorized
    # exclude_self = true or false, should the exemplar not be compared to itself? default true
    # exclude = a list of columns in cloud to also exclude
    
    # Make a copy of the cloud and call it exemplars. 
    #    This is what we'll return at the end
    exemplars = cloud.copy()
    
    # Remove the stimulus from the cloud
    if exclude_self == True:
        exemplars=cloud[~cloud.isin(test)].dropna()  
    
    if alsoexclude != None:
        for feature in alsoexclude:
            featval=test[feature].iloc[0]
            exclude_exemps=exemplars[ exemplars[feature] == featval ].index
            exemplars.drop(exclude_exemps, inplace=True)
        
    return exemplars

def reset_N(exemplars, N=1):      # Add or override N, default to 1
    '''
    Adds an N (base activation) column to the exemplar cloud so
    that activation with respect to the stimulus can be calculated
    Default value is 1, i.e., equal activation for each exemplar.
    Returns the exemplar data frame with added or reset column
    
    Required parameters:
    
    exemplars = data frame of exemplars to which the stimulus is being
        compared
        
    N = integer indicating the base activation value to be added to
        each exemplar (row) in the dataframe. Defaults to 1
    '''
    exemplars['N'] = N
    return exemplars

def probs(bigdf,cats):
    
    '''
    Calculates the probability that the stimulus will be categorized with a
    particular label for a given category (e.g., vowel labels 'i', 'a', 'u' for
    the category 'vowel'). Probability is calculated by summing the activation
    across all exemplars sharing a label, and dividing that by the total amount
    of activation in the system for the category. Returns a dictionary of dictionaries.
    Each key is a category; values are dictionaries where keys are labels and values
    represent probability of the stimulus being categorized into that label.
    
    Required parameters: 
    
    bigdf = a dataframe produced by activation(), which contains a row for each
        exemplar with the additional column 'a' representing the amount of 
        activation for that exemplar with respect to the stimulus
    
    cats = a list of strings containing at least one item, indicating which
        categories probability should be calculated for (e.g. ['vowel','gender']).
        Items should match the name of columns in the data frame
    '''
    prs = {}
    
    # Loop over every category in the list of categories
    for cat in cats: 
        # make that category match the exemplar category in name
        label = cat+'_ex'
        # Sum up activation for every label within that category
        cat_a = bigdf.groupby(label).a.sum()
        # Divide the activation for each label by the total activation for that category
        pr = cat_a/sum(cat_a)
        # rename a for activation to probability
        pr = pr.rename_axis(cat).reset_index().rename(columns={"a":"probability"})
        # add this to the dictionary 
        prs[cat]=pr
    return prs
    

    def choose(pr,test,cats,runnerup=False):
    '''
    Chooses a label for each category which the stimulus will be categorized as.
    Returns the test/stimulus dataframe with added columns showing what was 
    chosen for a category and with what probability. Optionally will give the
    second most probable label as well. 
    
    Required parameters:
    pr = dictionary of probabilities, given from probs(). Each key should represent
        a category (e.g. 'vowel'), with values as dictionaries with keys for category
        labels (e.g. 'i','a','u')
        
    test = single line data frame representing the test/stimulus being categorized
        
    cats = a list of strings containing at least one item, indicating which
        categories probability should be calculated for (e.g. ['vowel','gender']).
        Items should match the name of columns in the data frame
            
    Optional parameters:
    runnerup = boolean; when true the label with the second highest probability
        will also be included in the dataframe. Defaults to False. 
    
    '''
    
    
    
    newtest = test.copy()

    for cat in cats:
        choicename = cat + 'Choice'
        choiceprobname = cat + 'Prob'
        
        best2 = pr[cat]['probability'].nlargest(n=2).reset_index(drop=True)        # Get the two highest probs for each cat type
        
        choiceprob = best2[0]                                                      # Match the prob to the category
        choice = pr[cat].loc[pr[cat]['probability']==choiceprob,cat].iloc[0]
        
        newtest[choicename] = choice
        newtest[choiceprobname] = choiceprob
        
        if runnerup == True: 
            choice2name = cat + 'Choice2'
            choice2probname = cat +'Choice2Prob'
            choice2prob = best2[1]                                                      
            choice2 = pr[cat].loc[pr[cat]['probability']==choice2prob,cat].iloc[0]
            newtest[choice2name] = choice2
            newtest[choice2probname] = choice2prob
            
    return newtest

def gettestset(cloud,balcat,n):     #Gets n number of rows per cat in given cattype
    '''
    Gets a random test set of stimuli to be categorized balanced across a particular
    category, e.g., 5 instances of each label 'i','a', 'u' for category 'vowel'. 
    Returns a data frame of stimuli.
    
    Required parameters:
    
    cloud = dataframe of exemplars
        
    balcat = category stimuli should be balanced across 
        
    n = number of stimuli per category label to be included
    '''
    testlist=[]
    for cat in list(cloud[balcat].unique()):
        samp = cloud[cloud[balcat]==cat].sample(n)
        testlist.append(samp)
    test=pd.concat(testlist)
    return test

def categorize(testset,cloud,cats,dims,c,exclude_self=True,alsoexclude=None, N=1, runnerup=False):
    '''
    Categorizes a stimulus based on functions defined in library. 
    1. Exclude any desired stimuli
    2. Add N value
    3. Calculate activation
    4. Calculate probabilities
    5. Choose labels for each category
    Returns the output of choose(): test/stimulus dataframe with added columns showing what was 
    chosen for a category and with what probability
    
    Required parameters:
    
    testset = a dataframe with one row, a stimulus to be categorized
        must have columns matching those given in the 'dims' dict. These columns
        should be dimensions of the stimulus (e.g., formants)
        
    cloud = A dataframe of stored exemplars which every stimulus is compared to. 
        Each row is an exemplar, which, like testset should have columns matching
        those in the dims dict
        
    cats = a list of strings containing at least one item, indicating which
        categories probability should be calculated for (e.g. ['vowel','gender']).
        Items should match the name of columns in the data frame
    
    dims = a dictionary with dimensions as keys and weights, w, as values. 
    
    c = an integer representing exemplar sensitivity. Defaults to .01. 
    
    exclude_self = boolean. If True, stimulus will be removed from exemplar cloud
        so that it isn't compared to itself. Defaults to True 
        
    Optional parameters:
    alsoexclude = a list of strings matching columns in the cloud (categories) to exclude 
        if value is the same as that of the test. (E.g., to exclude all exemplars from
        the speaker to simulate categorization of novel speaker)
    
    N = integer indicating the base activation value to be added to
        each exemplar (row) in the dataframe. Defaults to 1
        
    runnerup = boolean; when true the label with the second highest probability
        will also be included in the dataframe. Defaults to False.

    '''
    
    test=testset
    exemplars=exclude(cloud,test,exclude_self=exclude_self,alsoexclude=alsoexclude)
    reset_N(exemplars, N=N)
    bigdf=activation(test,exemplars,dims=dims,c=c)
    pr=probs(bigdf,cats)
    choices=choose(pr,test,cats,runnerup=runnerup)
    return choices 
    
def getactiv(activation,x,y,cat):
    
    """ 
    Creates a simplified data frame showing the activation for each exemplar 
    with respect to the stimulus. Primarily for use with the activplot()
    function. 
    
    Required parameters:
    
    activation = DataFrame resulting from the activation() function, containing
        one row per stored exemplar, with their activation 'a' as a column
        
    x = String. Dimension to be plotted as x axis in scatterplot (e.g., F2). Matches
        the name of a column in the activation DataFrame.
    
    y = String. Dimension to be plotted as y axis in scatterplot (e.g., F1). Matches
        the name of a column in the activation DataFrame.
    
    cat = String. Category used to color code exemplars in scatter plot. Matches the name
        of a column in the activation DataFrame.
    """
    xname = x + "_ex"
    yname = y + "_ex"
    catname = cat + "_ex"
    
    acts = activation['a']
    xs = activation[xname]
    ys = activation[yname]
    cats = activation[catname]
    
    activ = pd.concat([acts,xs,ys,cats], axis=1)
    activ.rename(columns={xname:x, yname:y, catname:cat}, inplace=True)
    
    return activ

def activplot(a,x,y,cat, test):
    """
    Plots each exemplar in x,y space according to specified dimensions. Labels within
    the category are grouped by color. The stimulus or test exemplar is plotted in dark
    blue on top of exemplars. Note: axes are inverted, assuming F1/F2 space
    
    Required parameters:
    
    a = DataFrame produced by getactiv() function. Contains a row for each exemplar
        
    """
    
    pl = sns.scatterplot(data=a,x=x,y=y,hue=cat,size='a',size_norm=(0,a.a.max()),
                     alpha=0.5,sizes=(5,100),legend=False)
    pl = sns.scatterplot(data=test, x=x,y=y,alpha=.5,color='darkblue',marker="X", s= 50, legend=False)
    
    pl.invert_xaxis()
    pl.invert_yaxis()

    return pl

def multicat(testset,cloud,cats,dims,c,exclude_self=True,alsoexclude=None, N=1, runnerup=False):
    '''
    Categorizes a dataframe of multiple stimuli based on functions defined in library. 
    1. Exclude any desired stimuli
    2. Add N value
    3. Calculate activation
    4. Calculate probabilities
    5. Choose labels for each category
    Returns the output of choose(): test/stimulus dataframe with added columns showing what was 
    chosen for a category and with what probability
    
    Required parameters:
    
    testset = a dataframe with one or more rows, each a stimulus to be categorized
        must have columns matching those given in the 'dims' dict. These columns
        should be dimensions of the stimulus (e.g., formants)
        
    cloud = A dataframe of stored exemplars which every stimulus is compared to. 
        Each row is an exemplar, which, like testset should have columns matching
        those in the dims dict
        
    cats = a list of strings containing at least one item, indicating which
        categories probability should be calculated for (e.g. ['vowel','gender']).
        Items should match the name of columns in the data frame
        
    dims = a dictionary with dimensions as keys and weights, w, as values. 
    
    c = an integer representing exemplar sensitivity. Defaults to .01. 
    
    exclude_self = boolean. If True, stimulus will be removed from exemplar cloud
        so that it isn't compared to itself. Defaults to True 
        
    Optional parameters:
    alsoexclude = a list of strings matching columns in the cloud (categories) to exclude 
        if value is the same as that of the test. (E.g., to exclude all exemplars from
        the speaker to simulate categorization of novel speaker)
    
    N = integer indicating the base activation value to be added to
        each exemplar (row) in the dataframe. Defaults to 1
        
    runnerup = boolean; when true the label with the second highest probability
        will also be included in the dataframe. Defaults to False.
    '''
    
    choicelist=[]
    for ix in list(testset.index.values):
        test = testset.loc[[ix,]]
        exemplars=exclude(cloud,test,exclude_self=exclude_self,alsoexclude=alsoexclude)
        reset_N(exemplars,N=N)
        bigdf=activation(test,exemplars,dims = dims,c=c)
        pr=probs(bigdf,cats)
        choices = choose(pr,test,cats,runnerup=runnerup)
        choicelist.append(choices)
    choices=pd.concat(choicelist, ignore_index=True)
    return choices

def checkaccuracy(choices,cats):
    '''
    Check rather the choices made my the model match the 'intended' label for each category.
    Returns a copy of the testset dataframe with column added indicating whether the choice for
    each category was correct (y) or incorrect (n)
    
    Required parameters:
    
    choices = output of choose() function: the test/stimulus dataframe with added columns showing what was 
        chosen for a category and with what probability.
    
    cats = a list of strings containing at least one item, indicating which
        categories probability was calculated for (e.g. ['vowel','gender']).
        Items should match the name of columns in the data frame
    '''
    acc = choices.copy()                     # Make a copy of choices to muck around with
    
    for cat in cats:                     # Iterate over your list of cats
        accname = cat + 'Acc'            # Get the right column names
        choicename = cat + 'Choice'
        
        # If choice is the same as intended, acc =y, else n
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

def accplot(acc,cat):
    '''
    Plots a bar graph showing the proportion of trials which were categorized
    veridically, that is, accuracy of categorization.
    
    Required parameters:
    
    acc = output of checkaccuracy() function: a copy of the testset dataframe with column
        added indicating whether the choice for each category was correct (y) or incorrect (n)
        
    cat = string ndicating which category accuracy should be assessed for. String should match
        column in acc.
    
    '''
    perc = dict(acc.groupby(cat)[cat+'Acc'].value_counts(normalize=True).drop(labels='n',level=1).reset_index(level=1,drop=True))
    pc=pd.DataFrame.from_dict(perc, orient='index').reset_index()
    pc.columns=[cat,'propcorr']
    
    obs=str(len(acc))
    pl = sns.barplot(x=cat,y='propcorr',data=pc,palette=colors)
    plt.ylim(0,1.01)
    pl.set(ylabel='Proportion accurate of '+obs+' trials')
    pl.set_xticklabels(
    pl.get_xticklabels(), 
    rotation=45, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-large')
    plt.show()
    return pl

def multiaccplot(choices,cats):
    '''
    Plots accuracy of multiple categories
    
    Required parameters:
    
    choices = output of choose() function: the test/stimulus dataframe with added columns showing what was 
        chosen for a category and with what probability.
    
    cats = a list of strings containing at least one item, indicating which
        categories probability was calculated for (e.g. ['vowel','gender']).
        Items should match the name of columns in the data frame
    '''
    accuracy = checkaccuracy(choices,cats)
    for cat in cats:
        proportion = propcorr(accuracy,cat)
        accplot(proportion,cat,accuracy)
        print(proportion)
        
def confusion(choices,cats):
    '''
    Returns a confusion matrix comparing intended category with categorization.
    
    Required parameters:
    
    choices = output of choose() function: the test/stimulus dataframe with added columns showing what was 
        chosen for a category and with what probability.
    
    cats = a list of strings containing at least one item, indicating which
        categories probability was calculated for (e.g. ['vowel','gender']).
        Items should match the name of columns in the data frame
    '''
    matrices={}
    for cat in cats:
        matrices[cat]=pd.crosstab(choices[cat],choices[cat+'Choice'],normalize='index').round(2).rename_axis(None)
    return matrices

