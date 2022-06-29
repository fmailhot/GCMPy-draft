"""
Created on Wed January 20, 2021
Last updated May 17, 2022
@author: Emily Remirez (eremirez@berkeley.edu)
@version: 0.1
"""
import numpy as np
import pandas as pd
from GCMPy_utils import HzToBark, checkaccuracy, overallacc, get_testset
from GCMPy_viz import activplot, accplot, cpplot


def activation(testset,cloud,dims,c=25):
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
    
    c = an integer representing exemplar sensitivity. Defaults to 25. 
        
    '''
    # Get stuff ready
    # (Normalize weights to sum to 1)
    dims.update((x, (y/sum(dims.values()))) for x, y in dims.items())
    
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
    extemp = exemplars.copy()
    extemp['N'] = N
    return extemp


def bias_N(exemplars, cat, catbias):
    '''
    Adds or overwrites an N (base activation) colummn to the exemplar 
    cloud so that activation with respect to the stimulus can be 
    calculated. Unlike reset_N, which assigns the same N value to all exemplars,
    bias_N will set N values according to values in a dictionary. That is, within a 
    category type, each category will have the N value specified in the dictionary
    
    Required parameters:
    
    exemplars = dataframe of exemplars to which the stimulus is being compared
    
    cat = a string designating the category type which is being primed
    
    catbias = dictionary with categories (e.g. vowels) as keys and N value for the  
        category as values
    '''
 
    extemp = exemplars.copy()
    extemp['N'] = extemp[cat].map(catbias)
    return extemp


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
    
    if type(cats) != list:
        cats = [cats]
    
    # Loop over every category in the list of categories
    for cat in cats: 
        if cat in bigdf:
            label = cat
        else: 
            # make category match the exemplar category in name if i and j share column names
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
    

def choose(probsdict,test,cats,runnerup=False,fc=None):
    '''
    Chooses a label for each category which the stimulus will be categorized as.
    Returns the test/stimulus dataframe with added columns showing what was 
    chosen for a category and with what probability. Optionally will give the
    second most probable label as well. 
    
    Required parameters:
    pr = dictionary of probabilities, given from probs(). Each key should represent
        a category (e.g. 'vowel'), with values as dataframe. Dataframe should
        have a probability for each category label
        
    test = single line data frame representing the test/stimulus being categorized
    
    cats = list of categories to be considered (e.g., ["vowel"])
            
    Optional parameters:
    runnerup = boolean; when true the label with the second highest probability
        will also be included in the dataframe. Defaults to False. 
        
    fc = Dict where keys are category names in the dataframe and values are a list of category labels.
        Used to simulate a forced choice experiment in which the perceiver has a limited number
        of alternatives. For example, if fc = {'vowel':['i','a']}, the choice will be the alternative 
        with higher probability, regardless of whether other vowels not listed have higher probabilities. 
        There can be any number of alternatives in the list.
    
    '''
    newtest = test.copy()      # make a copy of the test set to add to
    pr=probsdict.copy()        # make a copy of the probs dict to subset if forced choice is set       
    
    if fc!=None: 
        fccats = fc.keys()
        for fccat in fccats:
            options = fc[fccat]
            scope = probsdict[fccat]
            toconsider = scope.loc[scope[fccat].isin(options)]
        pr[fccat] = toconsider

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


def categorize(testset, cloud, cats, dims, c,
               exclude_self=True, alsoexclude=None, N=1, runnerup=False, fc=None):
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
    choices=choose(pr,test,cats,runnerup=runnerup,fc=fc)
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
    
    if cat in activation:
        catname = cat + "_ex"
    else: 
        catname = cat
     
    acts = activation['a']
    xs = activation[xname]
    ys = activation[yname]
    cats = activation[catname]
    
    activ = pd.concat([acts,xs,ys,cats], axis=1)
    activ.rename(columns={xname:x, yname:y, catname:cat}, inplace=True)
    
    return activ


def multicat(testset,cloud,cats,dims,c=25,N=1,biascat=None,catbias=None,rescat=None, ncyc= None, exclude_self=True,alsoexclude=None,runnerup=False,fc=None):
    '''
    Categorizes a dataframe of 1 or more stimuli based on functions defined in library
    
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
    
    c = an integer representing exemplar sensitivity. Defaults to 25. 
    
    exclude_self = boolean. If True, stimulus will be removed from exemplar cloud
        so that it isn't compared to itself. Defaults to True 
        
    Optional parameters:
    
    biascat = A string indicating the category type to be biased or primed on (e.g. 'vowel', 'speaker')
    
    catbias = Dict where keys are categories of biascat and values are
        ints that indicate relative N values. (e.g., {'i':5,'a':1} would make every 'i' exemplar 
        contribute 5 times as much activation as each 'a)
    
    rescat = Category to resonate on. If given, 
    
    ncyc = Int indicating how many cycles of resonance
    
    alsoexclude = a list of strings matching columns in the cloud (categories) to exclude 
        if value is the same as that of the test. (E.g., to exclude all exemplars from
        the speaker to simulate categorization of novel speaker)
    
    N = integer indicating the base activation value to be added to
        each exemplar (row) in the dataframe. Defaults to 1
        
    runnerup = boolean; when true the label with the second highest probability
        will also be included in the dataframe. Defaults to False.
        
    fc = Dict where keys are category names in the dataframe and values are a list of category labels.
        Used to simulate a forced choice experiment in which the perceiver has a limited number
        of alternatives. For example, if fc = {'vowel':['i','a']}, the choice will be the alternative 
        with higher probability, regardless of whether other vowels not listed have higher probabilities. 
        There can be any number of alternatives in the list. 
    '''
    choicelist=[]
    for ix in list(testset.index.values):
        test = testset.loc[[ix,]]
        
        # exclusions
        exemplars=exclude(cloud,test,exclude_self=exclude_self,alsoexclude=alsoexclude)
        
        #add N 
        if catbias != None: 
            exemplars = bias_N(exemplars,biascat,catbias)
        else: exemplars = reset_N(exemplars, N=N)
        
        # calculate probabilities
        bigdf=activation(test,exemplars,dims = dims,c=c)
        pr=probs(bigdf,cats)
        
        # resonate if applicable -- recalculate probs based on a resonance term
        if rescat != None:
            for n in range(0,ncyc):
                edict = pr[rescat].set_index(rescat).to_dict()['probability']
                # resonance term = probability of category divided by number of cycles
                    ## so that effect decays over time
                exemplars['resterm'] = exemplars[rescat].map(edict) / (n+1)
                # Add resterm to N value; N only ever goes up
                exemplars['N'] = exemplars['N'] + exemplars['resterm']
                bigdf=activation(test,exemplars,dims = dims,c=c)
                pr=probs(bigdf,cats)
        
        # Luce's choice rule
        choices = choose(pr,test,cats,runnerup=runnerup,fc=fc)
        
        choicelist.append(choices)
    choices=pd.concat(choicelist, ignore_index=True)
    return choices


def errorfunc(x, testset, cloud, dimslist, cat):
    ''' 
    Returns a proportion representing the total amount of error for a single category that
    the categorizer makes given a certain set of c and w values. This is intended to
    be used with an optimization function so that the total amount of error can be 
    minimized; that is, the accuracy can be maximized. 
    Note that z0 is automatically set to 1.
    
    Required parameters: 
    
    x = a vector of values to be used by multicat. x[0] should be c, x[1], x[2], x[3]
        should correspond to dimslist[1], dimslist[2], dimslist[3]
        
    testset = a dataframe with one or more rows, each a stimulus to be categorized
        must have columns matching those given in the dims list. These columns
        should be dimensions of the stimulus (e.g., formants)
        
    cloud = A dataframe of stored exemplars which every stimulus is compared to. 
        Each row is an exemplar, which, like testset should have columns matching
        those in the dims list
    
    dimslist = a list of dimensions (e.g., formants), for which weights w should be given,
        and along which exemplars should be compared.
    
    cat = the category, 
    '''
    #x = [c,z1,z2,z3]
    catlist=[cat]
    c=x[0]
    dimsdict={dimslist[0]:1,dimslist[1]:x[1],dimslist[2]:x[2],dimslist[3]:x[3]}
    choices=multicat(cloud,testset,catlist,dims=dimsdict,c=c)
    accuracy=checkaccuracy(choices,catlist)
    err = accuracy[cat+'Acc'].value_counts(normalize=True)['n']
    return err


if __name__ == '__main__':
    print('load pb52 data, covert to Bark & sample testset')
    pb52 = pd.read_csv('data/pb52.csv')
    pbbark = HzToBark(pb52, ['F0','F1','F2','F3'])
    print(pbbark.sample(5).head())
    # Get a balanced test set, 50 obs per vowel
    test = get_testset(pbbark, 'vowel', 50)

    print('set GCM params, categorize testset & check accuracy')
    # set c, the sensitivity of exemplar cloud
    cval=5
    # set dimensions m as keys, 
    # set weight of each dimension w_m as values
    dimsvals = {'z0':1, 'z1': 2.953, 'z2':.924, 'z3':3.420}
    # set categories to be considered as items in a list
    catslist=['vowel','type']
    # categorize testset
    choices = multicat(test, pbbark, catslist, dimsvals, cval, exclude_self=True,
                       alsoexclude=None, N=1, runnerup=False)
    # check accuracy
    acc = checkaccuracy(choices,catslist)
    print("overall accuracy: " + str(overallacc(acc,'vowel')))
