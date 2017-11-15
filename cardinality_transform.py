import pandas as pd

'''
    This code is for applying Daniele Micci-Barreca transformation to data according to the following paper:

   https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf 

    This application is for binary classification only.

'''
def MICC_CARDINALITY_TRANSFORM(dataframe,target, variable, k, f):

    '''

        dataframe = A pandas dataframe containing the data.
        target    = The name of the target variable
        variable  = The name of the categorical variable in which will be applied the transformation
        k         = Hyperparameter from the model. It is the threshold value. Example:
                    If this values is equal a 1000, if the category has 1000 values with target equal to 1
                    then the prior and the posterior distribution will be assign a weight of half each.

        f         = Smoothness parameter. The higher this parameter, the more abrupt will be the use of either
                    prior or posterior probability.

        lb_card   = Function of the lambda varible used to ponder between prior and posterior
        return df = Returns a datafram with a single column containg the transformed varible
    '''

    '''Function from paper'''

    def lb_card(n,k,f):
        lb = 1/(1+np.exp(-(n-k)/f))
    return lb


    df = dataframe.copy()
    g = pd.DataFrame(df.groupby(variable).count().iloc[:,0]).reset_index()
    posterior = pd.DataFrame(df[df[target]>0]
                             .groupby(variable).count().iloc[:,0]).reset_index().iloc[:,-1]/g.iloc[:,-1]
    prior = df[df[target]>0].count().iloc[0]/df.count().iloc[0]
    
    variable_tf = lb_card(g.iloc[:,-1],k,f)*posterior + (1-lb_card(g.iloc[:,-1],k,f))*prior
    
    g.iloc[:,-1] = variable_tf.values
    
    df[variable+'_micc'] = df.merge(g,on=variable,how='left').iloc[:,-1]
    df.drop(df.columns[0:-1],axis=1,inplace=True)

    return df
