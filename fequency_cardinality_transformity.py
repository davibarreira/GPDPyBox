import pandas as pd

def FREQUENCY_CARDINALITY(dataframe,variable):
    df = dataframe.copy()
    g = pd.DataFrame(df.groupby(variable).count().iloc[:,0]).reset_index()
    g.iloc[:,-1] = g.iloc[:,-1]/df.count().iloc[0]
    g.iloc[:,-1] = g.sort_values(by=g.columns[-1]).cumsum().iloc[:,-1]
    df[variable+'_freq'] = df.merge(g,on=variable,how='left').iloc[:,-1]
    return df
