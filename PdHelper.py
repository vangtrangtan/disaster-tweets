import pandas as pd

def print_full(x):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.float_format', '{:20,.2f}'.format)
    pd.set_option('display.max_colwidth', None)
    print("-----------FULL DATA --------------")
    print(x)
    print("-----------END--------------")
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')

def print_all_column_stats(df):
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.float_format', '{:20,.2f}'.format)
    pd.set_option('display.max_colwidth', None)
    print("-----------OVERVIEW DATA --------------")
    print(df.describe(include = 'all'))
    print("\n--------COUNT NAN VALUES--------\n")
    print(df.isna().sum())
    print("\n--------DATA TYPES--------------\n")
    print(df.dtypes)
    print("-----------END--------------")
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')

def get_values_distribute(fieldname1,fieldname2,df):
    #get percentage fieldname2/fieldname1
    df['count'] = df.groupby([fieldname1, fieldname2])[fieldname2].transform('count')
    df['sum'] = df.groupby([fieldname1])[fieldname1].transform('count')
    df['rate'] = df['count'] / df['sum']
    res = list(set([(df[fieldname1][idx],df[fieldname2][idx], df['rate'][idx]) for idx in df[[fieldname1,fieldname2, 'rate']].dropna().index]))
    res.sort(key= lambda x:x[0])
    return res

def write_df_to_csv(df,path):
    df.to_csv(path, header=True, index=False)