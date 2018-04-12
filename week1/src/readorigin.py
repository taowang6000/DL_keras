import pandas as pd
import matplotlib.pyplot as plt

def dateparse(x):
    return pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
#Falling back to the 'python' engine because the 'c' engine does not support regex separators 
#(separators > 1 char and different from '\s+' are interpreted as regex);
series = pd.read_csv('1.csv', sep = '\\t', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=dateparse)

series.to_csv('transformed.csv', sep = ',')

series.plot()
plt.show()
print(series.head())
