# Pandas 연습 문제 101

[www.machinelearningplus.com/python/101-numpy-exercises-python/](https://www.machinelearningplus.com/python/101-numpy-exercises-python/)

[www.machinelearningplus.com/python/101-pandas-exercises-python/](https://www.machinelearningplus.com/python/101-pandas-exercises-python/)

#### 1. 다음 표를 생성하라





![image](https://user-images.githubusercontent.com/71866756/151500261-726fbb79-906a-45f4-8a2a-77b86befd24f.png)

```python
import pandas as pd
import dateutil

index=["2018-01-01", "2018-01-02","2018-01-03"]
index=list(map(dateutil.parser.parse, index))

data = pd.Series([737, 750, 770], index)
data1 = pd.Series([755, 780, 770], index)
data2 = pd.Series([700, 710, 750], index)
data3 = pd.Series([750, 770, 730], index)

df = pd.concat([data, data1, data2, data3], axis=1)
name = ["open", "high", "low", "close"]
df.columns = name
df["volatility"] = [55, 70, 20]

```



#### 2. How to create a series from a list, numpy array and dict?

```python
import numpy as np
mylist = list('abcedfghijklmnopqrstuvwxyz')
myarr = np.arange(26)
mydict = dict(zip(mylist, myarr))
```

- **Solution**

```python
import pandas as pd
s_list = pd.Series(mylist)
s_arr = pd.Series(myarr)
s_dict = pd.Series(mydict)
```



#### 3. How to convert the index of a series into a column of a dataframe?

```python
mylist = list('abcedfghijklmnopqrstuvwxyz')
myarr = np.arange(26)
mydict = dict(zip(mylist, myarr))
ser = pd.Series(mydict)
```

- **Solution**

```python
df = ser.to_frame().reset_index()
# to_frame()은 Series를 DataFrame으로 변경해주는 메소드이다. 
# reset_index()를 통해 0부터 index를 지정할 수 있고, 대신 이전 Series의 index도 DataFrame의 칼럼이 된다. 

df = ser.to_frame(name="hi").reset_index()
# 인자로 column 이름을 지정해줄 수 있다. 

df = ser.to_frame(name="hi").reset_index(drop=True)
# 기존의 index를 삭제하고 새로운 index만 남긴다. 

df
```



#### 4. How to combine many series to form a dataframe?

```python
import numpy as np
ser1 = pd.Series(list('abcedfghijklmnopqrstuvwxyz'))
ser2 = pd.Series(np.arange(26))
```

- **Solution1**

```python
df = pd.concat([ser1, ser2], axis=1)
# concat은 series끼리도 가능하다.
df
```

- **Solution2**

```python
df = pd.DataFrame({'col1':ser1, 'col2':ser2})
df
```



#### 5. How to assign name to the series' index?

```python
ser = pd.Series(list('abcedfghijklmnopqrstuvwxyz'))
```

- **Solution**

```python
ser.name = "alphabet"
ser
```



#### 6. How to get the items of series A not present in series B?

```python
ser1 = pd.Series([1, 2, 3, 4, 5])
ser2 = pd.Series([4, 5, 6, 7, 8])
```

- **Solution**

```python
ser1.isin(ser2)
# boolean으로 출력된다.
# ser1에서 ser2의 원소와 겹치는 값은 True가 된다.

ser1[~ser1.isin(ser2)]
# ~을 붙여 겹치지 않는 값을 True로 만들고, ser1에서 뽑아낸다.
# []안에 index를 넣어서 값을 뽑아내는 줄만 알았지만, 이런식으로 다른 series의 boolean 값을 넣어서 뽑을 수도 있는 것 같다. 
```



#### 7. How to get the items not common to both series A and series B?

```python
ser1 = pd.Series([1, 2, 3, 4, 5])
ser2 = pd.Series([4, 5, 6, 7, 8])
```

- **Solution1**

```python
pd.concat([ser1[~ser1.isin(ser2)], ser2[~ser1.isin(ser2)]
```

- **Solution2**

```python
import numpy as np

union = pd.Series(np.union1d(ser1,ser2))
intersect = pd.Series(np.intersect1d(ser1, ser2))
union[~union.isin(intersect)]
# numpy의 union1d와 intersect1d를 이용할 수 있다. 
```



#### 8. How to get the minimum, 25th percentile, median, 75th, and max of a numeric series?

```python
ser = pd.Series(np.random.normal(10, 5, 25))
```

- **Solution**

```python
state = np.random.RandomState(100)
# 난수 생성하는 코드로
# np.random.seed(seed=0)
# num = np.random.random(size=100)
# 과 같지만, np.random.seed를 사용하는 경우 전체 코드의 seed가 고정되지만, 
# RandomState의 경우, 이 object에 대해서만 seed가 정해진다. 

ser1 = pd.Series(state.normal(10, 5, 25))
np.percentile(ser1, q=[0,25,50,75,100])
# 데이터의 분위수(백분위)를 구하는 코드이다. 
# 25개의 데이터에서 백분위를 구해서 array형태로 반환한다.
```



#### 9. How to get frequency counts of unique items of a series?

```python
ser = pd.Series(np.take(list('abcdefgh'), np.random.randint(8, size=30)))
```

- **Solution**

```python
ser.value_counts()
```



#### 10. How to keep only top 2 most frequent values as it is and replace everything else as ‘Other’?

```python
np.random.RandomState(100)
ser = pd.Series(np.random.randint(1, 5, [12]))
```

```python
ser[~ser.isin(ser.value_counts(sort=True,ascending=False).index[:2])] = 'other'
# value_counts를 통해 가장 많은 순대로 내림차순으로 정렬 
# index가 해당 값, value가 해당  빈도수
# 해당 값에 속하지 않는 값들에 대해서 True로 바꿔준다. 
# 그 다음 ser에 indexing하여 Ture인 값들을 'other'로 변경
ser
```



#### 12. How to convert a numpy array to a dataframe of given shape? (L1)

```python
ser = pd.Series(np.random.randint(1, 10, 35))
```

- **Solution**

```python
df = pd.DataFrame(ser.values.reshape(7,5))
df
```



#### 13. How to find the positions of numbers that are multiples of 3 from a series?

```python
ser = pd.Series(np.random.randint(1, 10, 7))
```

- **Solution1**

```python
ser[ser.values % 3 == 0].index
```

- **Solution2**

```python
np.where(ser%3==0)
```



#### 14. How to extract items at given positions from a series

```python
ser = pd.Series(list('abcdefghijklmnopqrstuvwxyz'))
pos = [0, 4, 8, 14, 20]
```

- **Solution1**

```python
ser.take(pos)
```

- **Solution2**

```python
ser[pos]
```



#### 15. How to stack two series vertically and horizontally ?

```python
ser1 = pd.Series(range(5))
ser2 = pd.Series(list('abcde'))
```

- **Solution**

```python
pd.concat([ser1, ser2], axis=1)
```



#### 16. How to get the positions of items of series A in another series B?

```python
ser1 = pd.Series([10, 9, 6, 5, 3, 1, 12, 8, 13])
ser2 = pd.Series([1, 3, 10, 13])
```

- **Solution1**

```python
[int(np.where(ser1==i)[0]) for i in ser2]
```

- **Solution2**

```python
[pd.Index(ser1).get_loc(i) for i in ser2]
```



#### 17. How to compute the mean squared error on a truth and predicted series?

```python
truth = pd.Series(range(10))
pred = pd.Series(range(10)) + np.random.random(10)
```

- **Solution**

```python
np.mean((truth-pred)**2)
```



#### 18. How to convert the first character of each element in a series to uppercase?

```python
ser = pd.Series(['how', 'to', 'kick', 'ass?'])
```

- **Solution1**

```python
ser.map(lambda x:x.capitalize())
```

- **Solution2**

```python
ser.map(lambda x:x[0].upper()+x[1:])
```



#### 19. How to calculate the number of characters in each word in a series?

```python
ser = pd.Series(['how', 'to', 'kick', 'ass?'])
```

- **Solution**

```python
ser.map(lambda x:len(x))
```



#### 20.How to compute difference of differences between consequtive numbers of a series?

```python
ser = pd.Series([1, 3, 6, 10, 15, 21, 27, 35])
```

- **Solution**

```python
print( list(ser.diff()) )
print( list(ser.diff().diff()) )
```



####  21. How to convert a series of date-strings to a timeseries?

```python
ser = pd.Series(['01 Jan 2010', '02-02-2011', '20120303', '2013/04/04', '2014-05-05', '2015-06-06T12:20'])
```

- **Solution1**

```python
from dateutil.parser import parse
ser.map(lambda x: parse(x))
```

- **Solution2**

```python
pd.to_datetime(ser)
```



#### 22. How to get the day of month, week number, day of year and day of week from a series of date strings?

```python
ser = pd.Series(['01 Jan 2010', '02-02-2011', '20120303', '2013/04/04', '2014-05-05', '2015-06-06T12:20'])
```

- **Solution**

```python
ser_ts = pd.to_datetime(ser)

print( list(ser_ts.dt.day) )
print( list(ser_ts.dt.weekofyear) )
print( list(ser_ts.dt.dayofyear) )
print( list(ser_ts.dt.day_name()) )
```



#### 23. How to convert year-month string to dates corresponding to the 4th day of the month?

```python
ser = pd.Series(['Jan 2010', 'Feb 2011', 'Mar 2012'])
```

- **Solution**

```python
ser_ts = pd.to_datetime(ser)
ser_datestr = ser_ts.apply(lambda x : str(x.year)+'-'+str(x.month)+'-'+'4')
ser_date = pd.to_datetime(ser_datestr, format="%Y-%m-%d")
ser_date
```



#### 24. How to filter words that contain atleast 2 vowels from a series?

```python
ser = pd.Series(['Apple', 'Orange', 'Plan', 'Python', 'Money'])
```

- **Solution**

```python
from collections import Counter
mask = ser.apply(lambda x: sum([Counter(x.lower()).get(i, 0) for i in list('aeiou')]) >= 2)
ser[mask]
```



#### 25. How to filter valid emails from a series?

```python
emails = pd.Series(['buying books at amazom.com', 'rameses@egypt.com', 'matt@t.co', 'narendra@modi.com'])
pattern ='[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,4}'
```

- **Solution**

```python
mask = emails.apply(lambda x : len(re.findall(pattern, x)) > 0 )
emails[mask]
```



#### 26. How to get the mean of a series grouped by another series?

```python
fruit = pd.Series(np.random.choice(['apple', 'banana', 'carrot'], 10))
weights = pd.Series(np.linspace(1, 10, 10))
print(weights.tolist())
print(fruit.tolist())
```

- **Solution**

```python
weights.groupby(fruit).mean()
```



#### 27. How to compute the euclidean distance between two series?

```python
p = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
q = pd.Series([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
```

- **Solution**

```python
np.sqrt(((p-q)**2).sum())
```



#### 28. How to find all the local maxima (or peaks) in a numeric series?

```python
ser = pd.Series([2, 10, 3, 4, 9, 10, 2, 7, 3])
```

- **Solution**

```python
dd = np.diff(np.sign(np.diff(ser)))
np.where(dd==-2)[0]+1
```



#### 29. How to replace missing spaces in a string with the least frequent character?

```python
my_str = 'dbc deb abed gade'
```

- **Solution**

```
my_str = 'dbc deb abed gade'
s = pd.Series(list(my_str)).value_counts()
s.drop(" ", inplace=True)
my_str.replace(" ", s.index[-1])
```



#### 30. How to create a TimeSeries starting ‘2000-01-01’ and 10 weekends (saturdays) after that having random numbers as values?

- **Solution**

```python
ser = pd.Series(np.random.randint(1, 10, 10), pd.date_range("2000-01-01", periods=10, freq="W-SAT"))
ser
```



#### 31. How to fill an intermittent time series so all missing dates show up with values of previous non-missing date?

```
ser = pd.Series([1,10,3,np.nan], index=pd.to_datetime(['2000-01-01', '2000-01-03', '2000-01-06', '2000-01-08']))
```

- **Solution**

```python
ser.resample("D").ffill()
```



#### 32. How to compute the autocorrelations of a numeric series?

```python
ser = pd.Series(np.arange(20) + np.random.normal(1, 10, 20))
```

- **Solution**

```python
autocorrelations = list(ser.autocorr(i).round(2) for i in range(11))
autocorrelations[1:]
```



#### 33. How to import only every nth row from a csv file to create a dataframe?

- **Solution**

```python
df = pd.read_csv("test.csv", chunksize=10)
# 10개 단위로 df을 구성한다. 
# index는 계속해서 지속된다. 즉, 첫번째 df는 0~9까지, 두번째 df는 10~19까지 이런방식이다. 
df2 = pd.concat([chunk.iloc[0] for chunk in df])
# 따라서, 원하는 간격으로 한 row씩 가져오고 싶다면 이런식으로 하면 된다. 
```



#### 34. How to change column values when importing csv to a dataframe?

- **Solution**

```
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv', converters={'medv' : lambda x : 'High' if float(x) > 25 else "Low"})
df
```



#### 35. How to create a dataframe with rows as strides from a given series?

```
L = pd.Series(range(15))
```

- **Solution**

```python
def gen_strides(a, stride_len=5, window_len=5):
	n_strides = ((a.size-window_len)//stride_len)+1
	return np.array([a[s:(s+window_len)] for s in np.arange(0, a.size, stride_len)[:n_strides]])
	
gen_strides(L, stride_len=2, window_len=4)
```



#### 36. How to import only specified columns from a csv file?

- **Solution**

```
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv', usecols=["crim","medv"] )
df
```



#### 37. How to get the n*rows, n*columns, datatype, summary stats of each column of a dataframe? Also get the array and list equivalent.

- **Solution**

```
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
df.info()
```



#### 38. How to extract the row and column number of a particular cell with given criterion?

- **Solution**

```
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
df.loc[df.Price == np.max(df.Price), ["Manufacturer", 'Model', 'Type']]
row, col = np.where(df.values == np.max(df.Price))
df.iloc[row[0], col[0]]
```



#### 39. How to rename a specific columns in a dataframe?

```
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
print(df.columns)
```

- **Solution1**

```
df.columns = [column_name.replace('.', '_') for column_name in df.columns]
```

- **Solution2**

```
df.columns = df.columns.map(labmda x: x.replace('.', '_'))
```



#### 40. How to check if a dataframe has any missing values?

```
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
```

- **Solution**

```
df.isnull().values.any()
```



#### 41. How to count the number of missing values in each column?

```
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
```

- **Solution1**

```
missing = df.apply(lambda x: x.isnull().sum())
# 출력은 dataframe
```

- **Solution2**

```
missing = np.sum(df.isna().values, axis=0)
```



#### 42. How to replace missing values of multiple numeric columns with the mean?

```
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
```

- **Solution**

```python
df_out = df.apply(lambda x:x.fillna(x.mean()) if x.dtypes=='float64' else x)
# lambda에서 x로 받아오는 것은 칼럼 그 자체인 것 같다. 전체 칼럼에 대해서 병렬적으로 수행하는 느낌?
# df_out[np.array(df.columns)[list(df_out.dtypes=='float64')]].isna().sum()
# 제대로 됐는지 확인하는 코드
```

#### 43. How to use apply function on existing columns with global variables as additional arguments?

```
```

- **Solution**

```python
d = {'Min.Price': np.nanmean, 'Max.Price': np.nanmedian}
# np.nanmean은 NaN 값을 제외하고 나머지 값들의 평균을 계산
# np.nanmedian은 NaN 값을 제외하고 나머지 값들의 중앙값을 계산

df[['Min.Price', 'Max.Price']] = df[['Min.Price', 'Max.Price']].apply(lambda x, d: x.fillna(d[x.name](x)), args=(d, ))
# 여기서 전역변수인 d를 추가적인 argument로 넣어주기 위해서 args=(d,)를 추가한다.
# x.name은 칼럼의 이름을 가르킨다. 
```



#### 44. How to select a specific column from a dataframe as a dataframe instead of a series?

```python
df = pd.DataFrame(np.arange(20).reshape(-1, 5), columns=list('abcde'))
```

- **Solution**

```python
df[['a']]
# 안에 중괄호를 넣어주면 df로 반환하지만, 중괄호를 빼면 series로 반환한다.
```

#### 45. How to change the order of columns of a dataframe?

```python
df = pd.DataFrame(np.arange(20).reshape(-1, 5), columns=list('abcde'))
```

- **Solution**

```python
df[sorted(df.columns)[::-1]]
# 이렇게 하면 edcba순으로 칼럼이 뒤바뀐다.
# 단순히 칼럼명이 바뀌는 것이 아닌 값들도 순서가 바뀐다. 
```

#### 46. How to set the number of rows and columns displayed in the output?

```python
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
```

- Solution

```python
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 10)
# pd의 set_option 메소드를 통해 원하는 개수만큼의 columns와 rows를 출력할 수 있다. 
```

#### 47. How to format or suppress scientific notations in a pandas dataframe?

```python
df = pd.DataFrame(np.random.random(4)**10, columns=['random'])
```

- **Solution1**

```python
df['random']=df['random'].apply(lambda x: f'{x:.4f}')
```

- **Solution2**

```python
df.round(4)
# df.round({'random':2}) 
# random이라는 컬럼을 소수점 2개를 남기고 round하겠다. 
```

#### 48. How to format all the values in a dataframe as percentages?

```python
df = pd.DataFrame(np.random.random(4), columns=['random'])
```

- **Solution**

```python
df['random'].apply(lambda x:f'{x*100:.2f}%')
```

#### 49. How to filter every nth row in a dataframe?

```python
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv')
```

- **Solution**

```python
df.iloc[[x for x in range(0, len(df), 20)]][['Manufacturer', 'Model', 'Type']]
```

#### 50. How to create a primary key index by combining relevant columns?

```python
df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv', usecols=[0,1,2,3,5])
```

- **Solution**

```python
df[['Manufacturer', 'Model', 'Type']] = df[['Manufacturer', 'Model', 'Type']].apply(lambda x:x.fillna('missing'))
df.index = df['Manufacturer']+'_'+df['Model']+'_'+df['Type']
print(df.index.is_unique)
```

#### 51. How to get the row number of the nth largest value in a column?

```python
df = pd.DataFrame(np.random.randint(1, 30, 30).reshape(10,-1), columns=list('abc'))
```

- **Solution**

```python
n=5
df['a'].argsort()[::-1][n]
```

#### 52. How to find the position of the nth largest value greater than a given value?

```python
ser = pd.Series(np.random.randint(1, 100, 15))
```

- **Solution**

```python
n = 2
ser[ser>ser.mean()].sort_values()[::-1][n]
```

#### 53. How to get the last n rows of a dataframe with row sum > 100?

```python
df = pd.DataFrame(np.random.randint(10, 40, 60).reshape(-1, 4))
```

- **Solution**

```python
df[df.sum(axis=1)>100][-n:]
```

#### 54. How to find and cap outliers from a series or dataframe column?

```python
ser = pd.Series(np.logspace(-2, 2, 30))
```

- **Solution**

```python
Q1 = np.percentile(ser.values, 5)
Q3 = np.percentile(ser.values, 95)
ser[(ser<Q1)]= Q1
ser[ser>Q3] = Q3
```

#### 56. How to swap two rows of a dataframe?

```python
df = pd.DataFrame(np.arange(25).reshape(5, -1))
```

- **Solution**

```python
temp = df.iloc[0].copy()
df.iloc[0] = df.iloc[1]
df.iloc[1] = temp
```

#### 57. How to reverse the rows of a dataframe?

```python
df = pd.DataFrame(np.arange(25).reshape(5, -1))
```

- **Solution**

```python
df.iloc[::-1, :]
```

#### 58. How to create one-hot encodings of a categorical variable (dummy variables)?

```python
df = pd.DataFrame(np.arange(25).reshape(5,-1), columns=list('abcde'))
```

- **Solution**

```python
df_onehot = pd.concat([pd.get_dummies(df['a']), df[list('bcde')]], axis=1)
# 이렇게 하면 column명이 a의 값이 된다. 

pd.get_dummies(df, columns=['a'], prefix='a')
# 이렇게 하면 column명이 a_1, a_2이런식으로 된다. 
```

#### 59. Which column contains the highest number of row-wise maximum values?

```python
df = pd.DataFrame(np.random.randint(1,100, 40).reshape(10, -1))
```

- **Solution**

```python
df.apply(np.argmax, axis=1).value_counts().index[0]
# apply에 np.argmax, axis=1을 넣어서 각 row의 가장 큰 값의 column을 따오고, 
# value_counts를 통해 각 column의 개수를 센다.
# 이 때, 내림차순 정렬되므로, index의 0번째가 큰 값이 가장 많은 column이다. 
```

#### 61. How to know the maximum possible correlation value of each column against other columns?

```python
df = pd.DataFrame(np.random.randint(1,100, 80).reshape(8, -1), columns=list('pqrstuvwxy'), index=list('abcdefgh'))
```

- **Solution**

```python
np.round(np.abs(df.corr()).apply(lambda x: sorted(x)[-2]).tolist(), 2)
# corr은 column끼리의 상관계수를 구함
# np.abs를 통해 절대값을 구함
# column별로 sort하고, 최대는 자기자신인 1이므로 1바로 앞의 숫자를 가져와서 list만듬
# 소수점 셋째자리에서 반올림
```

#### 62. How to create a column containing the minimum by maximum of each row?

```python
df = pd.DataFrame(np.random.randint(1,100, 80).reshape(8, -1))
```

- **Solution**

```python
df.apply(lambda x:np.min(x)/np.max(x), axis=1)
# apply에 axis를 추가하면 행 단위 연산도 가능하다. 
```

#### 63. How to create a column that contains the penultimate value in each row?

```python
df = pd.DataFrame(np.random.randint(1,100, 80).reshape(8, -1))
```

- **Solution**

```python
out = df.apply(lambda x: x.sort_values().unique()[-2], axis=1)
# 여기서 unique()나 to_list()를 써야 각 row마다 정렬된 값이 나온다. 
df['penultimate'] = out
df
```

#### 64. How to normalize all columns in a dataframe?

```python
df = pd.DataFrame(np.random.randint(1,100, 80).reshape(8, -1))
```

- **Solution**

```python
out = df.apply(lambda x : (x-np.mean(x))/np.std(x))
out = df.apply(lambda x : ((x.max()-x)/(x.max()-x.min())))
out
```

#### 65. How to compute the correlation of each row with the suceeding row?

```python
df = pd.DataFrame(np.random.randint(1,100, 80).reshape(8, -1))
```

- **Solution**

```python
[df.iloc[i].corr(df.iloc[i+1]).round(2) for i in range(df.shape[0])[:-1]]
```

#### 66. How to replace both the diagonals of dataframe with 0?

```python
df = pd.DataFrame(np.random.randint(1,100, 100).reshape(10, -1))
```

- **Solution**

```python
for i in range(df.shape[0]):
    df.iloc[i, i] = 0
df
```

#### 67. How to get the particular group of a groupby dataframe by key?

```python
df = pd.DataFrame({'col1': ['apple', 'banana', 'orange'] * 3,
                   'col2': np.random.rand(9),
                   'col3': np.random.randint(0, 15, 9)})

df_grouped = df.groupby(['col1'])
```

- **Solution1**

```python
for i, diff in df_grouped:
    if i=='apple':
        print(diff)
```

- **Solution2**

```pytho
df_grouped.get_group('apple')
```

#### 68. How to get the n’th largest value of a column when grouped by another column?

```python
df = pd.DataFrame({'fruit': ['apple', 'banana', 'orange'] * 3,
                   'rating': np.random.rand(9),
                   'price': np.random.randint(0, 15, 9)})
```

- Solution

```python
df.groupby(['fruit'])['taste'].get_group('banana').sort_values().iloc[-2]
# get group해서 banana그룹을 가져온다. 
# 이때 value는 taste이고 series형태로 가져온다.
# series역시 dataframe과 마찬가지로 iloc으로 접근가능
```

#### 69. How to compute grouped mean on pandas dataframe and keep the grouped column as another column (not index)?

```python
df = pd.DataFrame({'fruit': ['apple', 'banana', 'orange'] * 3,
                   'rating': np.random.rand(9),
                   'price': np.random.randint(0, 15, 9)})
```

- **Solution**

```python
df.groupby(['fruit'])['price'].mean().reset_index()
# reset_index를 통해 원래는 index였던 fruit이 칼럼으로 들어가게 된다. 
```

#### 70. How to join two dataframes by 2 columns so they have only the common rows?

```python
df1 = pd.DataFrame({'fruit': ['apple', 'banana', 'orange'] * 3,
                    'weight': ['high', 'medium', 'low'] * 3,
                    'price': np.random.randint(0, 15, 9)})

df2 = pd.DataFrame({'pazham': ['apple', 'orange', 'pine'] * 2,
                    'kilo': ['high', 'low'] * 3,
                    'price': np.random.randint(0, 15, 6)})
```

- **Solution**

```python
pd.concat([df1, df2], axis=1, join='inner')
```

#### 71. How to remove rows from a dataframe that are present in another dataframe?

```python
df1 = pd.DataFrame({'fruit': ['apple', 'banana', 'orange'] * 3,
                    'weight': ['high', 'medium', 'low'] * 3,
                    'price': np.random.randint(0, 15, 9)})

df2 = pd.DataFrame({'pazham': ['apple', 'orange', 'pine'] * 2,
                    'kilo': ['high', 'low'] * 3,
                    'price': np.random.randint(0, 15, 6)})
```

- **Solution**

```python
df1[~df1.isin(df2).all(1)]
# all에서 1은 axis를 가르키고, df1의 한 로우가 모두 df2에 있다면 선택하지 않는다. 
```

#### 72. How to get the positions where values of two columns match?

```python
df = pd.DataFrame({'fruit1': np.random.choice(['apple', 'orange', 'banana'], 10),
                    'fruit2': np.random.choice(['apple', 'orange', 'banana'], 10)})
```

- **Solution**

```python
np.where(df['fruit1']==df['fruit2'])
```

#### 73. How to create lags and leads of a column in a dataframe?

```python
df = pd.DataFrame(np.random.randint(1, 100, 20).reshape(-1, 4), columns = list('abcd'))
```

- **Solution**

```python
df['a_lag1'] = [np.nan] + [lag for lag in df['a'][:-1]]
df['b_lead1'] = [lead for lead in df['b'][1:]] + [np.nan]
# 휴리스틱한 방법

df['a_lag1'] = df['a'].shift(1)
df['b_lead1'] = df['b'].shift(-1)
```

#### 74. How to get the frequency of unique values in the entire dataframe?

```python
df = pd.DataFrame(np.random.randint(1, 10, 20).reshape(-1, 4), columns = list('abcd'))
```

- **Solution**

```python
pd.value_counts(df.values.reshape(-1))
# 안에 value는 numpy이고, value_counts는 pandas메소드이기 때문에
# 뒤에 .value_counts는 안된다. 
```

#### 75. How to split a text column into two separate columns?

```python
df = pd.DataFrame(["STD, City    State",
"33, Kolkata    West Bengal",
"44, Chennai    Tamil Nadu",
"40, Hyderabad    Telengana",
"80, Bangalore    Karnataka"], columns=['row'])
```

- **Solution**

```python
df_out = df.row.str.split(',|    ', expand=True)
# 처음에 row 칼럼을 split하기 위해서는 .str을 붙여줘야 한다.
# 그리고 expand=True로 놓는다. 
new_header = df_out.iloc[0]
df_out = df_out[1:]
df_out.columns = new_header
df_out
```

