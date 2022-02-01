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

```
df = pd.concat([ser1, ser2], axis=1)
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



####  
