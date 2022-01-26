# Pandas 연습 문제 101

#### 1. 다음 표를 생성하라

![image-20220125232701553](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220125232701553.png)

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

