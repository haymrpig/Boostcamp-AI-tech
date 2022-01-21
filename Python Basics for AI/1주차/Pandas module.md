# Pandas Module

- pd.read_csv()

  ```python
  import pandas as pd
  
  data_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
  df_data = pd.read_csv(data_url, sep='\s+', header=None)
  # sep은 구분자로, \s+은 정규표현식으로 띄어쓰기가 여러개라는 뜻이다. 
  ```

- head()

  ```python
  df_data.head()
  # 가져온 데이터 중, 상위 5개를 default로 가져온다. 
  # 값을 입력하면 그 개수만큼 가져온다. 
  ```

  ![image-20220120155142934](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220120155142934.png)

- columns

  ```python
  name = "a b c d e f g h i j k l m n".split()
  df_data.columns = name
  # 리스트 형태로 column명을 지정해줄 수 있다. 
  
  df_data.head()
  ```

  ![image-20220120155430740](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220120155430740.png)

- values

  ```python
  df_data.values
  # 값은 numpy type으로 되어있다. 
  ```

  ![image-20220120155526251](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220120155526251.png)

- **Series**

  `series` : 한 줄의 column

  - Series()

    ```python
    from pandas import Series, DataFrame
    import pandas as pd
    
    list_data = [1,2,3,4,5]
    sample = Series(data = list_data)	
    print(sample)	# index(0부터)와 data가 같이 나온다. 
    
    list_name = "a b c d e".split()
    sample2 = Series(data=list_data, index=list_name)
    print(sample2)	# index가 a,b,c,d,e
    
    dict_data = {"a" : 1, "b": 2}
    sample3 = Series(data=dict_data, dtype=np.float32, name="data")
    print(sample3)	# index는 key, 값은 value, type은 float, table이름은 data
    ```

  - index로 값 접근

    ```python
    print( sample3["a"] )
    
    sample3["a"]=3.1
    print( sample3 )
    
    sample3 = sample3.astype(int)
    print( sample3 )
    ```

  - values, index, name

    ```python
    print( sample3.values )	# 값 리스트 반환
    print( sample3.index )	# index 리스트 반환
    
    sample3.name = "data3"
    print( sample3.name )
    
    sample3.index.name = "alphabet" # index의 이름 변경 보통 default로 해서 잘 안쓴다. 
    print( sample3 )
    
    dict_data_1 = {"a":1, "b":2, "c":3}
    indexes = ["a", "b", "c", "d"]
    sample4 = Series(dict_data_1, index=indexes)
    print( sample4 )
    # 값은 3까지지만 index기준으로 생성하기 때문에, "d" index의 경우 NaN의 값으로 채워 table을 뱉는다. 
    ```

- **DataFrame**

  `DataFrame` : 테이블 전체 (보통 csv나 excel로 불러옴)

  ```python
  raw_data = {'first_name' : ['jason', 'hong'],
  			'age' : [1,2]}
  			
  df = pd.DataFrame(raw_data, columns = ['first_name', 'age'])
  print( df )
  print( df.first_name )	# column 추출 (Series 데이터로 추출)
  print( df["first_name"] )
  ```

  - loc, iloc

    ```python
    df.loc[:3]	
    df.loc[[1,2],["first_name", "age"]]
    df.loc[:, ["first_name", "age"]]	# 이런식으로 원하는 column만 가져올 수 있다. 
    # index이름으로 추출
    # 만약 index순서가 51,52,1,2 이렇게 되어있으면 51부터 2까지 추출한다. 
    
    df["age"].iloc[1:]	
    df.iloc[1:]
    # index number로 추출
    # 즉, index가 문자로 되어있든 숫자로 되어있든 그 개수만큼만 추출
    ```

  - 새로운 데이터 할당

    ```python
    df_new = df.age > 40	# 조건을 걸어 새로 할당
    print( df_new )
    
    values = Series(data=["M","M","M"], index=[0,1,2])
    df["sex"]=values
    print(df)
    # Series 할당
    ```

  - transpose

    ```python
    df.T
    ```

  - to_csv()

    ```python
    df.to_csv()
    # csv형태로 변환
    # 매개변수로 파일로 저장도 가능
    ```

  - drop(), del

    ```python
    del df["age"]		# 메모리 자체를 삭제하는 거라, 실제 dataframe에 영향 O
    df.drop("age", axis=1)	# column 기준 삭제, 실제 dataframe에 영향 X
    ```

- Selection & drop

  - feature로 뽑기

    ```python
    df["age"].head(3)	# 3개의 row까지 출력
    
    df[["age", "first_name"]].head(3)	
    
    df["age"]	# Series 데이터로 뽑힘
    df[["age"]]	# dataframe으로 뽑힘
    ```

    

  - index로 뽑기

    ```python
    df[:3]
    # index가 0,1,2,3 이런식으로 되어있으면 뽑히지만, 알파벳 등으로 되어있으면 안뽑힌다. 
    
    df[[0,1,3]]	# index 0,1,3을 뽑음
    
    df[df<3000] # 범위로 boolean으로 뽑음, index가 3000이하인 것 다 뽑음
    
    # index를 새로 할당
    df.index = df["account"]
    
    df[["first_age"]][:2]	# column 명 먼저
    ```

  - reset_index()

    ```python
    df.index = list(range(0,15))
    # 새로 index 할당
    
    df.reset_index(inplace=True, drop=True)
    # 새로 index할당
    # drop은 기존 index 삭제
    # inplace는 기존 df 대체
    ```

  - drop

    ```python
    df.drop(1, inplace=True)
    # 1번 index 삭제
    # 기존 df 대체
    
    df.drop("city", axis=1)
    # city 칼럼 삭제
    # city 대신 index 넣고, axis=0하면 row 삭제
    # 마찬가지로 inplace 넣어야 기존 df 대체
    ```

  

- **DataFrame operation**

  series끼리, dataframe끼리, series와 dataframe끼리도 가능

  - add ( + , 숫자만 가능)

    ```python
    s1 = Series(range(0,2), index=list("ab"))
    s2 = Series(range(5,7), index=list("bc"))
    
    s1.add(s2)
    s1+s2
    # 같은 index끼리 더한다. index가 겹치지 않는 것은 NaN으로 출력
    
    s1.add(s2, fill_value=0)
    # NaN값을 0으로 채운다. 
    
    df.add(s2, axis=0)
    # row로 더함, broadcasting
    ```

  - lambda, map, unique, replace

    ```python
    s1 = Series(np.arange(10))
    print( s1.head(4) )
    
    s1.map(lambda x:x**2).head(4)
    # 모든 값을 제곱
    
    z = {1:'A', 2:'B'}
    s1.map(z).haed(5)
    # 기존 값을 z로 대체, s1에는 있지만, z에는 없는 index의 경우 해당 row의 값을 NaN으로 채움
    
    s2 = Series(np.arange(10,20))
    s1.map(s2).head(5)
    # 기존 s1을 s2로 대체
    # 모두 index 기준 mapping
    
    df.sex.unique()
    # sex 칼럼의 값을 중복없이 list로 보여줌
    df["sex_code"]=df.sex.map({"male":0, "female":1})
    # 새로운 sex_code 칼럼에 sex 칼럼에서의 값이 male일 경우 0으로, female일 경우 1로 채운다. 
    df.sex.replace({"male":0, "female":1})
    # 기존 column의 값에 따라 대체도 가능하다. 
    df.sex.replace(["male", "female"], [0,1], inplace=True)
    ```

  - apply

    ```python
    df.apply(lambda x:x.max()-x.min())
    # 전체 dataframe에 대한 연산으로
    # 각각의 column마다 함수가 적용되어 나온다. 
    
    df.sum()
    df.apply(sum)
    # 모든 dataframe의 column마다 적용
    
    def f(x):
        return Series([x.min(), x.max()], index=["min", "max"])
    df.apply(f)
    ```

  - applymap

    ```python
    f = lambda x:-x
    df.applymap(f).head(5)
    # 모든 값에 대해 적용
    
    df["first_age"].apply(f).head(5)
    ```

  - value_counts

    ```python
    df.sex.value_counts(sort=True)
    # 각각의 value의 개수를 count한다.
    
    df.sex.value_counts(sort=True) / len(df)
    # 비율로 출력이 가능하다. 
    ```

    

- **정보**

  - describe()

    ```python
    df.describe()
    # 숫자나, boolean 값들을 보여줌
    ```

  - isnull

    ```python
    df.isnull()
    # True, false로 반환
    df.isnull().sum()
    # 얼마나 값이 채워져 있는지 통계적으로 확인 가능
    ```

  - sort_values()

    ```python
    df.sort_values(["age", "earn"], ascending=True).head(5)
    # ascending : 오름차순 정렬
    # age먼저 정렬 후, earn 정렬
    ```

  - corr, cov, corrwith

    ```python
    df.age.corr(df.earn)
    # 상관계수 구하기
    
    df.age.cov(df.earn)
    # 공분산 구하기
    
    df.corrwith(df.earn)
    # 모든 column에 대해 earn과의 상관계수 구하기
    
    df.corr()
    # 모든 column끼리의 상관계수 구하기
    
    df.age[(df.age < 45) & (df.age > 15)].corr(df.earn)
    # boolean으로 제약 걸어줌
    
    df.dtypes
    # type보기
    ```

  - pd.options.display.max_rows

    ```python
    pd.options.display.max_rows = 100
    # row가 많을 때, ...으로 보여지는데 이 코드를 적으면 그 개수만큼 다 보여준다.
    ```

- **groupby**

  - groupby

    ```python
    df.groupby("Team")["Points"].sum()
    # 기준이 되는 칼럼 : Team
    # 기준 칼럼의 value들의 Points들의 sum을 출력
    
    df.groupby(["Team", "Year"])["Poinst"].sum()
    # Team으로 묶고, 그 안에서 Year로 묶어서 해당 Points들의 합 출력
    # 두개로 묶는 경우 index가 두 개 생긴다. 
    ```

  - unstack(), stack()

    ```python
    df.unstack()
    # data를 matrix 형태로 풀어준다.
    
    df.stack()
    # 다시 groupby로 묶기
    
    df.reset_index()
    # 원래 Team, year순으로 index가 묶여있었는데, index를 다시 풀어준다. 
    ```

  - swaplevel()

    ```python
    df.swaplevel()
    # 원래 Team이 1번, Year가 2번 index인데, 둘의 순서를 바꾼다. 
    
    df.swaplevel().sort_index(level=0)
    # 0번 level의 index 기준으로 sort한다. 
    
    df.sort_values()
    # default level 기준으로 sort
    
    df.sum(level=0)
    df.std(level=0)
    # multi index의 경우 level만 정해주면 기본 연산 수행 가능
    ```

  - grouped

    ```python
    grouped = df.groupby("Team")
    # grouped된 상태로 추출할 수 있음
    # generator 형태이다. 
    
    for name, group in grouped:
    	print(name)	# 기준 그룹 이름
    	print(value) # 그룹된 value들
        
    grouped.get_group
    ```

  - get_group

    ```python
    grouped.get_group("Devils")
    # 특정 key값을 가진 그룹의 정보만 추출 가능
    ```

  - aggregation

    요약된 통계정보를 추출

    ```python
    grouped.agg(sum)
    grouped.agg(max)
    # 이게 같은 row가 아닐 수도 있다. 각 칼럼별로 계산됨
    grouped.agg(np.mean)
    grouped['Points'].agg([np.sum, np.mean, np.std])
    ```

    

  - transformation

    해당 정보 변환

    ```python
    score = lambda x: (x.max())
    score1 = lambda x: (x-x.mean())/x.std()
    # 정규화
    grouped.transform(score)
    # 각 컬럼별로 적용
    ```

    

  - filteration

    특정 정보를 제거하여 보여주는 필터링 기능

    ```python
    df.groupby('Team').filter(lambda x:len(x)>=3)
    df.groupby('Team').filter(lambda x:x["Rank"].sum() > 2)
    ```

- **Data 분석하기**

  ```python
  import dateutil
  import pandas as pd
  import matplotlib.pyplot as plt
  
  data_url = 'https://www.shanelynn.ie/wp-content/uploads/2015/06/phone_data.csv'
  df_data = pd.read_csv(data_url, sep=',')
  # sep은 구분자로, \s+은 정규표현식으로 띄어쓰기가 여러개라는 뜻이다. 
  print( df_data.dtypes )
  print( df_data.head() )
  
  df_data["date"] = df_data["date"].apply(dateutil.parser.parse, dayfirst=True)
  # dateutil.parser.parse는 문자형태로 되어있는 데이터를 날짜로 바꾼다. 
  print( df_data.dtypes)
  
  df_data.groupby("month")["duration"].sum().plot()
  # 월별 통화량을 plot함 (matplotlib의 plot)
  
  df_data[df_data["item"]=="call"].groupby("month")["duration"].sum().plot()
  # item column의 값이 call인 것만 취합
  
  print( df_data.groupby(["month", "item"])["duration"].count() )
  print( df_data.groupby(["month", "item"])["duration"].count().unstack() )
  df_data.groupby(["month", "item"])["duration"].count().unstack().plot()
  
  df_data.groupby("month", as_index=False).agg({"duration" : "sum"})
  df_data.groupby("month").agg({"duration" : "sum"}).reset_index()
  # 위 두 코드는 동일하게 출력된다. as_index는 month를 index로 할 것인지 아닌지를 얘기한다.
  
  df_data.groupby(["month","item"]).agg({"duration" : sum,
                                         "network_type" : "count", 
                                         "date" : "first"})
  # first는 각 group의 첫번째 값을 의미한다.
  
  df_data.groupby(["month","item"]).agg({"duration" : [min, max, np.mean,sum],
                                         "network_type" : "count", 
                                         "date" : "first"})
  # first는 각 group의 첫번째 값을 의미한다.
  
  grouped = df_data.groupby("month").agg({"duration" : [min, max, np.mean]})
  grouped.rename(
      columns={"min" : "min_duration", "max" : "max_duration"}
  )
  # 컬럼 이름 변경하기
  
  grouped = df_data.groupby("month").agg({"duration" : [min, max, np.mean]})
  grouped.add_prefix("duration_")
  # 모든 컬럼 이름 앞에 duration_ 추가하기
  ```

- **Pivot Table**

  column에 labeling값 추가 가능 (groupby 후 unstack과 동일)

  ```python
  df_data.pivot_table(["duration"],
  					index=(df_data.month, df_data.item),
  					columns=df_data.network, aggfunc="sum", fill_value=0)
  # fill_value : NaN은 0으로 채운다. 
  ```

  

- **Crosstab**

  두 칼럼의 교차 빈도, 비율, 덧셈 등을 구할 때 사용한다. 

  ```python
  pd.crosstab(index=df.movie.critic, columns=df_movie.title, values=df_movie.ratin, aggfunc="first".fillna(0))
  # 평론가가 해당 영화에 대한 평점을 x,y축으로 표현한 테이블
  # NaN은 0으로 채운다. 
  
  df_movie.groupby(["critic", "title"]).agg({"rating":"first"}).unstack()
  # 위 코드와 동일한 결과가 나온다. 
  ```

- **Merge & concatenation**

  - merge

    ```python
    pd.merge(df_a, df_b, on='subject_id')
    # on의 경우 같은 값만 합쳐진다. 
    
    pd.merge(df_a, df_b, left_on="subject_id", right_on="subject_id1")
    # 합칠 기준 컬럼의 이름이 다른 경우 위 코드처럼 지정해서 합칠 수 있다. 
    ```

  - merge how

    ```python
    pd.merge(df_a, df_b, on="subject_id", how="left")
    # left merge
    
    pd.merge(df_a, df_b, on="subject_id", how="right")
    # right merge
    
    pd.merge(df_a, df_b, on="subject_id", how="inner")
    # 교집합 (default)
    
    pd.merge(df_a, df_b, on="subject_id", how="outer")
    # 합집합
    ```

  - index값 기준으로 합치기

    ```python
    pd.merge(df_a, df_b, right_index=True, left_index=True)
    # 같은 index값끼리 붙임, 같은 column명을 가진 feature들은 column_x, column_y처럼 두개가 생겨서 나중에 지워야 할 수도 있다. 
    ```

  - concat, append

    ```python
    df_new = pd.concat([df_a, df_b])
    # 두 dataframe row방향으로 붙이기
    
    df_a.append(df_b)
    # 위 코드와 같은 역할이다.
    # 두 코드 모두 기본적으로 같은 컬럼을 가지고 있어야 한다. 
    
    df_new = pd.concat([df_a, df_b], axis=1)
    df_new.reset_index(drop=True)
    # 옆으로 붙이기
    ```



- **Database connection**

  ```python
  import sqlite3
  
  conn = sqlite3.connect("./data/flights.db")
  cur = conn.cursor()
  cur.execute("select * from airlines limit 5;")
  results = cur.fetchall()
  results
  
  df_airplanes = pd.read_sql_query("select * from airlines;", conn)
  df_airports = pdf.read_sql_query("select * from airports;", conn)
  df_routes = pdf.read_sql_query("select * from routes;", conn)
  ```

  ```python
  !conda install --y XlsxWriter
  !conda install openpyxl
  # 둘 중 하나 설치
  
  writer = pd.ExcelWriter("./data/df_routes.xlsx", engine="xlsxwriter")
  df_routes.to_excel(writer, sheet_name="Sheet1")
  
  df_routes.to_pickle("./data/df_routes.pickle")
  
  df_routes_pickle = pd.read_pickle("./data/df_routes.pickle")
  df_routes_pickle.head()
  df_routes_pickle.describe()
  ```

  