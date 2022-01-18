# 목차

1. **컴퓨터 OS**

2. **파일 시스템**

3. **터미널**

4. **Python**

   

# 1. 컴퓨터 OS

Operating System (운영체제)는 프로그램이 동작할 수 있는 구동 환경이다. 쉽게 말하면 software와 hardware를 연결하는 기본 운영체제를 의미한다.



# 2. 파일 시스템

os에서 파일을 저장하는 트리구조의 저장 체계이다.  

- 디렉토리

  폴더 또는 디렉토리로 파일과 다른 디렉토리를 포함할 수 있다. 

- 파일

  컴퓨터에서 정보를 저장하는 논리적인 단위로 파일명과 확장자로 식별되고, 실행, 쓰기, 읽기 등을 수행할 수 있다. 

- 경로

  파일의 고유한 위치로, 트리구조상의 노드의 연결된다.

  - 절대 경로

    루트 디렉토리부터 파일위치까지의 경로이다. 

  - 상대 경로

    현재 있는 디렉토리부터 타겟 파일까지의 경로이다. 



# 3. 터미널

- CLI (Command Line Interface)

  GUI와 달리 텍스트를 사용하여 컴퓨터에 명령을 입력하는 인터페이스 체계이다. 

  ex) Window - CMD window, Windows Terminal, cmder

  ​	  Mac, Linux - Terminal



# 4. Python

- **파이썬 컨벤션**

  파이썬 언어로 작성 시 지켜야할 규칙들을 의미한다. 

  black을 install하면 알아서 고쳐준다. 

  ![image](https://user-images.githubusercontent.com/71866756/149901096-b70581bc-d3bf-4392-84e5-f66b654d07af.png)

  

- **pythonic code**

  단순 for loop append보다 list가 조금 더 빠르고 코드도 짧아진다. 

  많은 개발자들이 python 스타일로 코딩하여 이해하기 편하다. 

  - split

    ```python
    example = "my, name, is, hong"
    for word in example.split(','):	# , 기준으로 나눔
        print(word.strip())			# 좌, 우 공백 삭제
    ```

    ![image](https://user-images.githubusercontent.com/71866756/149901143-ca8a4c1e-d463-48c0-95ca-e16860b3153a.png)

  - join

    ```python
    example = ['my','name', 'is','hong']
    result = ''.join(example)
    print(result)
    
    result = '-'.join(example)
    print(result)
    
    # ' '사이 문자를 추가하여 list의 각 원소들을 합침
    ```

    ![image](https://user-images.githubusercontent.com/71866756/149901181-f9117a1d-5ed6-4690-9329-26869fa60689.png)

    

  - list comprehension

    ```python
    result = [i for i in range(10) if i % 2 == 0]
    print(result)
    ```

    ![image](https://user-images.githubusercontent.com/71866756/149901287-87d79394-06f4-43a4-a59b-0f1d7a5132e5.png)

    

    ```python
    word1 = "hong"
    word2 = "ten"
    result = [i+j for i in word1 for j in word2]
    print(result)
    # 이중 for문을 생각하면 된다. 
    ```

    ![image](https://user-images.githubusercontent.com/71866756/149901330-3b7f7af0-9fb9-43f2-afff-0cf561dec192.png)

    

    ```python
    word1 = "hong"
    word2 = "ten"
    result = [i+j for i in word1 for j in word2 if not(i==j)]
    print(result)
    
    result_ = [i+j if not(i==j) else i for i in word1 for j in word2]
    print(result_)
    # if문 하나일 경우 맨 뒤로
    # if else문일 경우 중간에
    ```

    ![image](https://user-images.githubusercontent.com/71866756/149901383-7b1b1a42-4329-474a-a475-8d2ff10e45e3.png)

    

    ```python
    import pprint
    words = 'The quick brown fox jumps over the lazy dog'.split()
    print(words)
    
    stuff = [[w.upper(), w.lower(), len(w)] for w in words]
    pprint.pprint(stuff)
    # 2차원 배열
    # pprint는 세로로 출력
    ```

    ![image](https://user-images.githubusercontent.com/71866756/149901443-d9b63354-fa44-4d91-9ffb-7205776ec079.png)

    

  - lambda

    ```python
    f = lambda x, y : x + y
    print(f(1,3))
    print( (lambda x, y : x + y)(1,3) )
    ```

    ![image](https://user-images.githubusercontent.com/71866756/149901487-d55325f4-f02a-4017-b2e3-c8c64975a2f3.png)

    

    ```python
    twice = lambda x : "-".join(x.split()*2)
    print(up_low("My Happy"))
    ```

    ![image](https://user-images.githubusercontent.com/71866756/149901535-05300e35-ae06-4e18-a46a-3af99232af38.png)

    

    ```python
    def f(x,y):
        return x * y + 5
    
    example = [1,2,3,4,5]
    print(list(map(f, example, example)))
    ```

    ![image](https://user-images.githubusercontent.com/71866756/149901571-74d205a9-c273-4e35-87c4-eb376910e10f.png)

    

  - reduce function

    ```python
    from functools import reduce
    
    print(reduce(lambda x, y: x+y, [1,2,3,4,5]), end=" end")
    # 누적
    ```

    ![image](https://user-images.githubusercontent.com/71866756/149901624-2d7219ed-6921-4b4b-8f38-761d698fe097.png)

    

  - iter

    ```python
    cities = ['a','b','c']
    data_iter = iter(cities)
    print(next(data_iter))
    print(next(data_iter))
    print(next(data_iter))
    # iteration 같은 경우, 다음 element의 주소를 가지고 있기 때문에
    # next 함수를 통해서 불러올 수 있다. 
    ```

    ![image](https://user-images.githubusercontent.com/71866756/149901665-8d8b93cc-8021-4076-ac94-b15ff11f5dc6.png)

    

    

  - generator

    ```python
    def generator_list(value):
        result = []
        for i in range(value):
            yield i
    # yield는 실행되는 시점에서 값을 던져준다. 
    # 평소에는 메모리 주소만 가지고 있기 때문에 메모리를 낭비하지 않는다. 
    
    print( generator_list(50) )
    
    for a in generator_list(50):
        print(a, end=' ')
    # 반복문을 통한 출력
    
    print('\n', list(generator_list(50)))
    # list를 통한 출력
    ```

    ![image](https://user-images.githubusercontent.com/71866756/149901715-27d221aa-5deb-4655-988a-e83b4cfe0425.png)  

    ```python
    gen_ex = (n*n for n in range(500))
    # 튜플이 아닌 generator이다. 
    
    print( gen_ex )
    print( list(gen_ex) )
    ```

    ![image](https://user-images.githubusercontent.com/71866756/149901789-d67dcbdf-2f57-4675-87dc-019c438d0a40.png)

    

    

  - 가변인자 (variable length parameter)

    ```python
    def asterisk_test(a,b,*args):
        return a+b+sum(args)
    # 입력된 값은 튜플 타입으로 되어 있다. 
    
    print(asterisk_test(1,2,3,4,5))
    ```

    ![image](https://user-images.githubusercontent.com/71866756/149901836-db1b87a5-29fb-452f-8717-9113e314508a.png)

    ```python
    # 키워드 가변인자
    # dict type으로 값을 삽입
    def kwargs_test(**kwargs):
        print(kwargs)
    kwargs_test(first=1, second=2, third=3)
    ```

    ![image](https://user-images.githubusercontent.com/71866756/149901885-b3235ee1-cc55-475b-88eb-401bf067f218.png)  

    ```python
    def kwargs_test_1(one, two, *args, **kwargs):
        print(one+two+sum(args))
        print(args)
        print(kwargs)
    kwargs_test_1(3,4,5,6,7,8,first=3, second=4)
    # 순서대로 인자를 전달하는 것이 중요하다. 
    ```

    ![image](https://user-images.githubusercontent.com/71866756/149901925-274e63d2-0ba9-4b27-b669-c0a283112f76.png)

    

  - unpacking container

    ```python
    def asterisk_test(a, *args):
        print(a, *args)
        print(a, args)
        print(type(args))
    
    test=(3,4,5)
    asterisk_test(1, *test)
    # *은 unpacking의 의미로 보면 된다. 
    # test는 현재 튜플로 되어있으니 인자로 전달할 때, 풀어서 3,4,5 각각으로 전달
    # 함수의 인자는 *args로 튜플 형태로 데이터가 입력이 된다. 
    # 함수 내부에 *args를 print하니 튜플이 unpack되어 3,4,5가 출력되는 것이다. 
    ```

    ![image](https://user-images.githubusercontent.com/71866756/149901993-51c6a569-b92b-4070-9f2d-f1bb5a65a136.png)

    ```python
    data = [[1,2],[3,4]]
    print(*data)
    
    data_1 = ([1,2],[3,4])
    print(*data_1)
    # *은 Sequence data에 사용됨
    
    def asterisk_test(a, b, c, d,):
        print(a, b, c, d)
    data_2 = {'b':1, 'c':2, 'd':2}
    asterisk_test(10, **data_2)
    # **은 dict에 사용됨
    
    ex = ([1,2],[3,4],[5,6],[7,8],[9,10])
    print(ex)
    for value in zip(*ex):
        print(value)
    ```

    ![image](https://user-images.githubusercontent.com/71866756/149902034-eb0e02a6-1d90-4a72-9ecb-306027bfa679.png)

    

- **Variable**

  변수는 메모리 주소를 가지고 있고, 변수에 들어가는 값은 메모리 주소에 할당된다. 

  - 폰 노이만 아키텍처

    사용자가 컴퓨터에 값을 입력하거나 프로그램을 실행할 경우, 메모리에 저장 -> CPU가 정보 해석, 계산 -> 사용자에게 결과값 전달 

  - Dynamic Typing

    코드가 해석되는 순간에 데이터 타입이 결정이 된다. 

  - 데이터 형변환

    ```python
    a=float(a)
    print( type(a) )	# float
    
    a=int(a)	# 내림처리
    print( a )
    
    a="76.3"
    a=float(a)
    print( a )	# 76.3
    ```

- **List**

  - 주소 (offset)

    ```python
    direction = ['북', '북동', '동', '남동', '남', '남서', '서', '북서']
    print(direction[0:-1:2])	# start idx, end idx, step => 북, 동, ...
    print(direction[::-1])		# 역순=>북서, 서....
    ```

  - join

    ```python
    direction = ['북', '북동', '동', '남동', '남', '남서', '서', '북서']
    result = ''.join(direction)
    # 북북동동남동남남서서북서
    ```

  - in

    ```python
    direction = ['북', '북동', '동', '남동', '남', '남서', '서', '북서']
    print( '북' in direction )
    ```

  - append, extend

    ```python
    color = ['빨','주']
    print( color.append('노') ) # 빨,주,노, 원래 변수 color에는 추가 x
    
    color.extend('노')
    print( color )	# 빨, 주, 노, 원래 color 변수에 추가
    ```

  - remove, del

    list 원소 삭제

    ```python
    color.remove('노')
    del color[0]
    ```

  - sort

    ```python
    a = [5,4,3,2,1]
    a.sort()
    print( a )	# 1,2,3,4,5
    ```

  - unpacking

    ```python
    t = [1,2,3]
    a,b,c = t
    print( a,b,c )	# 1,2,3
    ```

  - 복사

    ```python
    import copy
    test_copy = copy.deepcopy(test)		
    # 1차원의 경우는 a_copy = a[:]로 깊은 복사가 가능
    # 2차원부터는 copy 모듈을 사용
    ```

  

- **입출력**

  - 출력

    ```python
    print("{0} : {1:5.2f}".format("cost", 15.032))
    # cost :      15.03
    # 5칸 띄우고, 소수점 2자리까지 출력
    ```

  - 최근 출력

    ```python
    print(f"{name}, {age}")	# 변수명 바로 적는다.
    print(f'{name:20}')		# 20칸 지정, 왼쪽 정렬
    print(f'{name:>20}')	# 20칸 지정, 우측 정렬
    print(f'{name:*<20}')	# 왼쪽 정렬, 빈 칸 모두 * 출력
    print(f'{name:*>20}')	# 오른쪽 정렬, 빈 칸 모두 * 출력
    print(f'{name:*^20}')	# 중앙 정렬, 빈 칸 모두 * 출력
    print(f'{number:.2f}')	# 소수점 2자리까지 출력
    ```

- **비교 연산**

  - is

    ```python
    a = 100
    b = 100
    print( a==b )		# True
    print( a is b )		# True
    
    a = 300
    b = 300
    print( a==b )		# True
    print( a is b)		# False
    # is 연산의 경우 메모리가 같은 경우 True, else False이다. 
    # 예전 파이썬의 경우 속도를 높이기 위해서 -5~256까지의 수를 일정 메모리에 할당했었다. 따라서 100일 경우는 a와 b가 같은 공간을 가르키지만, 300의 경우 새로운 메모리가 할당되어 둘의 출력값이 다르다. 
    ```

  - all, any

    ```python
    bool_list = [True, False, True]
    print( all(bool_list) )		# False, 모두 True이면 True
    print( any(bool_list) )		# True, 하나라도 True이면 True
    ```

  

- **반복문**

  - in

    ```python
    for looper in [1,2,3,4,5]:
    	print( looper )		# 1,2,3,4,5
    ```

  
  
  
- **문자열 (string)**

  - slicing

    ```python
    text = "hello my name is"
    print( text[-9:] )	# -9부터 끝까지
    ```

  - 문자열 함수

    ```python
    text= "My name is 1"
    print(len(text))				# 길이
    print(text.upper())				# 대문자로 변환
    print(text.lower())				# 소문자로 변환
    print(text.capitalize())		# 첫글자만 대문자
    print(text.title())				# 첫글자, 띄어쓰기 후 대문자
    print(text.count('abc'))		# 'abc' 개수 count
    print(text.find('abc'))			# 'abc'가 있으면 True
    print(text.rfind('abc'))		# 'abc'가 있으면 True
    print(text.startswith('My'))	# 'My'로 시작하면 True
    print(text.endswith('is'))		# 'is'로 끝나면 True
    print(text.split())				# 띄어쓰기 단위로 나눠 list생성
    print(text.isdigit())			# 숫자이면 True
    print(text.count("My"))			# 문자열 개수 세기
    
    ```

    - 결과 

    ![image](https://user-images.githubusercontent.com/71866756/149902209-9a521499-bd87-4934-977f-90a623399c5f.png)

  - strip

    ```python
    text = "https://naver.com"
    print( text.strip('hmoc./:tps'))	# 선,후행 문자에서 문자제거
    print( text.lstrip('https://'))		# 선행 문자에서 문자제거
    print( text.rstrip('.com'))			# 후행 문자에서 문자제거
    ```

    ![image](https://user-images.githubusercontent.com/71866756/149902246-e484f090-48ea-4804-8f61-995d2a0bdbf2.png)

- **함수**

  - function type hints

    ```python
    def func(index : int)->None:
    	'''
    	함수 내용
    	'''
    	
    # type hints는 사용자가 함수를 사용하기 쉽게 힌트를 주는 역할이다. 
    # 변수 명 : 변수 타입으로 넣어주고, ->를 통해 return type이 뭔지 알려준다. 
    ```

  - docstring

    ```python
    def funcEx(x : int, y : int)-> int:
        """[summary]
    
        Args:
            x (int): [description]
            y (int): [description]
    
        Returns:
            int: [description]
        """
        return x*y
    # summary : 함수 목적
    # docstring을 적는 습관이 중요!
    ```

  

- **data structure**

  - 스택 (stack)

    Last in First Out (LIFO)

    입력 : push, 출력 : pop

    ```python
    a=[1,2,3,4,5]
    a.append(10)
    print(a)
    a.pop()
    print(a)
    ```

    ![image](https://user-images.githubusercontent.com/71866756/149902298-db52343d-b4f6-4b4e-a15e-2524fa8794a7.png)

  - 큐 (queue)

    ```python
    a = [1,2,3,4,5]
    a.append(10)
    print(a)
    a.pop(0)			# 0번째 원소 pop
    print(a)
    ```

    ![image](https://user-images.githubusercontent.com/71866756/149902348-ae731ae4-d550-41c5-87a1-478c7288f530.png)

  - 튜플 (tuple)

    변경 불가능한 list라 생각, 리스트의 연산, 인덱싱, 슬라이싱은 동일하게 사용

    ```python
    a = (1,2,3,4,5)
    print(a)
    a=a*2
    print(a)
    a[0]=2			# error, 튜플은 값의 변경이 불가능하다. 
    print(a)
    
    # 값이 하나인 Tuple은 ,를 붙여야 한다. 
    b = (1)			# 정수로 인식
    b = (1,)		# 튜플로 인식
    ```

    ![image](https://user-images.githubusercontent.com/71866756/149902400-bc915a76-a0b1-4fd1-814c-606371dc725a.png)

  - 세트 (set)

    ```python
    s = set([1,2,3,3])		# set 선언 및 초기화
    print(s)
    
    s = {1,2,3,3}			# set 선언 및 초기화
    print(s)
    
    s.add(5)				# set에 원소 추가
    print(s)
    
    s.remove(1)				# set의 원소 삭제
    print(s)
    
    s.update([6,7,8])		# set에 원소 여러개 추가
    print(s)
    
    s.discard(6)			# 6 삭제
    print(s)
    
    s.clear()				# set 모두 삭제
    print(s)
    
    s1 = set([1,3])			
    s2 = set([3,4])
    s3 = s1.union(s2)			# 합집합 (s1|s2)
    print(s3)
    
    s3 = s1.intersection(s2)	# 교집합 (s1&s2)
    print(s3)
    
    s3 = s1.difference(s2)		# 차집합 (s1-s2)
    print(s3)
    ```

    ![image](https://user-images.githubusercontent.com/71866756/149902447-6a651388-6807-4d9e-9b72-2743aae45a79.png)

  - 사전 (dictionary)

    key, value로 구성

    ```python
    code = {1:"hi", 2:"bye"}
    print(code.keys())
    print(code.values())
    print(code.items())
    print(code[1])
    ```

    ![image](https://user-images.githubusercontent.com/71866756/149902499-6c3a47e5-dc87-4d2b-bf6e-5b961b50e4e7.png)

    ```python
    import csv
    
    
    def getKey(item):  # 정렬을 위한 함수
        return item[1]  # 신경 쓸 필요 없음
    
    
    command_data = []  # 파일 읽어오기
    with open("command_data.csv", "r", encoding="utf8") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=",", quotechar='"')
        for row in spamreader:
            command_data.append(row)
    
    command_counter = {}  # dict 생성, 아이디를 key값, 입력줄수를 value값
    
    for data in command_data:  # list 데이터를 dict로 변경
        if data[1] in command_counter.keys():  # 아이디가 이미 Key값으로 변경되었을 때
            command_counter[data[1]] += 1  # 기존 출현한 아이디
        else:
            command_counter[data[1]] = 1  # 처음 나온 아이디
    
    dictlist = []  # dict를 list로 변경
    for key, value in command_counter.items():
        temp = [key, value]
        dictlist.append(temp)
    sorted_dict = sorted(dictlist, key=getKey, reverse=True)  # list를 입력 줄 수로 정렬
    print(sorted_dict[:100])
    ```

    

  - collections 모듈

    ```python
    from collections import deque			
    # linked list의 특성을 지원함 (rotate, reverse)
    # 리스트보다 훨씬 빠르게 동작한다. 
    
    
    
    from collections import Counter
    # Sequence type의 data element들의 개수를 dict 형태로 반환
    # Set의 연산들을 지원함
    
    c = Counter()
    c = Counter('gallahand')
    print(c)	# {'a':3, 'd':1, ...}
    
    c = Counter({'a':2, 'b'=1})
    print(list(c.elements()))	# ['a','a','b']
    
    
    
    from collections import OrderedDict
    # 원래는 dict이 정렬이 안되지만, 현재 python 3.6부터는 정렬이 되어서 필요가 없다. 
    
    
    
    from collections import defaultdict
    # key값이 없는 경우에는 default값을 설정해서 value로 넣음
    d = defaultdict(lambda : 0)
    print(d['first'])
    
    def default_value():
        return 0
    d = defaultdict(default_value)
    print(d['second'])
    
    
    
    from collections import namedtuple
    # tuple 형태로 data 구조체를 저장
    
    Coordinate = namedtuple('Coordinate',['y','z'])
    c = Coordinate(y=1,z=2)
    print(c.y, c[0])
    ```

    

