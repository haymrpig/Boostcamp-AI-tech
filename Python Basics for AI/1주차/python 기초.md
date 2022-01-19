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
    color = ['빨강','주황']
    color.append('초록')
    print(color)
    # append는 하나로 인식하여 초록을 넣음
    # ['빨강', '주황', '초록']
    
    color.extend('노랑')
    print(color)
    # extend는 문자 하나씩 삽입
    # ['빨강', '주황', '초록', '노', '랑']
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
    # 원래는 dict이 삽입된 순서대로 저장이 되지 않았다. 그래서 OrderedDict을 통해 삽입된 순서를 보장받았는데, python 3.6 이후부터는 일반 dict도 순서를 보장받아서 잘 쓰지 않는 모듈이다.  
    
    
    
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




- **클래스와 객체**

  `객체` : 속성 (Attribute, 변수)와 행동 (Action, 함수)을 가짐

  `클래스` : 붕어빵틀

  `인스턴스` : 붕어빵, 메모리에 저장되는 실제 구현체

  `상속` : 부모클래스로부터 속성과 method를 물려받은 자식 클래스를 생성하는 것

  `다형성 (polymorphism)` : 같은 이름의 메소드의 내부 로직을 다르게 작성하는 것

   `가시성 (visibility)` : 객체의 정보를 볼 수 있는 레벨을 조절하는 것, 소스의 보호

  `캡슐화/정보은닉 (information hiding)` : class 설계 시, 클래스 간 간섭/정보공유 최소화

  

  - 일등함수 / 일급 객체 (first-class objects)

    변수나 데이터 구조에 할당이 가능한 객체

    parameter로 전달이 가능 + 리턴 값으로 사용가능

    파이썬의 함수는 모두 일급함수이다.

    ex) map(func, sample)에서 func는 함수로 map의 parameter로 전달이 가능하다. 

    ```python
    def square(x):
        return x*x
    
    f = square
    
    print(f(5))		# 25 반환
    ```

    

  - inner function

    함수를 return함으로 비슷한 목적을 가진 함수들을 하나로 만들 수 있다. (parameter만 조정하여)

    ```python
    def print_msg(msg):
        def printer():
            print(msg)
        return printer
    
    another = print_msg("hello")
    another()
    # hello 출력
    ```

    

  - decorator

    ```python
    def star(func):
        def inner(*args, **kwargs):
            print(args[1]*30)
            func(*args, **kwargs)
            print(args[1]*30)
        return inner
    
    def percent(func):
        def inner(*args, **kwargs):
            print(args[2]*30)
            func(*args, **kwargs)
            print(args[2]*30)
        return inner
    
    @star
    @percent
    def printer(msg, *args):
        print(msg)
              
    printer("Hello", "*", "%")
    
    # *************************
    # %%%%%%%%%%%%%%%%%%%%%%%%%
    # Hello
    # %%%%%%%%%%%%%%%%%%%%%%%%%
    # *************************
    # 위 예시처럼 decorator를 통해 앞뒤로 꾸며줄 수가 있다. 
    ```



- **모듈, 패키지**

  `pycache` : 폴더를 로딩할 때(import), 더 빠르게 로드하기 위해 미리 컴파일 된 파일

  - Alias

    ```python
    import numpy as np
    ```

  - 특정 함수, 클래스 호출

    ```python
    from PIL import Image
    ```

  - 모듈에서 모든 함수 또는 클래스 호출하기

    ```python
    from numpy import *
    ```

  - built-in module

    `random` : 난수 생성하는 모듈

    `time` : 시간 관련 모듈



- **예외처리 (Exception Handling)**

  - try, except, else, finally

    ```python
    a=[1,2,3]
    for i in range(10):
        try:
            print(f'i : {i}, answer : {10//i}')
            print(a[i])
        except ZeroDivisionError as err:
            print(err)
        
        except IndexError as err:
            print(err)
        
        # 아래는 권장되지 않는다. 만약 에러가 발생하였을 시 위치를 찾기 어렵기 때문에
        except Exception as err:
            print(err)
        
        # 에러가 없을 시 출력됨
        else:
            print("else")
        
        # 항상 출력됨
        finally:
            print("final")
    ```

    

  ![image](https://user-images.githubusercontent.com/71866756/150141630-edf1d1ae-cf5a-4480-9382-280e249db82c.png)

  - raise

    ```python
    sample = [1,2,'a',3,4,5]
    
    for num in sample:
        if num not in [1,2,3,4,5]:
            raise ValueError("숫자가 아닙니다.")
        print(f'숫자는 {num}')
    # 강제로 error 발생시키기
    ```

    ![image](https://user-images.githubusercontent.com/71866756/150141682-41e207ce-9d02-4d08-9910-59fa98dbd9a4.png)

    

  - assert

    ```python
    sample = [1,2,'a',3,4,5]
    
    for num in sample:
        assert isinstance(num, int)
    # 조건에 맞지 않으면, 즉 False이면 AssertionError를 띄운다. 
    ```

    ![image](https://user-images.githubusercontent.com/71866756/150141742-f21fd985-a157-49a4-a608-7c4812371ada.png)

  

- **파일**

  `binary file` : 컴퓨터만 이해할 수 있는 형태인 이진 형식으로 저장된 파일 (엑셀, 워드 등등)

  `text file` : 문자열 형식(ASCII, Unicode)으로 저장된 파일 (파이썬 코드 파일, HTML 파일 등등)

  - open, close

    ```python
    f = open("hi.txt", "r")
    contents = f.read()
    print(contents)
    f.close()
    ```

  

  - with open as

    ```python
    with open("hi.txt", "r") as my_file:
        contents = my_file.read()
        print(type(contents), contents)
        # type : str
    print(contents.split())
    ```

  

  - readlines()

    ```python
    # 한 줄씩 읽어와서 list형태로 저장
    # 메모리가 충분한 경우
    with open("hi.txt", "r") as my_file:
        contents = my_file.readlines()
    print(contents, contents[0])
    ```

  

  - readline()

    ```python
    # 한 줄씩 읽어옴
    # 메모리가 부족한 경우
    with open("hi.txt", "r") as my_file:
        i = 0
        while True:
            line = my_file.readline()
            if not line:
                break
            print( str(i) + " === " + line.replace("\n",""))
            i += 1
    # 반복문을 통해 파일의 끝까지 읽는다. 
    # 한줄씩 읽을 경우, 마지막 개행문자까지 저장이 되기 때문에
    # replace문을 이용하여 개행문자 제거
    ```

  

  - file write (새로 쓰기)

    ```python
    with open("hi1.txt", "w", encoding="utf8") as file:
        data = "hello my name is {}\n".format("hong")
        file.write(data)
    ```

  

  - file write (이어서 쓰기)

    ```python
    with open("hi1.txt", mode="a", encoding="utf8") as file:
        data = "hello my name is {}\n".format("Lee")
        file.write(data)
    # 파일의 끝에 이어서 쓴다. 
    ```

  

  - 디렉토리 생성/확인

    ```python
    import os
    
    os.mkdir("log")
    # log라는 이름의 디렉토리 생성
    
    try:
        os.mkdir("log")
    except FileExistsError as e:
        print(e)
    # 폴더 존재 시 error 반환
    
    if os.path.exists("log"):
        print("file exists")
    ```

  

  - 파일 복사하기

    ```python
    import shutil
    
    source = "hi.txt"
    dest = os.path.join("log", "dest.txt")
    
    shutil.copy(source, dest)
    # 파일 복사
    ```

  

  - 객체 형식으로 파일 다루기

    ```python
    import pathlib
    pathlib.Path.cwd()
    # 현재 위치한 디렉토리 출력
    
    cwd = pathlib.Path.cwd()
    cwd.parent
    # 현재 위치한 디렉토리의 상위 폴더 출력
    
    list(cwd.parents)
    # 상위 폴더들을 list 형태로 출력
    # cwd = "/content/log/hong"이면
    # [PosixPath('/content/log'), PosixPath('/content'), PosixPath('/')] 출력
    
    list(cwd.glob('*'))
    # 현재 경로 내의 모든 폴더를 list형태로 반환
    # generator이기 때문에 list로 만듦
    ```

  

  - Pickle

    파이썬의 객체를 영속화 (persistence)하는 built-in 객체

    데이터, object 등 실행중 정보를 저장 -> 불러와서 사용

    pickle은 파이썬에 최적화된 binary file이다. 

    ```python
    import pickle
    f = open("list.pickle", "wb")
    test = [1,2,3,4,5]
    pickle.dump(test,f) 
    # file에 test를 저장
    f.close()
    
    f = open("list.pickle", "rb")
    test_pickle = pickle.load(f)
    print(test_pickle)
    f.close()
    ```

    위의 예에서의 list 뿐만이 아니라 class도 영속화 가능하다.

    ```python
    class Multiply():
        def __init__(self, multiplier):
            self.multiplier = multiplier
    
        def multiply(self, number):
            return number * self.multiplier
    multiply = Multiply(3)
    multiply.multiply(100)
    f = open("multiply_object.pickle", "wb")
    pickle.dump(multiply, f)
    f.close()
    
    f = open("multiply_object.pickle", "rb")
    multi = pickle.load(f)
    print(multi.multiply(100))
    f.close()
    ```

  

- **Logging**

  ```python
  import logging
  
  logger = logging.getLogger("main")
  logging.basicConfig(level=logging.DEBUG)
  logger.setLevel(logging.INFO)
  
  steam_handler = logging.FileHandler(
      "my.log", mode="w", encoding="utf8")
  logger.addHandler(steam_handler)
  
  # 개발 시점
  logging.debug("틀림")
  logging.info("확인")
  
  # 기본 설정은 여기부터 사용자가 정보를 확인할 수 있다. 
  # 운영 시점
  logging.warning("조심")
  logging.error("에러")
  logging.critical("망")
  ```



- **configparser file**

  - config 파일 읽기

    cofig 파일은 []로 이루어진 section과 나머지 key, value값으로 구성되어있다. 

    ```python
    import configparser
    
    config = configparser.ConfigParser()
    print(config.sections())
    # 읽은 파일이 없어 아무것도 출력되지 않는다. 
    
    config.read('example.cfg')
    print(config.sections())
    # example.cfg 파일에 있는 section 이름을 list로 가져온다. 
    
    for key in config['SectionOne']:
        value = config['SectionOne'][key]
        print("{0} : {1}".format(key,value))
    
    print(config['SectionTwo']["FavoriteColor"])
    
    ###########################################
    # 모든 section에 대해 key, value값을 가져오기#
    ###########################################
    import configparser
    
    config = configparser.ConfigParser()
    print(config.sections())
    
    config.read('example.cfg')
    print(config.sections())
    
    for section in config.sections():
        for key in config[section]:
            value = config[section][key]
            print("{0} : {1}".format(key,value))
    
    print(config['SectionTwo']["FavoriteColor"])
    ```

    

  ![image](https://user-images.githubusercontent.com/71866756/150141822-11ee997d-791d-4eb5-88d6-5623b8a961e3.png)  

  - config 파일 쓰기

    ```python
    import configparser
    
    config = configparser.ConfigParser()
    
    config['section1'] = {} # 섹션 만들기
    config['section1']['batchsize']= '128'
    config['section1']['epoch'] = '1000'
    
    config['section2'] = {}
    config['section2']['lr'] = '0.001'
    
    with open("example2.cfg", "w") as f:
        config.write(f)
    ```

    ![image](https://user-images.githubusercontent.com/71866756/150141879-ebddfdbd-d7ac-4d03-91c4-8911e8101722.png)



- **argparser**

  Console 창에서 프로그램 실행시 setting 정보를 저장하며 Command-line option이라고 부른다. 

  ```python
  import argparse
  
  parser = argparse.ArgumentParser(description='Sum two integers.')
  
  parser.add_argument(
      '-a', "--a_value", 
      dest="a", help="A integers", type=int,
      required=True)
  parser.add_argument(
      '-b', "--b_value", 
      dest="b", help="B integers", type=int,
      required=True)
  # 짧은 이름, 긴이름, 표시명, help 설명, argument type
  args = parser.parse_args()
  
  print(args)
  print(args.a)
  print(args.b)
  print(args.a + args.b)
  ```

  ![image](https://user-images.githubusercontent.com/71866756/150141931-72d2d244-e259-46db-a48c-78e2cb22733a.png)

  

- **Data handling**

  - CSV (Comma Separate Value)

    `csv` : 필드를 쉼표로 구분한 텍스트 파일, 엑셀 양식의 데이터를 프로그램에 상관없이 쓰기 위한 데이터 형식 cf) TSV (탭), SSV (빈칸)

    csv의 경우, file open으로 한줄씩 읽어오면서 split을 이용하여 처리할 수 있지만, 한글로 되어 있는 경우, 하나의 값 안에 ,가 있는 경우 따로 전처리를 해줘야해서 까다롭다. 

    그래서 python에서 제공하는 csv객체를 사용한다. 

    ```python
    # csv example
    import csv
    reader = csv.reader(f, 
    		delimiter=',',	 
    		quotechar='"',
    		quoting=csv.QUOTE_ALL
    		)
    # delimiter : 글자를 나누는 기준 (default = ,)
    # lineterminator : 줄 바꿈 기준 (default = \r\n)
    # quotechar : 문자열을 둘러싸는 신호 (default = ")
    # quoting : 데이터 나누는 기준이 quotechar에 의해 둘러싸인 레벨 (default = QUOTE_MINIMAL)
    ```

    

    ```python
    import csv
    
    seoung_nam_data = []
    header = []
    rownum = 0
    
    with open("korea_floating_population_data.csv","r", encoding="cp949") as p_file:
        # 한글이 utf8로 되어있지 않은 경우를 위해서 encoding을 cp949로 설정한다. 
        
        csv_data = csv.reader(p_file) 
        #csv 객체를 이용해서 csv_data 읽기
        
        for row in csv_data: #읽어온 데이터를 한 줄씩 처리
            if rownum == 0:
                header = row #첫 번째 줄은 데이터 필드로 따로 저장
                location = row[7]
               #“행정구역”필드 데이터 추출, 한글 처리로 유니코드 데이터를 cp949로 변환
                if location.find(u"성남시") != -1:
                    # u는 unicode의 약자
                    seoung_nam_data.append(row)
             #”행정구역” 데이터에 성남시가 들어가 있으면 seoung_nam_data List에 추가
                rownum +=1
    with open("seoung_nam_floating_population_data.csv","w", encoding="utf8") as s_p_file:
        writer = csv.writer(s_p_file, delimiter='\t', quotechar="'", quoting=csv.QUOTE_ALL)
        # csv.writer를 사용해서 csv 파일 만들기 delimiter 필드 구분자
        # quotechar는 필드 각 데이터는 묶는 문자, quoting는 묶는 범위
        writer.writerow(header) #제목 필드 파일에 쓰기
        for row in seoung_nam_data:
            writer.writerow(row) #seoung_nam_data에 있는 정보 list에 쓰기
    ```

    

  - Web

    World Wide Web (WWW), 줄여서 웹이라고 부른다.

    `HTML (Hyper Text Markup Language)` : 웹 상의 정보를 구조적으로 표현하기 위한 언어로, 제목, 단락, 링크 등의 요소를 표시하기 위해 Tag를 사용하고 트리 모양의 포함관계를 가진다. (모든 요소들은 <>로 둘러 쌓여 있다. )

    ex) <title> Hello, World </title> =>제목 요소, 값은 Hello, World

    

  - 정규식 (regular expression)

    정규식 연습장 (http://www.regexr.com/)

    ```python
    import re
    import urllib.request
    
    url = "https://goo.gl/U7mSQl"
    html = urllib.request.urlopen(url)
    html_contents = str(html.read())
    id_results = re.findall(r"([A-Za-z0-9]+\*\*\*)", html_contents)
    
    for result in id_results:
    	print(result)
    ```

  

  - XML

    TAG와 TAG 사이에 값이 표시되고, 구조적인 정보를 표현할 수 있다. 

    PC와 스마트폰 같은 이기종에 유용한 방식으로 사용되었다. 

    BeautifulSoup과 lxml을 많이 쓴다. 

    ```python
    from bs4 import BeautifulSoup
    
    with open("books.xml", "r", encoding="utf8") as books_file:
    	books_xml = books_file.read()
    	
    # xml 모듈을 사용하여 데이터를 분석한다.
    soup = BeautifulSoup(books_xml, "lxml")
    # author가 들어간 모든 element를 추출한다.
    for book_info in soup.find_all("author"):
    	print(book_info) 
    	# TAG DATA TAG 형식으로 나온다.
    	# <author>Carson</author>
    	print(book_info.get_text())
    	# Carson
    ```

    

  - JSON (JavaScript Object Notation)

    간결하고, 데이터 용량이 적다. 

    dict type과 유사하며, key:value로 되어있어 dict type과 호환이 가능하다. 

    ```python
    import json
    
    with open("json_example.json", "r", encoding="utf8") as f:
        contents = f.read()
        json_data = json.loads(contents)
        # dict type으로 읽어온다. 
        # write의 경우는 json.dump(contents, f)
        for employee in json_data["employees"]:
    	    print(employee)
            print(employee["lastname"])
            # 내부적으로 또 dict로 되어있다. 
    ```

    
