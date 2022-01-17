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

    