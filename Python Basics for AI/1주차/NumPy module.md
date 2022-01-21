# NumPy Module

- **array의 생성**

  - np.array()

    ```python
    import numpy as np
    
    sample = np.array([1,2,3,4], float)
    # data type은 한 종류로
    ```

  - shape

    ```python
    import numpy as np
    
    sample = np.array([1,2,3,4], float)
    print(sample.shape)	# (4,)
    
    sample = np.array([[1,2],[3,4]], float)
    print(sample.shape)	# (2,2)
    ```

  - ndim

    ```python
    import numpy as np
    
    sample = np.array([[1,2],[3,4]], float)
    print(sample.ndim)
    ```

  
  - arange()
  
    ```python
    import numpy as np
    
    sample = np.arange(0, 5, 0.5)	# 시작, 끝, step
    print( sample )		# 0, 0.5, 1, ..., 4.5
    
    sample = np.arange(10)
    # 0~9까지 생성
    print( sample )
    ```
  
  - zeros, ones, empty, ones_like
  
    ```python
    import numpy as np
    
    sample = np.zeros((3,4))	# shape
    print( sample )
    sample = np.ones((3,4))		# shape
    print( sample )
    sample = np.empty((3,4))
    print( sample )
    sample1 = np.ones_like(sample)
    print( sample1 )
    ```
  
  - identity(), eye()
  
    ```python
    import numpy as np
    
    sample = np.identity(n=3, dtype=np.int8)
    print(sample)
    # 대각성분이 1인 정방행렬 생성
    
    sample = np.eye(3,5,k=2)
    print( sample )
    # 시작 행을 적어줄 수 있으며, 정방행렬이 아니여도 된다. 
    ```
  
    ![image](https://user-images.githubusercontent.com/71866756/150507615-8069701c-04f8-4002-bd90-4d45a3ac3bfc.png)
  
  - diag()
  
    ```python
    import numpy as np
    
    sample = np.eye(3,5,k=2)
    print( sample )
    
    print(np.diag(sample, k=2))
    # k=2지점의 대각성분을 추출함
    ```
  
  - random.uniform()
  
    ```python
    import numpy as np
    
    sample = np.random.uniform(0,1,10).reshape(2,5)	# 시작, 끝, 개수
    print( sample )
    ```
  
    
  
- **array transformation**

  - reshape

    ```python
    import numpy as np
    
    sample = np.array([1,2,3,4], float)
    print(sample.reshape(2,-1))	# (2,2)배열
    ```

  - flatten

    ```python
    import numpy as np
    
    sample = np.array([[1,2],[3,4]], float)
    print( sample.flatten() )	# 1,2,3,4
    ```

  - T (transpose)

    ```python
    import numpy as np
    
    sample = np.array([[1,2],[3,4]], float)
    print( sample.T )
    print( sample.transpose() )
    ```

    

- **Indexing & slicing**

  - 행과 열 부분 나눠서 slicing (python list는 불가능)

    ```python
    import numpy as np
    
    sample = np.array([[1,2],[3,4]], float)
    print( sample[:, :1] )	# [[1],[3]
    ```

  

- **연산**

  - sum, std, mean, sqrt, exp

    ```python
    import numpy as np
    
    sample = np.ones((2,3))
    print( sample.sum() )
    print( sample.sum(axis=1))
    print( sample.std() )
    print( sample.mean() )
    
    sample = np.array([2,3])
    print( np.sqrt(sample) )
    ```

    ![image](https://user-images.githubusercontent.com/71866756/150507818-f651d9ea-3673-431f-9319-2470bf6efbe7.png)

  - dot()

    ```python
    import numpy as np
    
    sample = np.random.uniform(0,1,6).reshape(2,3)
    sample2 = np.random.uniform(0,1,6).reshape(3,2)
    print( sample.dot(sample2) )
    ```

    

- **concatenate**

  - vstack(), hstack(), concatenate

    ```
    import numpy as np
    
    a = np.array([1,2,3])
    b = np.array([4,5,6])
    
    print( np.vstack((a,b)) )
    print( np.hstack((a,b)) )
    print( np.concatenate((a,b), axis=0) )
    # 현재 축이 열밖에 없어서 열이 axis=0
    ```

    

  ![image](https://user-images.githubusercontent.com/71866756/150507935-54e71548-be5b-435d-9502-0c46c6a9fcf6.png)

  - newaxis

    ```python
    import numpy as np
    
    a = np.array([1,2])
    print( a[:, np.newaxis] )
    print( "newaxis로 행 생성 : ",a[np.newaxis, :] )
    ```

    ![image](https://user-images.githubusercontent.com/71866756/150508003-ca359d20-ac3f-4a2a-baf8-d295e991f024.png)
