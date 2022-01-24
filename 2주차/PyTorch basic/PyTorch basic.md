# 목차

- [**Introduction to PyTorch**](#1.-introduction-to-pytorch)
- [**PyTorch operation**](#2.-pytorch-operation)
- [**PyTorch 프로젝트 구조 이해하기**](3.-pytorch-프로젝트-구조)

# 1. Introduction to PyTorch

## Tensorflow v.s PyTorch

`computational graph` : 연산의 과정을 그래프로 표현 한 것

**EX)** g = (x+y)*z

<img src="https://miro.medium.com/max/481/0*ohO11wTD8DCUMVR8" alt="Deep Neural Networks As Computational Graphs | by Tyler Elliot Bettilyon |  Teb&#39;s Lab | Medium" style="zoom:67%;" />

- **tensorflow** 

  Define and Run (그래프를 먼저 정의 -> 실행시점에 데이터 feed, 장점은 production&scalability)

- **pytorch** 

  Define by Run (실행을 하면서 그래프를 생성하는 방식, 장점은 debugging이 용이하다. 중간중간의 값을 확인할 수 있다. )

  - Numpy 구조를 가지는 Tensor 객체로 array 표현
  - 자동미분 (auto grad)을 지원하여 DL 연산을 지원
  - 다양한 형태의 DL을 지원하는 함수와 모델을 지원 (Dataset, Multi-GPU 등등 지원)



# 2. PyTorch operation

## Tensor 객체의 생성

- torch.FloatTensor()

  ```python
  import torch
  import numpy as np
  array = np.array([[2,3],[1,4]])
  sample = torch.FloatTensor(array)
  print( sample )
  print( sample.shape )
  print( sample.ndim )
  ```

- torch.tensor()

  ```python
  import torch
  import numpy as np
  array = np.array([[2,3],[1,4]])
  sample = torch.tensor(array)
  print( sample )
  print( sample.shape )
  print( sample.ndim )
  ```

- torch.from_numpy()

  ```python
  import torch
  import numpy as np
  array = np.array([[2,3],[1,4]])
  sample = torch.from_numpy(array)
  print( sample )
  print( sample.shape )
  print( sample.ndim )
  ```

  

## Tensor의 변형

- ones_like(), flatten(), numpy(), shape, dtype 등등 numpy에서 사용하는 것 대부분 사용 가능

  ```python
  torch.ones_like(sample)
  sample.flatten()
  sample.shape
  sample.size()
  sample.dtype
  sample.numpy()		# tensor를 numpy로
  ```

- view, reshape

  ```python
  sample = torch.rand(size=(2,3,2))	# 0~1의 값을 (2,3,2) shape으로 생성
  
  sample.view([-1, 6])		# (-1,6) 사이즈로 변경
  sample.reshape([-1, 6])
  ```

- fill_

  ```python
  sample = torch.rand((3,2))
  sample.fill_(1)			# tensor의 모든 원소를 1로 채운다. 
  ```

- squeeze, unsqueeze

  ```python
  # squeeze() : 차원 중 값이 1인 경우 차원 축소
  # unsqueeze() : 차원 확장
  
  sample = torch.rand((2,1,2))
  sample = sample.squeeze()	# 1차원 삭제
  sample.unsqueeze(0).shape	# 0번째 dimension에 1 차원 추가
  ```



## Tensor의 연산

- 기본적으로 +, -, *, / 사용가능

- 내적 (dot, mm, matmul, @)

  ```python
  s1 = torch.rand((1,2))
  s2 = torch.rand((1,2))
  
  print( s1.dot(s2) )		# 불가능, 1차원 tensor만 연산 가능
  print( s1.mm(s2) )		# 내적, 보통 dot보다 mm을 더 많이 쓴다.(broadcasting X) 
  print( s1.matmul(s2) )	# 내적 (broadcasting O)
  print( s1@s2 )			# 내적
  ```



## Tensor 관련 함수들

- torch.scatter_

  ```python
  # 지정한 위치에 값을 대입하는 함수
  sample = torch.arange(0, 10).view(2,5)
  index = torch.tensor([[0,1,2,2,1]])	# input과 같은 ndim을 가져야 한다. 
  
  output = torch.zeros((5,5), dtype=src.dtype)
  output = output.scatter_(dim=1, index, sample)	
  # 행 방향으로 값을 대입
  # 즉, 첫번째 열에서는 0번째 행에, 2번째 열에서는 1번째 행에, 3번째 열에서는 2번째 행에....값을 sample에서 순서대로 하나씩 가져와서 대입한다. 
  ```

- torch.swapdims

  ```python
  sample = torch.arange(0, 10).view(2,5)
  torch.swapdims(sample, 0, 1)
  # 0번째와 1번째 차원 swap
  ```

- torch.chunk

  ```python
  # chunk는 tensor를 토막낸다고 생각하면 된다. 
  sample = torch.arange(0,10).view(2,5)
  torch.chunk(sample, 2, 0)
  # 2는 2토막, 0은 dimension을 가르킨다. 
  # 즉, 행을 2개로 쪼개갰다는 의미이다. (만약 2로 나누어 떨어지지 않는 3행 같은 경우는 먼저 크게 자르고 나머지를 나머지 토막에 배치한다.)
  ```

- torch.normal

  ```python
  # 평균 5, 표준편차 1의 정규분포
  sample = torch.normal(5,1, size=(2,5))
  print( sample )
  ```

- torch.rand

  ```python
  # 0~1의 Uniform 분포
  sample = torch.rand(2,5)
  print( sample )
  ```

- torch.bernoulli

  ```python
  # 먼저 0~1의 확률을 가지는 tensor를 생성
  # 그 element의 확률로 bernoulli분포를 따르는 tensor를 생성
  sample = torch.rand(2,5)
  sample = torch.bernoulli(sample)
  print( sample )
  ```

### Pointwise Ops

- torch.abs

  ```python
  # 절대값
  sample = torch.normal(-2,2, (2,5))
  print( sample )
  print( sample.abs() )
  ```

- torch.add

  ```python
  # element wise 합
  sample = torch.randint(0, 10, (2,5))
  sample2 = torch.randint(0, 10, (2,5))
  print( sample, sample2 )
  print( sample.add(sample2) )
  ```

- torch.bitwise_and

  ```python
  # bit 단위 and
  sample = torch.randint(0,2, (2, 5), dtype=bool)
  sample2 = torch.randint(0,2,(2, 5), dtype=bool)
  print( sample )
  print( sample2 )
  print( torch.bitwise_and(sample, sample2) )
  ```

### Reduction ops

- argmax, argmin

  ```python
  # 최대값의 index 반환
  sample = torch.rand((1, 10))
  print( sample )
  print( sample.argmax() )
  ```

- max

  ```python
  sample = torch.rand((1, 10))
  print( sample )
  print( torch.max(sample, dim=1) )
  # 최대값과 index 둘 다 반환
  ```

- all, any

  ```python
  # 모두 True일 경우 True
  sample = torch.randint(0,2,(2,5),dtype=bool)
  print( sample )
  print( torch.all(sample, dim=0) )
  print( torch.all(sample, dim=1) )
  ```

### Comparison Ops

- argsort

  ```python
  # sort한 결과를 index로 반환
  sample = torch.randn(4,4)
  print( sample )
  print( torch.argsort(sample, dim=1) )
  ```

- eq

  ```python
  # 두 tensor의 원소가 같은지 각각 비교 (broadcasting도 가능)
  sample = torch.randint(0,5, (2,5))
  sample1 = torch.randint(0,5, (1,5))
  print( sample )
  print( sample1 )
  print( torch.eq(sample, sample1) )
  ```

- isfinite

  ```python
  # 유한수인지 boolean으로 반환
  sample = torch.randint(0,5, (2,5))
  print( torch.isfinite(sample) )
  ```

### Other Ops

- diagonal

  ```python
  # 대각성분을 뽑아내는 메소드
  # 3차원 tensor의 경우에도 dim을 조절하여 뽑아낼 수 있다. 
  sample = torch.randint(0, 10, (2,2))
  print( sample )
  print( torch.diagonal(sample, offset=0, dim1=0))
  ```

- flatten

  ```python
  # 1차원으로 축소
  sample = torch.tensor([[[1], [2]]])
  print( sample.flatten() )
  ```

- atleast_2d

  ```python
  # 최소 2차원 tensor로 반환
  sample = torch.tensor([1])
  print( torch.atleast_2d(sample) )
  
  sample = torch.tensor([[[1]]])
  print( torch.atleast_2d(sample) )
  ```

### BLAS and LAPACK ops

- eig

  ```python
  # 고유벡터와 고유값 구하기
  sample = torch.tensor([[1,2],[1,2]], dtype=float)
  e, v = torch.eig(sample)
  print( e )
  print( v )
  ```

- pinverse

  ```python
  # 무어 펜로즈, 유사역행렬 구하기
  sample = torch.tensor([[1,2,3],[4,5,6]], dtype=float)
  print( torch.pinverse(sample) )
  ```

- inverse

  ```python
  # 역행렬 구하기
  sample = torch.tensor([[1,2],[4,5]], dtype=float)
  print( torch.pinverse(sample) )
  print( torch.inverse(sample) )
  ```

  

## Tensor의 GPU 사용

tensor의 경우 GPU에 올려서 사용할 수 있다.

```python
device = 'cuda' if torch.cuda.is_available else 'cpu'

sample = sample.to(device)
# GPU 사용이 가능할 경우 'cuda' 아니면 'cpu'
```

 

## nn.functional

- softmax, argmax, one_hot

  ```python
  import torch
  import torch.nn.functional as F
  
  tensor = torch.FloatTensor([0.5, 0.5, 1])
  h_tensor = F.softmax(tensor, dim=0)
  max = tensor.argmax(dim=0)
  print( h_tensor )
  print( tensor.argmax(dim=0) )
  print( F.one_hot(max) )
  ```

- cartesian_prod

  ```python
  a = [1,2,3]
  b = [4,5]
  
  tensor_a = torch.tensor(a)
  tensor_b = torch.tensor(b)
  print( torch.cartesian_prod(tensor_a, tensor_b) )
  # 모든 가능한 조합을 2차원 tensor로 뱉음
  ```

- nn.Linear

  ```python
  # dense layer에 많이 사용한다. 
  import torch
  from torch import nn
  
  X = torch.Tensor([[1, 2],
                    [3, 4]])
  
  X = X.flatten()
  X = nn.Linear(X.shape[0], 10)(X).view(2,-1)
  print(X)
  ```

- nn.Identity

  ```python
  # 보통 Identity를 사용할 때는 if else문으로 작성하고 싶을 때이다. 
  # 만약 BN을 적용할지 안할지에 대한 코드를 작성할 때, BN=Identity로 놓게 되면 BN을 적용하지 않는 것과 같은 역할을 한다. 
  
  # 두번째 역할은 net에서 특정 layer를 지울 때 사용한다. 
  # net.classifier를 지우고 싶을 때는 단순히 net.classifier=Identity()로 놓으면 손쉽게 처리할 수 있다. 
  import torch
  from torch import nn
  
  X = torch.Tensor([[1, 2],
                    [3, 4]])
  
  X = nn.Identity()(X)
  print(X)
  ```

- nn.ModuleList

  ```python
  # 모듈을 리스트 형태로 만들어서 적용할 수 있다. 
  class mul(nn.Module):
  	def __init__(self, value):
  		super().__init__()
  		self.value = value
  		
  	def forward(self, x):
  		return x * self.value
  		
  class Cal(nn.Module):
  	def __init__(self, value):
  		super().__init__()
  		self.layer = nn.ModuleList(mul(1), mul(2), mul(3))
  	
  	def forward(self, x):
  		for i in self.layer:
  			x = i(x)
  		return x
      
  x = torch.tensor([1])
  
  calculate = Cal()
  output = calculate(x)
  print( output )			# 1*1*2*3 = 6
  ```

- nn.ModuleDict

  ```python
  # 모듈을 dict형태로 만들어서 적용할 수 있다. 
  class mul(nn.Module):
  	def __init__(self, value):
  		super().__init__()
  		self.value = value
  		
  	def forward(self, x):
  		return x * self.value
  		
  class Cal(nn.Module):
  	def __init__(self, value):
  		super().__init__()
  		self.layer = nn.ModuleDict('mul1' : mul(1), 'mul2' : mul(2), 'mul3' :mul(3))
  	
  	def forward(self, x):
  		for key in self.layer.keys():
  			x = self.layer[key](x)
  		return x
      
  x = torch.tensor([1])
  
  calculate = Cal()
  output = calculate(x)
  print( output )			# 1*1*2*3 = 6
  ```

### Buffer

- register_buffer

  ```python
  # parameter는 아니지만, 일반적인 Tensor를 기록하여 model.state_dict()으로 보고 싶다면, register_buffer를 이용한다.
  # gradient가 계산되지 않고, 값이 업데이트 되지 않지만, 모델 저장시에 값이 같이 저장된다. 
  
  import torch
  from torch import nn
  from torch.nn.parameter import Parameter
  
  class Model(nn.Module):
      def __init__(self):
          super().__init__()
  
          self.parameter = Parameter(torch.Tensor([1]))
          self.tensor = torch.Tensor([1])
  
          self.register_buffer('buffer', self.tensor)
  		# 버퍼에 tensor 등록하기
          
  model = Model()
  
  print( model.state_dict() )
  print( model.get_buffer('buffer') )
  ```

- model.named_modules()

  ```python
  # model의 모듈 이름과 모듈 그 자체를 모두 반환
  for name, module in model.named_modules():
      print(f"[ Name ] : {name}\n[ Module ]\n{module}")
      print('\n')
  ```

- model.named_children()

  ```python
  # model의 모듈의 바로 아래 모듈을 return한다. 
  # 한 단계 아래의 submodule까지만 표시
  for name, child in model.named_children():
      print(f"[ Name ] : {name}\n[ Children ]\n{child}")
      print('\n')
  ```

- model.named_buffers()

  ```python
  # model에 속하는 buffer 전체 목록 가져오기
  for name, buffer in model.named_buffers():
      print(f"[ Name ] : {name}\n[ Buffer ] : {buffer}")
  	print('\n')
  ```

  

- nn.Module.get_submodule

  ```python
  # model에서 ab.a라는 이름의 module 가져오기
  submodule = nn.Module.get_submodule(model, "hi.h")
  
  print( submodule )
  print( model.ab.a )
  ```

- nn.Module.get_buffer

  ```python
  # model에서 cd.c이름의 duck이름의 버퍼 가져오기
  buffer = nn.Module.get_buffer(model, "hi.h.buf")
  print( buffer )
  ```

- model.__ doc __

  ```python
  # docstring을 볼 수 있는 코드
  print( model.__doc__)
  ```

  

## Auto grad 

- backward()

  ```python
  w = torch.tensor(2.0, requires_grad=True)
  y = w**2
  z = 10*y + 25
  z.backward()	# autograd
  w.grad
  
  # a와 b 편미분
  # a와 b의 값이 각각 2개니깐 grad도 2개씩 나와야 하므로 external_grad로 사이즈를 정해준다. 
  a = torch.tensor([2., 3.], requires_grad=True)
  b = torch.tensor([6., 4.], requires_grad=True)
  Q = 3*a**3 - b**2
  external_grad = torch.tensor([1., 1.])
  Q.backward(gradient=external_grad)
  a.grad
  b.grad
  ```



# 3. PyTorch 프로젝트 구조

**실행, 데이터, 모델, 설정, 로깅, 지표, 유틸리티 등 다양한 모듈들을 분리하여 프로젝트 템플릿화!!!**

- PyTorch Template 추천 repo
  - https://github.com/FrancescoSaverioZuppichini/PyTorch-Deep-Learning-Template
  - https://github.com/PyTorchLightning/deep-learning-project-template
  - https://github.com/victoresque/pytorch-template

- **colab에 원격 접속하기**

  - **ngroc 회원가입하기 (https://ngrok.com/)**

    ![image](https://user-images.githubusercontent.com/71866756/150808357-08255cc3-5847-46e7-9c68-3638b9da9d2f.png)

    -> 아래 토큰 복사 후, colab 접속

  - **colab에서 토큰 붙여넣기 & 비밀번호 설정 및 colab-ssh 설치**

    ![image](https://user-images.githubusercontent.com/71866756/150808415-18dbf641-4687-4d6f-85e6-9c8420f1ccf3.png)

  - **ssh launch하기**

    ![image](https://user-images.githubusercontent.com/71866756/150808471-08a9f65f-5992-4628-816b-311e3500c629.png)

  - **local에서 vscode 실행 후, remote ssh 설치**

    ![image](https://user-images.githubusercontent.com/71866756/150808523-b66ec4fc-89a8-4ee3-a209-d5f64a1c2225.png)

  - **ctrl+shift+p로 Remote-SSH:Add 검색**

    ![image](https://user-images.githubusercontent.com/71866756/150808562-19d69bad-38ae-485a-a044-70e26bf4aba3.png)

    -> Add New SSH Host 선택

    ![image](https://user-images.githubusercontent.com/71866756/150808615-88a3dfda-bf93-4bc4-9221-b6cb8091beb0.png)

    -> ssh root@ 이후 colab에서 .io까지 복사 + -p (port번호) 복사

    ![image](https://user-images.githubusercontent.com/71866756/150808700-c6190bd5-6d48-478a-be86-47a6e93fae85.png)

    -> Open Config 선택

  - **ctrl+shift+p remote-ssh connect to Host**

    ![image](https://user-images.githubusercontent.com/71866756/150808770-e19ddb56-59ba-411e-a951-6f60ea06c361.png)

    -> enter

    ![image](https://user-images.githubusercontent.com/71866756/150808823-ae6aab32-5fdc-482c-82ff-7779decfb796.png)

    -> enter 후 password 입력하면 접속이 된다. 

  - **접속 됐는지 확인하기**

    새 터미널을 open하여 확인, 좌측 탐색기에서 /content로 폴더 열기 -> 완료!



# 질문사항

- auto grad는 어떤 방식으로 동작하는가?

