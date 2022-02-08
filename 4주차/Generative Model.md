# 목차

- **Generative Model**
- **Auto Regressive Model**
- **Latent Variable Model**



# 1. Generative model

### 1-1. Generative model 개요

generative model은 단순히 새로운 이미지를 만들어내는 것이 아니다. 

generative model로 할 수 있는 것의 예시는 아래와 같다. 

- **Implicit models**

  `Generation (implicit models)` : 만약 강아지 사진 x에 대한 분포 p(x)를 학습했다면, p(x)에서 sampling한 새로운 x 또한 강아지와 비슷할 것이다. 

- **Explicit models**

  `Density estimation (explicit models)` : 만약 새로운 x가 강아지처럼 생겼다면 p(x)는 굉장히 높은 값을 가르킬 것이다. 이를 통해 할 수 있는 것은, 분포를 통해 강아지와 강아지가 아닌 것을 구분해 낼 수 있다는 의미가 된다. 

  > 흔히, 이러한 것을 anomaly detection이라고 하며, anomaly detection이란 normal (정상) sample과 abnormal (비정상) sample을 구분해내는 것이다. 

- **Feature learning**

  이미지들의 공통 분모를 찾을 수 있다. 



### 1-2. Generative model 이해를 위한 확률론

`Bernoulli distribution` : 간단하게 동전 던지기를 예로 들 수 있다. 

`Categorical distribution` : 간단하게 주사위 던지기를 예로 들 수 있다. 

> 위 두 분포의 차이는 확률에 있다. 
>
> 베르누이 분포는 하나의 확률 (p)을 알면 다른 확률 (1-p)를 알 수 있다. 

위 분포를 알고 있다고 가정하고 예시를 하나 들어보자. 

**EX1) RGB joint distribution을 modeling한다고 가정해보자**

이 때, 가능한 경우의 수는 256x256x256일 것이다. 

그렇다면 필요한 parameter의 수는 몇개인가?

256x256x256 - 1이다. 

(-1을 하는 이유는 나머지들을 알면 자동으로 마지막 하나는 알 수 있기 때문이다. )

**EX2) 이번에는 binary 이미지를 생각해보자**

가능한 경우의 수는 2^n일 것이다. 

parameter의 수는 2^n-1일 것이다. 

- **Structure Through Independence**  
  ![image](https://user-images.githubusercontent.com/71866756/152985170-e81b62af-5c67-4571-b67d-d588b220becb.png)  

  > 독립이라고 가정할 경우, parameter가 급격하게 줄어들지만, 이는 현실과는 너무 다르다. 
  >
  > 조금 더 현실과 맞는 가정을 해야 한다!

- **Conditional Independence**

  먼저 세가지 기본적인 법칙에 대해서 알아보자.   
  ![image](https://user-images.githubusercontent.com/71866756/152985231-daafdae1-63d9-43e6-a1b5-80aea06fd3b8.png)  
  그렇다면 위 수식을 이용하여 좀 더 현실적인 가정을 할 수 있지 않을까?

  먼저, Chain Rule을 이용한다. 

  베르누이 분포에서 Chain Rule을 생각해보고, 필요한 parameter의 수를 적어보자.   
  ![image](https://user-images.githubusercontent.com/71866756/152985273-c6911844-1e6a-42d9-aac3-71b7bc36103a.png)  
  Chain Rule은 독립이든 독립이 아니든 성립하는 식이기 때문에 parameter의 변화는 없다....

  그렇다면, 여기에 Markov assumption을 추가해보자.

  Markov assumption이란 현재 상태는 오직 이전 상태의 영향만 받는다는 가정이다. 

  그렇다면 Conditional Independence를 이용해서 식을 좀 더 단순화해보자.   
  ![image](https://user-images.githubusercontent.com/71866756/152985301-ad348b9f-4aa5-4e51-94e9-67b97dd604aa.png)  
  parameter가 매우 적어진 것을 확인할 수 있다!!!!

  따라서, **Auto-regressive models**에서는 위 가정을 이용하여 모델링한다. 



# 2. Auto-regressive Model

사실 Auto-regressive model은 바로 이전 값에 dependent한 경우도 있고, 현재 값보다 이전인 모든 값에 dependent한 경우가 있다. 

이를 AR1, ARN으로 나누고 이 경우들은 모두 Chain Rule과 Conditional independence를 이용한다. 

### 2-1. NADE (Neural Autoregressive Density Estimator)

NADE는 ARN으로 현재값이 이전 값들에 모두 dependent하다고 모델링한다. 

또한 NADE는 explicit model로 결과를 확률로써 나타낼 수 있다.

![image](https://user-images.githubusercontent.com/71866756/152985352-877798d3-1794-40db-be72-d91125bdc684.png)  
![image](https://user-images.githubusercontent.com/71866756/152985370-c5eb91e9-4426-409b-84c6-0174705aa6ce.png)  
위 식으로 나타낼 수 있다. 

그리고 위 수식을 Chain Rule과 Conditional distribution을 고려하면 아래와 같다.  
![image](https://user-images.githubusercontent.com/71866756/152985403-89623863-fce1-4a98-b7a6-2fa75c68dce9.png)  

> 즉, 1번째 pixel의 확률분포를 알고 있다면 2번째 pixel의 확률분포를 알고,
>
>  1,2번째 pixel의 확률분포를 알면 3번째 pixel의 확률 분포를 알기 때문에 대입하여 계산한다.   

즉, 단순히 generation 뿐만 아니라, 어떤 input에 대한 확률을 계산을 할 수 있는 explicit model이 되는 것이다. 

이 예시에서는 discrete한 random variable이지만, 연속인 경우, 마지막 layer에 Gaussian을 활용할 수 있다. 



### 2-2. Pixel RNN

generation model로 RNN을 활용할 수 있다. 

![image](https://user-images.githubusercontent.com/71866756/152985465-ed428f50-dba4-4658-b33e-6b1fe5f6dfe0.png)

이런 방식으로 R, G, B값을 생성해낸다. 

Pixel RNN은 chain의 ordering에 기반한 두 모델이 있다. 

- **Row LSTM**

- **Diagonal BiLSTM**

  ![image](https://user-images.githubusercontent.com/71866756/152985479-5a58f6c6-255f-49f4-b81a-519b10150d79.png)



# 3. Variational Auto-encoder

`Variantional inference` : VI의 목적은 posterior 분포에 가장 일치하는 variational 분포를 최적화하는데 있다.   
![image](https://user-images.githubusercontent.com/71866756/152985493-21c11c2d-515e-4471-bba4-fbfad0bfb946.png)  
observation이 주어졌을 때, 관심 있어하는 random variable의 확률분포이다. 

일반적으로 posterior distribution을 구하는 것은 어려우며, 때로는 불가능하다. 

따라서 posterior distribution을 제일 잘 근사하는 Variational distribution을 찾는 것이 목적이다.   
![image](https://user-images.githubusercontent.com/71866756/152985521-de7479bb-5b25-4e5f-9aa9-1e7e56fc357b.png)  
이 때, KL발산을 이용한다. 

[**Reference**]

[deep generative models](https://deepgenerativemodels.github.io/)
