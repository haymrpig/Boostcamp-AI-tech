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

가장 먼저 생각해야 하는 것은 왜 Variational Auto-encoder가 나왔냐이다. 

기존 training database에 있던 샘플과 유사한 샘플을 생성하기 위해서는 이미지의 분포를 파악해야 될 필요성이 있다. 

즉, prior값을 활용해야 한다는 뜻이다. 

학습이란 것은 training database에서 유사한 샘플들을 샘플링하여 그 분포와 비슷한 분포를 표현할 수 있어야 된다는 것을 의미한다.

> 쉽게 말해서, 고양이 이미지를 생성하고 싶으면 dataset에서 샘플링을 통해서 고양이들만을 추출하고 그에 따른 분포를 알아내야 한다는 것이다. 

그렇다면, training dataset에서 고양이 이미지 분포를 알기 위해서 몇 개의 이미지들을 샘플링하고 그에 따른 MLE를 최적화하면 고양이에 대한 분포를 학습할 수 있지 않을까?

그렇지 않다. 

실제로 MLE는 euclidean 거리를 이용하기 때문에 이미지가 비슷한 것을 잘 찾아내지 못한다. 

> 예를 들어, MNIST data에서 숫자 7가 있다고 하자. 
>
> 이 이미지에서의 숫자 7를 오른쪽으로 조금 이동시켰을 경우, 이미지의 유클리드 거리는 매우 커지게 된다.
>
> 그렇다면 이번에는 숫자 7과 9가 있다고 해보자. 이 둘은 매우 비슷하며 같은 위치에 있다고 해보자. 
>
> 아까의 경우 오른쪽으로 살짝 이동한 경우보다 이 경우의 유클리드 거리가 더욱 가까울 것이다. 
>
> 따라서, MLE를 직접적으로 이용하는 것은 옳지 못하다. 

다음 step으로 필요한 것은 적절히 샘플링을 해줄 수 있는 함수를 찾는 것이다. 

이러한 이상적인 샘플링 함수를 통해 적절한 분포를 찾는 것이 **Variantional inference**이다. 

`Variantional inference` : VI의 목적은 posterior 분포에 가장 일치하는 variational 분포를 최적화하는데 있다.   
![image](https://user-images.githubusercontent.com/71866756/152985493-21c11c2d-515e-4471-bba4-fbfad0bfb946.png)  
posterior distribution이란 observation이 주어졌을 때, 관심 있어하는 random variable의 확률분포이다. 

일반적으로 posterior distribution을 구하는 것은 어려우며, 때로는 불가능하다. 

따라서 posterior distribution을 제일 잘 근사하는 Variational distribution을 찾는 것이 목적이다.   
![image](https://user-images.githubusercontent.com/71866756/152985521-de7479bb-5b25-4e5f-9aa9-1e7e56fc357b.png)  
사실, posterior distribution을 모르는데, 근사한다는 것은 말이 안된다...

하지만, **ELBO(Evidence Lower bound)** 트릭과 **KL 발산**을 이용하면 구할 수가 있다!

먼저, 우리의 목적은 **KL 거리를 줄이는 방향**으로 가야한다는 것이며, p(x)를 최적화해야한다. 

따라서 p(x)의 최대값을 구하기 위해서 log를 취하고 아래와 같은 수식으로 전개가 된다. 

 ![img](https://blog.kakaocdn.net/dn/yDmgZ/btqFz3Et6Xt/z0YGviRwKQQ8GyuoDpNgYk/img.png)

[출처] https://deepinsight.tistory.com/127

우리는 Posterior distribution을 알 수 없기 때문에 KL 거리를 직접적으로 줄이는 것은 불가능하다. 

하지만, **ELBO (Evidence Lower bound)** 항을 크게 만들면, 자연스럽게 KL 거리가 최소가 되게 만들 순 있다. 

ELBO는 두 개 항으로 나눠지는데 아래 그림과 같다. 

![img](https://blog.kakaocdn.net/dn/chuWOj/btqFBeFdJrV/9Sgg3biBcVz4hfk8NDjyV1/img.png)

[출처] https://deepinsight.tistory.com/127

ELBO를 쪼갰을 때 나오는 좌측 항은 **reconstruction term**이라고 하며 **reconstruction loss**를 최소화한다. 

오른쪽 항은 **Prior Fitting Term**으로 **latent distribution**이 prior distribution과 **비슷하도록** 만들어준다. 

이제 쪼갠 ELBO 수식을 다시 위의 수식과 합치면 아래와 같다. 

![img](https://blog.kakaocdn.net/dn/bndyC4/btqFA59BaZS/DLvwBkBvsuLDYuP8hfYXX1/img.png)

그렇다면 우리는 위의 식을 최적화 해야 한다. 

전체적인 구조와 loss function은 아래와 같다. 

![img](https://blog.kakaocdn.net/dn/cTFbqv/btqFArSpAI2/Nrc0tHbUjRkIKXiEJKxSK0/img.png)

![img](https://blog.kakaocdn.net/dn/cqZd6V/btqFDxp8RTL/4fMcwX6ba0A3T9NLOeAt11/img.png)

> Variational Auto-encoder는 몇 가지 제한 사항이 있다. 
>
> 1. 먼저 VA는 emplicit하지 않다. 즉, 확률로 나오는 것이 아니다. (hard to evaluate likelihood)
>
> 2. ELBO의 우측항은 KL 거리를 구하기 위해서 SGD든 Adam이든 optimizer를 사용해 optimize해야한다. 따라서 이 항은 미분이 가능해야 한다. 
>
>    하지만, KL 발산 식에는 적분이 들어가 있고, 100프로 미분 가능한지 확신할 수 없기 때문에 보통 prior distribution으로 isotropic Gaussian을 사용하여 미분이 가능하도록 한다. 

아래 식은 isotropic gaussian을 prior distribution으로 사용한 경우이다. 

![image](https://user-images.githubusercontent.com/71866756/153200507-813cdbe1-a5d3-4893-b2ec-473de308869a.png)



# 4. Adversarial Auto-encoder

Variational Auto-encoder은 좋은 방법이 될 수 있지만, prior distribution으로 gaussian과 같은 미분가능한 분포를 사용한다는 제한사항이 있다.

이러한 제한사항을 무시하기 위해서 Adversarial Auto-encoder를 사용한다.  

GAN의 discriminator를 활용하여 latent distribution과 prior distribution 사이의 분포를 맞춰주는 것이다. 

즉, ELBO의 우측항을 GAN으로 대체한 것이다. 

이렇게 교체하여 샘플링만 가능하다면 어떠한 분포를 사용해도 상관이 없다. 

> 좀 더 직관적으로 이해해보자면, ELBO의 KL term은 prior 값과 sampling의 값이 일치하도록 만들어준다는 것이다. 
>
> 이것을 GAN에 대입해보자면, Prior는 진짜 이미지를, sampling한 값은 가짜 이미지를 의미한다. 

- **VAE 와 AAE의 차이점**

  VAE는 정규 분포에 가깝게 학습이 된다. 

  AAE는 prior에 가깝게 학습이 된다. 

# 5. GAN

GAN은 가짜 이미지를 생성하는 generator와 이미지의 진위여부를 판단하는 discriminator로 구분이 되면, 이 둘의 minmax game이라고 할 수 있다. 

![image](https://user-images.githubusercontent.com/71866756/153200541-e7a4019a-8611-4ad4-9d94-dd05d3d1ddca.png)

위 수식으로 optimize를 한다. 

먼저 **discriminator**를 따로 살펴보자.

![image](https://user-images.githubusercontent.com/71866756/153200560-a74bd18d-7372-42c7-9693-84a8a78c8364.png)

위 수식으로 D를 maximize하는 방향으로 학습을 진행한다. 

그렇다면 optimal discriminator는 아래 수식이 된다. 

![image](https://user-images.githubusercontent.com/71866756/153200577-cbf74a77-b84f-4401-bd7e-e4c4678d7718.png)

수식에서 만약 fix된 generator를 사용한다면, 수식의 값이 클수록 discriminator는 실제 이미지라고 판단해야 하고, 수식의 값이 작을수록 discriminator는 가짜 이미지라고 판단해야 한다. 

이제, **generator**를 살펴보자.  
![image](https://user-images.githubusercontent.com/71866756/153200654-a4b8d017-9800-41d7-9bea-93abfa6de51f.png)  

라고 표현할 수 있다.

그렇다면 위에서 구한 optimal discriminator수식을 대입해보자.

 ![image](https://user-images.githubusercontent.com/71866756/153200596-e578d11d-fb9c-4a7b-9aec-6820aed8484b.png)

위와 같은 수식을 얻을 수 있으며, 목적은 Jenson-Shannon divergence를 줄이는 것이 목적이 된다. 

사실 위 수식이 가능한 이유는 optimal discriminator가 수렴한다는 가정하에 가능한 것이다. 즉, 이론적으로는 make sense하지만, 실제로는 그렇지 않을 수도 있다.



# 6. ETC

- **DCGAN**

  이미지 domain에서 GAN을 활용한 것

- **Info-GAN**

  auxiliary class를 랜덤하게 넣어줌으로써 generation을 할 때, GAN이 특정 모드에 집중하게 해준다. 

- **Text2Image**

  문장이 주어지면 사진을 만들어낸다. 

- **Puzzle-GAN**

  물체의 sub patch로 물체 전체를 복원

- **CycleGAN**

  이미지의 domain을 바꿈 (말을 얼룩말로, 얼룩말을 말로, 사진을 그림으로, 그림을 사진으로 등등)

  `Cycle consistent loss`

- **Star-GAN**

- **Progressive-GAN**

  4x4부터 1024x1024로 점점 키워서 이미지를 생성한다. 

# 

[**Reference**]

[deep generative models](https://deepgenerativemodels.github.io/)

[variational inference](https://deepinsight.tistory.com/127)
