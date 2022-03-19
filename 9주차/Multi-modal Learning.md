# 목차

- [**Multi-modal learning이란**](#1-multi-modal-learning이란)
- [**Multi-modal learning tasks(Image&text)**](#2-multi-modal-learning-tasks(image&text))
- [**Multi-modal learning tasks (Audio&Image)**](#3-multi-modal-learning-tasks-(audio&image))

# 1. Multi-modal learning이란

`multi-modal learning` : 서로 다른 데이터 타입을 같이 학습하는 것을 말한다. 

> 예를 들어, text, vision, audio 등을 같이 학습하는 것

#### 1-1. Multi modal learning의 어려움

- **서로 다른 데이터 타입인 만큼 서로의 표현 방법이 다르다.**

  >  audio : 1d
  >
  > image : 2d or 3d
  >
  > text : embedding vector

- **Unbalnce between heterogeneous feature space**

  하나의 문장이 주어진 경우, 그 문장을 표현하는 이미지는 여러개일 수 있다. 

  > '초원을 달리는 말'이라는 문장이 주어졌을 때,
  >
  > 말의 이미지와 초원의 이미지는 여러가지가 나올 수 있다. 

- **특정한 modality에 bias될 수 있다.**

  딥러닝 모델은 쉬운 길을 택하는 경향이 있다. 

  > 여러 modality (이미지, 소리 등) 에서 이미지에 대해서 bias하게 train되었을 때, 만약 학습과정에서 모델이 이미지에 대한 loss를 낮추는게 효율적이라 판단하면, 이미지에 대해서편향되어 학습이 진행될 수 있다. 

#### 1-2. Multi modal learning 방법

- **Matching**

  서로 다른 modality를 공통의 영역으로 엮는 것

- **Translating**

  하나의 modality를 다른 modality로 전환하는 것

- **Referencing**

  동일한 modality를 입력과 출력으로 할 때, 다른 modality를 참고하는 것

# 2. Multi-modal learning tasks(Image&text)

#### 2-1. Matching방식 소개

Visual data와 text data를 **Matching 방식을 통해 학습**하는 것을 `Joint Embedding`이라고 한다.

먼저 Text가 어떻게 embedding되는지 알아보자.  

단어가 주어졌을 때, dense vector로 embedding한다.

![image-20220320001334369](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220320001334369.png)

위 그림에서 볼 수 있듯이, embedding 결과를 2차원으로 시각화 했을 때, 뛰어난 일반화 능력을 가진 것을 볼 수 있다. 

오른쪽 그림에서처럼 비슷한 단어들끼리 모이거나, 뜻이 상반되는 경우 비슷한 양상을 보이는 것을 알 수 있다. 

**Word2Vec** 모델에는 **Skip-gram model**이 있다. 

**Skip-gram model**은 중심 단어를 통해 주변단어를 예측하는 모델이다. 

![image-20220320001845693](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220320001845693.png)

하나의 단어와 주변 단어들의 관계 파악을 목적으로 하며 아래 그림처럼 나타낼 수 있다. 

![image-20220320002025830](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220320002025830.png)



이제 `Joint embedding`의 예시를 알아볼 것이다. 

**image tagging**은 대표적인 방식으로

이미지가 주어지고 tag를 추천 / tag가 주어지고 이미지를 추천하는 것을 예시로 들 수 있다. 

> 즉, 강아지 이미지가 주어지면 제일 잘 설명하는 tag를 추천
>
> 또는 tag가 주어지면 해당 tag와 가장 잘 어울리는 이미지를 추천하는 것!

![image-20220320002543921](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220320002543921.png)

위 그림에서처럼 text와 image를 고정된 feature vector로 만드는 것을 시작으로, 두 데이터를 같은 space를 공유하도록 한다. 

> 위처럼 학습할 경우 아주 재밌는 특성을 확인할 수 있는데,
>
> 강아지 이미지 embedding - 강아지 text embedding + 고양이 text embedding = 고양이 이미지 embedding으로 표현될 수 있다!
>
> 즉, 이미지와 text가 같은 space에 있다는 것이 증명이 되는 것이다! 



#### 2-2. Translating 방식 소개

이 예시도 image와 text가 관계되어 있는데, 그 중 대표적인 방법이 `Image Captioning`이다.

`Image Captioning`이란 이미지가 주어지면 이미지를 가장 잘 설명하는 text를 보여주는 것이다. 

> 강아지가 밥을 먹는 그림 -> '강아지가 밥을 먹는다.'

그렇다면, 이제 어떻게 이 방식이 가능한지에 대해 설명해보겠다.

Step1. image가 CNN을 통과 (encoder)

Step2. 그 결과가 RNN을 통과하여 text출력 (decoder)

생각보다 단순해 보이지만 효과적이다. 

여기에서 더 나아가면 어떤 단어가 이미지에서 어디 부분을 attention하는지도 확인할 수 있다. 

관련 정보는 **Show, attend and tell** 논문에서 확인할 수 있다. 

![img](https://miro.medium.com/max/720/0*bbvw5z9V83UmGnsS.jpg)

이 그림이 **Show, attend and tell**을 설명해주는 대략적인 그림이다. 

CNN을 통과한 image가 RNN을 통과하면서 각각의 feature에 해당하는 단어들을 생성하게 된다. 

`Image captioning` 다음은 `Text-to-image`가 있다. 

Generative model을 이용하여, text가 주어지면 image를 생성하는 것이다. 



#### 2-3. Referencing 방식 소개

`Multiple Stream`을 예시로 들 수 있는데, 말 그대로 stream이 여러개 있다는 의미이다. 

이미지가 주어지고, 그에 대해 질문을 입력하면 이미지를 토대로 질문에 대한 답을 출력한다. 

아래 그림처럼, Image Stream과 Question Stream이 존재한다. 

![image-20220320003850821](C:\Users\Administrator1\AppData\Roaming\Typora\typora-user-images\image-20220320003850821.png)



# 3. Multi-modal learning tasks (Audio&Image)

먼저 audio 데이터가 embedding되는 방식을 알아보자. 

audio는 time serial로 주어지고, 주어진 데이터를 Fourier transform을 거쳐 주파수 domain으로 바꾼다.

이 때, 모든 시간대에 걸쳐서 FT를 거칠 경우, 시간대에 따른 분석이 어려워진다.

> 긴 시간동안의 데이터가 모두 주파수 영역으로 바뀌기 때문에, 시간에 따라 분석이 힘들어진다.

따라서, **Short-time Fourier transform (STFT)** 를 이용한다. 

winodw를 정해 shift시키면서 FT를 하는 방식이다. 

여기서도 Matching, Translating, Referencing으로 나눠서 해당 task에 따른 모델을 찾아볼 수 있다. 

관련해서 더 찾아봐도 좋다. 

여기서는 간단하게만 설명하겠다. 

#### 3-1. Matching (SoundNet)

`SoundNet` : 소리에 따라 이미지를 추천하는 모델

#### 3-2. Translating (Speech2Face, Image-to-speech)

`Speech2Face` : 소리가 주어지면, 해당하는 얼굴 생성하는 모델

`Image-to-speech` : 이미지가 주어지면 해당하는 소리를 생성하는 모델

#### 3-3. Referencing (Sound source localization, Lip movements generation)

`Sound source localization` : 이미지와 소리가 주어지면, 소리가 이미지에서 어느 부분의 소리인지 heatmap 형식으로 출력하는 모델

> 예를 들어, 도로 사진과 경적 소리가 주어지면, 도로 위 자동차가 attention 된다. 

`Lip movements generation` : 소리에 따라 이미지가 실제로 말하는 것처럼 만들어주는 모델
