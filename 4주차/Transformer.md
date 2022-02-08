# 목차

- **Transformer**
  - Sequential model의 어려운 점
  - transformer 개요
  - transformer 구조
  - 잘되는 이유? 한계점?
- **Multiheaded attention (MHA)**

# 1. Transformer

### 1-1. Sequential model의 어려운 점

- **Trimmed sequence** : 일정 길이로 잘린 sequence
- **Omitted sequence** : 중간이 잘린 sequence
- **Permuted sequence** : 순서가 바뀐 sequence

### 1-2. Transformer 개요

`transformer`는 재귀적인 구조가 아닌, **attention**이라고 불리는 구조로 되어있다. 

6개의 동일하지만 (공유하지 않는) encoder와 decoder가 stack되어 있는 구조



### 1-3. Transformer 구조

- **Encoder**

  **Self-Attention + Feed forward neural network**가 하나의 Encoder를 구성한다. 

  - **Self-Attention** 

    ![image](https://user-images.githubusercontent.com/71866756/152986184-efabee0e-4ddb-4438-ac84-36adce35f253.png)

    `Queries`, `keys`, `values`의 세 벡터를 만들어낸다. (각각이 하나의 neural network라고 생각)

    > 즉, 하나의 input (단어) 에 대해서 세개의 벡터를 생성한다. 

    ![image](https://user-images.githubusercontent.com/71866756/152986207-046a5350-0d41-413e-93bc-1dbeeac61354.png)

    해당 단어의 queries 벡터와 나머지 단어들의 key 벡터의 내적을 구하여 score를 낸다. 

    > 현재 단어와 다른 단어들이 얼마나 관계되어 있는지를 의미한다. 

    ![image](https://user-images.githubusercontent.com/71866756/152986241-3588390f-f86a-4c5a-97c5-e833ca214bd8.png)

    score를 normalize를 진행한다. 

    > normalize하는 방법은 
    >
    > queries벡터의 dimension의 sqrt로 나눠주거나, keys 벡터의 dimension의 sqrt로 나눠주는 것
    >
    > 이후, softmax를 취한다. 

    ![image](https://user-images.githubusercontent.com/71866756/152986276-642998e6-2c96-4597-bf21-ad2b79abd56c.png)

    이렇게 나온 결과 **(attention weight)**를 value vector와 weighted sum 한 것이 최종 결과가 된다. 

    >이 값이 **attention value**가 된다. 

    ![image](https://user-images.githubusercontent.com/71866756/152986302-72084702-9d2b-4b92-b4e5-8e20d342bc9c.png)

    ![image](https://user-images.githubusercontent.com/71866756/152986323-5265925d-b759-4f67-b163-0acd1cf28e1d.png)

    위 과정을 모두 요약하면 위 그림처럼 표현할 수 있다. 

    > !!!!주의할 점!!!!
    >
    > Queries와 keys는 Q dot K.T 연산을 진행해야 하기 때문에 차원이 같아야 한다.
    >
    > Values는 차원이 같을 필요는 없지만, 구현상의 편의성을 위해서 맞추는 편!

     

  - **Feed forward neural network** : 기존 fully connected network와 동일

- **Decoder**

  마지막 Encoder에서 나온 Keys와 Values가 모든 decoder로 넘어가고,  decoder를 통해 해석된 단어는 다시 decoder의 input으로 들어가서 Queries로써 사용이 된다. 

  

- **동작 방식 (큰 틀)**

  아래 그림처럼 input이 단어 단위로 쪼개진다. 

  ![image](https://user-images.githubusercontent.com/71866756/152986354-cfd53d86-38e5-4735-8016-123bbcb7d4e4.png)
  

  쪼개진 단어들이 **self-Attention**을 거치는데, 이때, x1->z1으로 변환이 일어날 때, z1은 비단 x1뿐만이 아닌, x2...xn까지의 값들을 포함하게 된다.

  > 예를 들어, The animal didn't cross the street because it was too tired라는 문장이 있다고 하자.
  >
  > 이  때, it이 의미하는 것은 무엇인지 다른 단어와의 관계를 고려할 필요가 있다. 
  >
  > 실제로 transformer의 결과로 나오는 it은 animal과 깊은 관계가 있다. 

  **Feed Forward** 의 경우는 dependency가 없다. 즉, x1->z1으로 변환이 일어날 때, x2...부터는 고려되지 않는다. 

  ![image](https://user-images.githubusercontent.com/71866756/152986371-8b73e31d-e264-4f87-b529-5903fe81a0c3.png)



### 왜 잘될까? 한계점은?

MLP의 경우, 입력에 따라 출력이 고정되어 있는 것으로 인지할 수 있다. 

하지만, transformer의 경우 다른 값들과의 상관도를 고려해서 값이 정해지기 때문에, 출력이 고정되어 있지 않고 다른 값들에 따라 값이 변화될 수 있다. 

그러므로 MLP보다 더 flexible하며 더 많은 것을 표현할 수 있다.  

**하지만**, RNN과 다르게 transformer의 경우 한 번에 단어들을 embedding하기 때문에 Computation cost가 길이의 제곱만큼 많아진다. 

따라서 길이가 길어지면 처리하는데 어려움이 생길 수 있다. 

> RNN의 경우는 1000개의 단어가 있을 때, 시간이 오래 걸리겠지만, 1000번의 sequence로 처리를 할 순 있다. 
>
> 하지만 transformer의 경우는 한번에 1000개의 단어를 봐야하기 때문에 RNN보다 확실한 한계점이 존재한다.  



# 2. Multiheaded attention (MHA)

Queries, Keys, Values들을 한 단어당 여러개 (multiple attention heads)를 만드는 방법이다. 

- **n개의 head**

  n개의 head가 있을 때, n개의 attention value가 나오게 된다. 

  나온 n개의 attention heads들을 다시 encoder의 입력으로 넣어야 하기 때문에, 입력차원과 똑같이 맞춰줘야 한다.

  ![image](https://user-images.githubusercontent.com/71866756/152986416-534043f7-d16e-4eaa-9260-3f354ced7aca.png)

  따라서 n개의 attention heads들을 stack하고, n x input_dim의 weight를 곱해주어 입력차원과 동일하게 맞춰준다. 

- **Positional encoding**

  사실 위 과정만으로 transformer가 time sequential하다고 할 수 없다. 

  즉, abcde와 bcade랑 동일하다는 의미이다. 

  그렇기 때문에, embedding vector에 위치 정보들을 더해주는 positional encoding이 필요하다. 

  따라서 같은 단어라도 서로의 값이 달라지는 것이다. 

**[Reference]**

[The Illustrated Transformer]([The Illustrated Transformer – Jay Alammar – Visualizing machine learning one concept at a time. (jalammar.github.io)](http://jalammar.github.io/illustrated-transformer/))

# 질문

1. **왜 positional encoding이 필요한가? 왜 positional encoding이 없으면 단어들을 time independent할까?**

   만약 같은 단어가 각각 문자의 맨 앞, 맨 뒤에 배치되었다고 생각해보자. 

   해당 단어의 embedding값은 같을 것이다. 

   그렇다면 이 두 단어는 어떻게 구분해야 할까?

   위 예시만 보더라도 transformer는 time sequential을 보장하지 못한다는 것을 알 수 있다. 그렇기 때문에 embedding에 위치 정보를 나타내는 값을 더해주어 같은 단어라도 다른 위치에 있다는 것을 인지시켜줘야만 한다.  

   
