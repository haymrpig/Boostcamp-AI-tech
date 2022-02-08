# 목차

- **Vanilla RNN**
- **LSTM**
  - 기본 설명
  - 코드 설명
  - 파라미터 계산방법
- **GRU**

# 1. Vanilla RNN

`Sequential`데이터는 어느 정도 길이의 데이터를 통해 판단할지 모르기 때문에 매우 까다롭다. 

### Fixed timespan

정해진 양의 과거 데이터만큼만 고려

- **Recurrent Neural Network의 기본**

  과거의 정보를 포함해야 하기 때문에, 자기 자신으로 돌아오는 recurrent 구조를 가진다. 

  이러한 recurrent구조는 매우 복잡해보일 수 있지만, 일정 time span을 정하여 sequential 구조로 생각하면 쉽게 풀어낼 수가 있다. 

  - **Short term dependency**

    short-term dependency, 한참 먼 과거 데이터는 현재 시점까지 유지하기 어렵다. 

    > activation function이 sigmoid라고 했을 때, h1->h2->h3가 있다고 하자.
    >
    > h2는 sigmoid(h1)을 포함하고, h3는 sigmoid(h2, sigmoid(h1))을 포함하고 있다. 
    >
    > 따라서 h3에서의 h1정보는 매우 작아진다는 것을 알 수 있다. 
    >
    > 만약 h3....hn까지 매우 긴 sequential 연결이 있다면 h1에서의 정보는 거의 사라질 것이다. 
    >
    > 반대로 ReLU의 경우는 gradient가 exploding할 수도 있다.

- **Markov model**

  바로 이전의 데이터만 고려하는 것  
  ![image](https://user-images.githubusercontent.com/71866756/152985675-d118cf62-b92d-4546-95eb-020c710d49e7.png)  

  > 장점 : joint distribution을 표현하는데 용이하다. 
  >
  > 단점 : 더욱 더 먼 과거의 데이터를 포함하지 못한다. 

- **Latent autoregressive model**

  중간에 hidden state가 들어가 있으며, hidden state가 과거의 정보를 요약하고 있다고 본다. 

  > 장점 : markov와는 다르게 더욱 먼 과거의 데이터를 포함할 수 있다. 



# 2. Long Short Term Memory (LSTM)

![image](https://user-images.githubusercontent.com/71866756/152985722-bb7fa849-a5b3-4a2b-aeb0-9b3e0b536d4d.png)

> LSTM의 전체적인 개요를 보여주는 그림이다. 

### 2-1. LSTM의 핵심 요소

- **Forget Gate**

  어떤 정보를 배제할지 정해주는 gate  
  ![image](https://user-images.githubusercontent.com/71866756/152985750-2b76ca3f-7c5e-4bc6-b973-8af786163c47.png)  

  > Previous hidden state와 input에서 버릴 정보를 결정

- **Input Gate**

  어떤 정보를 cell state에 저장할지 결정하는 gate  
  ![image](https://user-images.githubusercontent.com/71866756/152985777-00b205cc-b9a2-42ad-9aeb-a6abb1248d87.png)  

  > Previous hidden state와 input에서 저장할 정보를 결정

- **Update cell**

  cell state를 업데이트한다.  
  ![image](https://user-images.githubusercontent.com/71866756/152985832-a2c29319-c156-44a5-b9a9-4723b634f667.png)  

- **Output Gate**

  output을 내보낸다.   
  ![image](https://user-images.githubusercontent.com/71866756/152985869-384521c0-8039-4053-aea0-e3f7a0a9129b.png)  

> Cell state 차원과 Output의 차원은 일치해야 한다. 
>
> LSTM의 경우, 생각이상으로 파라미터 수가 많다. (각각의 gated function이 dense layer이다.)

### 2-2. 코드 주석

```python
class RecurrentNeuralNetworkClass(nn.Module):
    def __init__(self,name='rnn',xdim=28,hdim=256,ydim=10,n_layer=3):
        super(RecurrentNeuralNetworkClass,self).__init__()
        self.name = name
        self.xdim = xdim
        self.hdim = hdim
        self.ydim = ydim
        self.n_layer = n_layer # K

        self.rnn = nn.LSTM(
            input_size=self.xdim,hidden_size=self.hdim,num_layers=self.n_layer,batch_first=True)
            # batch first는 output이 어떤 형태로 나오는지 정하는 것
            # [seq_len, batch, num_directions * hidden_size]가 [batch, seq_len, num_directions * hidden_size]로
            # 나온다. 

        self.lin = nn.Linear(self.hdim,self.ydim)

    def forward(self,x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(
            self.n_layer, x.size(0), self.hdim
        ).to(device)
        # h0는 처음 시점보다 이전시점의 정보는 없으므로 0으로 초기화를 시켜준다.
        # h0의 차원은 3차원으로 layer의 개수, batch 크기, hdim (feature 개수) 라고 생각하면 된다 .
        # nn.LSTM doc를 찾아보면, h0 : (D*num_layers, N, Hout)인데 여기서 D는 단방향일 때 1, 양방향일 때 2의 값이다. 
        # 현재 우리는 단방향이므로 D = 1이다. 
        c0 = torch.zeros(
            self.n_layer, x.size(0), self.hdim
        ).to(device)
        # c0 역시 마찬가지이다. 

        # RNN
        rnn_out,(hn,cn) = self.rnn(x, (h0,c0)) 
        # x:[N x L x Q] => rnn_out:[N x L x D]
        # N은 batch size, L은 sequence length, Q는 input feature이다. 
        # nn.Linear의 input으로는 input, (h_0, c_0)가 들어간다. 
        # nn.Linear의 output으로는 output, (h_n, c_n)이 나온다. 
        # output의 ndim은 3으로 (N:batch, L:sequence_length, D*H_out : D=1, H_out=output feature)

        # Linear
        out = self.lin(
            rnn_out[:,-1:]
            ).view([-1,self.ydim]) 
        # 현재 rnn_out은 (N:batch size, L:sequence lenght, Hout : feature 수)로 되어있는데,
        # LSTM의 결과로 나온 마지막 sequence의 결과를 사용할 것이므로
        # rnn_out[:, -1:]를 사용한다. 
        # 즉, 마지막 sequence의 featuer들을 batch만큼 가져오는 것!
        return out 

R = RecurrentNeuralNetworkClass(
    name='rnn',xdim=28,hdim=256,ydim=10,n_layer=2).to(device)
loss = nn.CrossEntropyLoss()
optm = optim.Adam(R.parameters(),lr=1e-3)
print ("Done.")
```

### 2-3. 파라미터 계산방법

![image](https://user-images.githubusercontent.com/71866756/152986036-17cbcb5e-f28f-462d-893c-ad06030583c7.png)

> 위 그림에서 하나의 수식에 대한 Wh, Wx, b parameter의 개수 계산방법이 나와있다. 
>
> 원하는 hout의 dim이 256일 경우 Dh는 256이라고 생각하면 되고, d는 input feature개수라고 생각할 수 있다. 
>
> 하지만 LSTM의 경우는 아래 수식처럼 Wf, Wi, Wo, Wc 각각의 weight들을 갖고 있으며, 
>
> 이들을 모두 고려해야 하기 때문에, weight의 행에 4배를 곱하면 된다. 

![image](https://user-images.githubusercontent.com/71866756/152986018-0053d830-67c4-47f5-a213-bea7b60c270c.png)

```python
np.set_printoptions(precision=3)
n_param = 0
for p_idx,(param_name,param) in enumerate(R.named_parameters()):
    if param.requires_grad:
        param_numpy = param.detach().cpu().numpy() # to numpy array 
        n_param += len(param_numpy.reshape(-1))
        print ("[%d] name:[%s] shape:[%s]."%(p_idx,param_name,param_numpy.shape))
        print ("    val:%s"%(param_numpy.reshape(-1)[:5]))
print ("Total number of parameters:[%s]."%(format(n_param,',d')))
# rnn.weight_ih_10는 첫번째 레이어에서 input과 연산되는 weight parameter를 의미한다.
# 여기서 hdim = 256이고, input feature = 28이다. 
# rnn.weight_ih_10의 parameter는 (1024, 28)인데, 
# 그 이유는 hdim=256이고 여기에 4를 곱하면 1024가 된다. 
# hdim을 128로 바꿨을 때는 512가 된다. 
# 즉, 항상 4배의 결과가 나오는데, 그 이유를 생각해보면,
# LSTM은 총 3개의 gate(forget, output, input)그리고 update cell 부분이 있다. 
# 각각의 식에서 input에 연산되는 weight가 존재하므로, 총 4개의 weight가 필요하고, 
# input (28x1)에서 hout (256x1)로 가기 위해서는 원래는 (256x28)이 필요하지만,
# 식이 네개이므로 총 (1024x28)의 parameter를 가지게 되는 것이다. 
# sequential length와는 상관이 없는게, 이 parameter들은 공유되기 때문이다. 
```



# 3. Gated Recurrent Unit (GRU)

![image](https://user-images.githubusercontent.com/71866756/152986074-9c008170-e4d5-4684-833e-73f2c6e6ec7e.png)

- **LSTM과의 차이점**

  - **Reset gate**와 **Update gate**로만 구성되어있다. 

  - **cell state가 존재하지 않고, hidden state만 존재** 



**[Reference]**

[LSTM의 원리와 수식 계산](https://docs.likejazz.com/lstm/)

[RNN 파라미터 개수 카운팅](https://datascientist.tistory.com/25)

[nn.LSTM input, output document](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
