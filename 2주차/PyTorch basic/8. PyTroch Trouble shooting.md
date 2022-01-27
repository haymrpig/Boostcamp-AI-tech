# PyTroch Trouble shooting

### OOM (out of  memory)

- 발생 원인을 알기 어려우면, 발생한 곳을 알기도 어렵다.
- Error backtracking이 이상한 곳으로 가며, 메모리의 이전상황의 파악이 어렵다. 

- 단순한 해결방법
  - Batch size를 줄인다.
  - GPU를 clean한다. 
  - colab의 경우 다시 launch한다. 

### GPUUtil

- nvidia-smi처럼 GPU의 상태를 보여주는 모듈 (nvidia와 달리 매 iteration마다 확인이 가능하다)

```python
!pip install GPUtil

import torch
import GPUtil
from GPUtil import showUtilization as gpu_usage

gpu_usage()

tensorList=[]
for x in range(10):
    tensorList.append(torch.randn(100000000, 10).cuda())
# 메모리 사용하는 중을 가정
# 메모리가 계속 늘어나면 잘못된 것, 메모리가 쌓이고 있는 것

gpu_usage()

del tensorList
gpu_usage()

torch.cuda.empty_cache()
gpu_usage()
```



### torch.cuda.empty_cache()

- 사용되지 않은 GPU상 cache를 정리, 가용 메모리 확보

- del과는 구분이 필요하며 reset 대신 쓰기 좋은 함수이다. 

  (del은 관계를 끊으면서 메모리를 free하는 것)

