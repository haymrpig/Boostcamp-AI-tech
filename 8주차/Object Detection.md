# 1. Object detection이란

classification + Bbox localization의 조합!

#### 1-1. Traditional methods

- **Gradient-based detector**

  `HOG` : 경계선의 분포를 검출하여 모델링

  `Selective search` : bbox proposal 알고리즘

- **Two-stage detector**

  - **R-CNN**

    Step1. region proposal

    Step2. warp region (fixed size)

    Step3. CNN (pretrained)에 input으로 삽입

    Step4. SVM을 학습하여 분류

    - 단점 : 느림, 성능향상 한계가 존재

  - **Fast R-CNN**

    Step1. region proposal

    Step2. ROI projection on feature map (feature map 재활용 가능)

    Step3. ROI pooling layer 거침

    Step4. classification + bbox regression 진행

    - 단점 : region proposal이 느린 것

  - **Faster R-CNN**

    첫 End-to-end 모델

    Step1. set pre-defined Anchor box (IoU에 따라 negative, positive sample로 구분)

    Step2. Region Proposal Network (RPN) + Non-Maximum Suppression (NMS)

    Step3. classification + bbox regression 진행

- **One-stage detector**

  - **YOLO**

  - **SSD**

    multiple feature map에 대해서 가능한 box 모양을 다양하게 함 (YOLO보다 정확도 상승)

  - **RetinaNet**

    Feature Pyramid Networks (FPN) + class/box regression

    SSD보다 빠르고 정확하다. 

  - **DETR (transformer)**

    

- **Two-stage v.s One-stage**

  One-stage의 경우 ROI pooling이 없기 때문에 모든 anchor box에 대해 loss가 계산된다. 

  -> 배경이 더 많은 이미지 특성상 일정 gradient가 발생한다. 

  -> Focal loss 추천 (잘 맞추는 애들은 더 작은 loss로, 못 맞추는 애들은 더 sharp한 loss로)

  >  sharp하다는 것은 gradient가 더 크다는 것