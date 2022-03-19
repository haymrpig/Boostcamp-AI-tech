# 목차

- [**3D data 표현방법**](#1-3d-data-표현방법)
- [**3D datasets**](#2-3d-datasets)
- [**3D tasks**](#3-3d-tasks)

# 1. 3D data 표현방법

3D data를 표현하는 방법에는 여러가지가 있다. 

![image](https://user-images.githubusercontent.com/71866756/159128961-626f157a-0829-413e-8145-1aa606c32375.png)

- **Multi-view images**

- **Volumetric (voxel)**

- **Part assembly**

- **Point cloud**

- **Mesh (Graph CNN)**

  vertex와 edge로 구성되며, 3개의 vertex를 하나의 삼각형으로 만들어 표현하는 방법

- **Implicit shape**

  2D에서 경계면을 함수로 나타내듯이, 3D에서도 경계를 함수로 나타내는 방법

# 2. 3D datasets

- **ShapeNet**

  3D이미지 51300장으로 3D는 귀하기 때문에 large scale 데이터셋이라고 표현할 수 있다. 

- **PartNet**

  Sharpnet 상위호환 버전으로 part assembly로 표현

- **SceneNet**

  실내 이미지에 대한 RGF-Depth로 표현된 데이터셋

- **ScanNet**

  실내 이미지를 실제로 스캔한 것 (실제로 스캔했기 때문에, 이미지 중간중간 잘 스캔되지 않은 영역이 있다. )

- **Outdoor datasets**

  - **KITTI**

    LiDAR, 3D Bbox

  - **Semantic KITTI**

    LiDAR

  - **Waymo Open Dataset**

    LiDAR, 3D Bbox

# 3. 3D tasks

- **3D recognition**

  2D 이미지를 3D로 바꿔줌

- **3D object detection**

- **3D semantic segmentation**

- **conditional 3D generation**

  - Mesh R-CNN

    Mask R-CNN에서 3D branch가 추가된 것!

    2D를 3D로 바꾸는데, 다양하게 표현 가능 (Voxels, Meshes 등등)

    ![image](https://user-images.githubusercontent.com/71866756/159128967-28de53af-262e-4aaf-bf7e-0e173c1fb053.png)

    
