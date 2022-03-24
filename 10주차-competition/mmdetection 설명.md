# MMCV

- runner

  > ex) EpochBasedRunner, IterBasedRunner

  - workflow

    ex) workflow = [('train', 3),('val', 1)] : EpochBasedRunner의 경우 train 3epoch돌고, val 1epoch도는 것!

    workflow = [('val', 1), ('train', 3)] 이런식으로도 사용가능



# Configs

- **config의 _ base _를 통해서 base 구조를 이용하라.** 

  > 만약 base 구조를 사용하고 싶지 않다면, 새로 config 디렉토리 안에 custom config파일을 작성하라. 
  >
  > 이 때, 파일 이름의 형식은 정해져 있는데
  >
  > 최대한 {model}_ [model setting]_ {backbone}_ {neck}_ [norm setting]_ [misc]_ [gpu x batch_per_gpu]_ {schedule}_{dataset} 이 형식을 따를 수 있도록
  >
  > 각각의 parameter는
  >
  > - `{model}`: model type like `faster_rcnn`, `mask_rcnn`, etc.
  > - `[model setting]`: specific setting for some model, like `without_semantic` for `htc`, `moment` for `reppoints`, etc.
  > - `{backbone}`: backbone type like `r50` (ResNet-50), `x101` (ResNeXt-101).
  > - `{neck}`: neck type like `fpn`, `pafpn`, `nasfpn`, `c4`.
  > - `[norm_setting]`: `bn` (Batch Normalization) is used unless specified, other norm layer type could be `gn` (Group Normalization), `syncbn` (Synchronized Batch Normalization). `gn-head`/`gn-neck` indicates GN is applied in head/neck only, while `gn-all` means GN is applied in the entire model, e.g. backbone, neck, head.
  > - `[misc]`: miscellaneous setting/plugins of model, e.g. `dconv`, `gcb`, `attention`, `albu`, `mstrain`.
  > - `[gpu x batch_per_gpu]`: GPUs and samples per GPU, `8x2` is used by default.
  > - `{schedule}`: training schedule, options are `1x`, `2x`, `20e`, etc. `1x` and `2x` means 12 epochs and 24 epochs respectively. `20e` is adopted in cascade models, which denotes 20 epochs. For `1x`/`2x`, initial learning rate decays by a factor of 10 at the 8/16th and 11/22th epochs. For `20e`, initial learning rate decays by a factor of 10 at the 16th and 19th epochs.
  > - `{dataset}`: dataset like `coco`, `cityscapes`, `voc_0712`, `wider_face`.

- **model의 training과 conifg 설정은 model dict 안에서 해라**

  ex)

  ```python
  model = dict(
     type=...,
     ...
     train_cfg=dict(...),
     test_cfg=dict(...),
  )
  ```

- **만약 _ base_를 통해 가져온 구조에서 수정하고 싶은 것이 있다면 _delete _=True를 이용해라**

  ex)

  ```python
  _base_ = '../mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py'
  model = dict(
      pretrained='open-mmlab://msra/hrnetv2_w32',
      backbone=dict(
          _delete_=True, # 이런식으로 True로 설정하면 기존 base에서 backbone부분을 덮어쓰게 된다. 
          type='HRNet',
          extra=dict(
              stage1=dict(
                  num_modules=1,
                  num_branches=1,
                  block='BOTTLENECK',
                  num_blocks=(4, ),
                  num_channels=(64, )),
              stage2=dict(
                  num_modules=1,
                  num_branches=2,
                  block='BASIC',
                  num_blocks=(4, 4),
                  num_channels=(32, 64)),
              stage3=dict(
                  num_modules=4,
                  num_branches=3,
                  block='BASIC',
                  num_blocks=(4, 4, 4),
                  num_channels=(32, 64, 128)),
              stage4=dict(
                  num_modules=3,
                  num_branches=4,
                  block='BASIC',
                  num_blocks=(4, 4, 4, 4),
                  num_channels=(32, 64, 128, 256)))),
      neck=dict(...))
  ```



# Customize Datasets

- **기본적으로 Data의 format을 바꾸려면, COCO를 추천하고, offline에서 convert하는 것을 추천**

  > COCO json의 기본 형식은
  >
  > - images
  > - annotations
  > - categories
  >
  > 이 셋을 무조건 포함해야 한다. 

- classes를 변경할 때는 data dict 내부의 train, val, test dict 안에 classes 입력

  > 주의할 점
  >
  > 1. model의 head 부분에 있는 num_classes 인자도 data에 맞춰서 바꿔줘야 한다. 
  > 2. annotation 내부의 categories가 변경한 classes와 일치해야 한다. (순서 또한)
  > 3. 

  ex)

  ```python
  # the new config inherits the base configs to highlight the necessary modification
  _base_ = './cascade_mask_rcnn_r50_fpn_1x_coco.py'
  
  # 1. dataset settings
  dataset_type = 'CocoDataset'
  classes = ('a', 'b', 'c', 'd', 'e') # 변경할 class 이름들 
  data = dict(
      samples_per_gpu=2,
      workers_per_gpu=2,
      train=dict(
          type=dataset_type,
          # explicitly add your class names to the field `classes`
          classes=classes, # 이런식으로 변경해야 한다. 
          ann_file='path/to/your/train/annotation_data',
          img_prefix='path/to/your/train/image_data'),
      val=dict(
          type=dataset_type,
          # explicitly add your class names to the field `classes`
          classes=classes,
          ann_file='path/to/your/val/annotation_data',
          img_prefix='path/to/your/val/image_data'),
      test=dict(
          type=dataset_type,
          # explicitly add your class names to the field `classes`
          classes=classes,
          ann_file='path/to/your/test/annotation_data',
          img_prefix='path/to/your/test/image_data'))
  ```

- **dataset wrapper는 데이터 분포를 변경하거나, 섞을 때 유용하다. (cutmix 등)**

  - `RepeatDataset`: simply repeat the whole dataset.

    ```python
    dataset_A_train = dict(
            type='RepeatDataset',
            times=N,
            dataset=dict(  # This is the original config of Dataset_A
                type='Dataset_A', # 내 dataset type (CocoDataset)
                ...
                pipeline=train_pipeline
            )
        )
    ```

  - `ClassBalancedDataset`: repeat dataset in a class balanced manner.

    > 이게 class unbalance된 데이터를 부족한 class에 대해서 반복해서 넣어주는 방법인 듯하다.
    >
    > [oversampling 기법 소개](https://wyatt37.tistory.com/10)

    ```python
    dataset_A_train = dict(
            type='ClassBalancedDataset', 
            oversample_thr=1e-3,
        	# 데이터 비율이 threshold 이하인 것을 oversampling해주는 듯?
        
            dataset=dict(  # This is the original config of Dataset_A
                type='Dataset_A',
                ...
                pipeline=train_pipeline
            )
        )
    ```

  - `ConcatDataset`: concat datasets.

    > 데이터 셋을 섞는 방법이다. 

    ex1)

    ```python
    dataset_A_train = dict(
        type='Dataset_A',
        ann_file = ['anno_file_1', 'anno_file_2'],
        # 두개의 dataset의 annotation 파일을 이렇게 넣어준다. 
        pipeline=train_pipeline
    )
    ```

    ```python
    # 애는 train인데 왜 eval을 넣는 건지 모르겠음
    # val_dataset은 아래 다른 예시로 있는데 둘의 차이는?
    dataset_A_train = dict(
        type='Dataset_A',
        ann_file = ['anno_file_1', 'anno_file_2'],
        separate_eval=False, # False로 두면 합쳐진 이미지로 eval
        pipeline=train_pipeline
    )
    ```

    ex2) 서로 다른 종류의 dataset concatenate

    ```python
    dataset_A_train = dict()
    dataset_B_train = dict()
    
    data = dict(
        imgs_per_gpu=2,
        workers_per_gpu=2,
        train = [
            dataset_A_train,
            dataset_B_train
        ],
        val = dataset_A_val,
        test = dataset_A_test
    )
    ```

    ex3) validation에서도 데이터를 섞고 싶을 때

    ```python
    dataset_A_val = dict()
    dataset_B_val = dict()
    
    data = dict(
        imgs_per_gpu=2,
        workers_per_gpu=2,
        train=dataset_A_train,
        val=dict(
            type='ConcatDataset',
            datasets=[dataset_A_val, dataset_B_val],
            separate_eval=False))
    ```

    ex4) 최고 빡쎈 방법으로 두개의 데이터셋을 각각 N,M번 반복한 걸 합치는 것

    ```python
    dataset_A_train = dict(
        type='RepeatDataset',
        times=N,
        dataset=dict(
            type='Dataset_A',
            ...
            pipeline=train_pipeline
        )
    )
    dataset_A_val = dict(
        ...
        pipeline=test_pipeline
    )
    dataset_A_test = dict(
        ...
        pipeline=test_pipeline
    )
    dataset_B_train = dict(
        type='RepeatDataset',
        times=M,
        dataset=dict(
            type='Dataset_B',
            ...
            pipeline=train_pipeline
        )
    )
    data = dict(
        imgs_per_gpu=2,
        workers_per_gpu=2,
        train = [
            dataset_A_train,
            dataset_B_train
        ],
        val = dataset_A_val,
        test = dataset_A_test
    )
    ```

- **비권장 사항**
  - validation 단계에서 concat (이게 제대로 동작할지 확실하지 않다고 하는 것 같음)
  - ClassBalancedDataset이랑 RepeatDataset도 evaluating을 지원하지 않는다고 함



# Customize Data Pipelines

![image-20220324125525140](../../../../AppData/Roaming/Typora/typora-user-images/image-20220324125525140.png)

- **transform 추가하기**

  ```python
  # my_pipeline.py
  
  import random
  from mmdet.datasets import PIPELINES
  
  
  @PIPELINES.register_module()
  class MyTransform:
      """Add your transform
  
      Args:
          p (float): Probability of shifts. Default 0.5.
      """
  
      def __init__(self, p=0.5):
          self.p = p
  
      def __call__(self, results):
          if random.random() > self.p:
              results['dummy'] = True
          return results
  ```

  ```python
  custom_imports = dict(imports=['path.to.my_pipeline'], allow_failed_imports=False)
  
  img_norm_cfg = dict(
      mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
  train_pipeline = [
      dict(type='LoadImageFromFile'),
      dict(type='LoadAnnotations', with_bbox=True),
      dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
      dict(type='RandomFlip', flip_ratio=0.5),
      dict(type='Normalize', **img_norm_cfg),
      dict(type='Pad', size_divisor=32),
      dict(type='MyTransform', p=0.2), # 이렇게 추가할 수 있다. 
      dict(type='DefaultFormatBundle'),
      dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
  ]
  ```

- **tools/misc/browse_dataset.py 이걸 쓰면 이미지 시각화/저장 가능**



# Customize runtime setting

- **lr_config를 통해 scheduler 변경 가능**

  ex) [여기](https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/lr_updater.py)에서 종류 더 찾아볼 수 있다. 

  ```python
  lr_config = dict(
      policy='CosineAnnealing', # 제공하는 종류가 많다.
      warmup='linear',
      warmup_iters=1000,
      warmup_ratio=1.0 / 10,
      min_lr_ratio=1e-5)
  ```

- **workflow는 train, val을 iter단위로, epoch 단위로 몇 번 할 건지 정할 수 있다.** 

  ex)

  ```python
  workflow = [('train', 1)]
  # [('train', 1), ('val', 1)] train 먼저
  # [('val', 1), ('train', 1)] val 먼저
  ```

- **customize hook이 가능하다**

  1. 먼저 mmdet/core/utils/__ init__.py을 아래 내용으로 수정

     ```python
     from .my_hook import MyHook
     ```

  2. config 파일에 아래 내용 추가

     ```python
     custom_imports = dict(imports=['mmdet.core.utils.my_hook'], allow_failed_imports=False)
     
     custom_hooks = [
         dict(type='MyHook', a=a_value, b=b_value, priority='NORMAL')
         # 우선순위도 지정가능
     ]
     ```

  ex)

  ```python
  # mmdet/core/utils/my_hook.py
  from mmcv.runner import HOOKS, Hook
  
  @HOOKS.register_module()
  class MyHook(Hook):
  
      def __init__(self, a, b):
          pass
  
      def before_run(self, runner):
          pass
  
      def after_run(self, runner):
          pass
  
      def before_epoch(self, runner):
          pass
  
      def after_epoch(self, runner):
          pass
  
      def before_iter(self, runner):
          pass
  
      def after_iter(self, runner):
          pass
  ```