# ImportError: libGL.so.1: cannot open shared object file: No such file or directory

- **Albumentations를 설치할 때 생기는 오류** 

  opencv error라고 하는데 해결하기 위해서는 우선

  ```
  apt-get update
  apt-get intsall ffmpeg libsm6 libxext6 -y
  ```

  위와 같이 커맨드를 입력하면 해결되는 경우도 있지만, 아래와 같은 에러가 또 뜰 수 있다. 

- **ImportError: cannot import name '_registerMatType' from 'cv2.cv2'**

  이 경우

  ```
  pip install "opencv-python-headless<4.3"
  ```

  위 커맨드를 입력하면 해결된다. 

[Ref]

https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo

https://github.com/opencv/opencv-python/issues/591