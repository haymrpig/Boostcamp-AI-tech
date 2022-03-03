# RuntimeError: unexpected EOF, expected 898 more bytes. The file might be corrupted.

- **timm model에서 pretrained weight 가져올 때 네트워크가 끊기는 등 한번 잘 못 가져오면 다음에도 오류가 생기는 것 같다.**

  ```
  os.environ['TORCH_HOME'] = 'weight파일 폴더 경로'
  ```

  

  