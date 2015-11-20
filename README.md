# pylibfreenect2

[libfreenect2](https://github.com/OpenKinect/libfreenect2) の python ラッパーです。

できること：

- RGB image transfer
- IR and depth image transfer
- registration of RGB and depth images


## インストール

### libfreenect2

pylibfreenect2を使う前に、 [libfreenect2](https://github.com/OpenKinect/libfreenect2) を事前にインストールする

cloneしたあと、

```
mkdir -p build && cd build
cmake .. -DENABLE_CXX11=ON
make 
make install
```

### pylibfreenect2

cloneしたあと、

```
python setup.py develop
```

もしくは、

```
python setup.py install
```

注意：要numpy and cython

その他、何かパッケージが足りないと言われてしまったら、condaかpipでインストールしてください

## 動作確認

kinect v2が接続されていることを確認して、

```
python examples/multiframe_listener.py
```

color, ir, depth が表示される

libfreenect2の動作確認には、libfreenect2に付属のProtonectを使用してください
