## 블로그에서 찾아본 책상 분석

### 1. scikit-image 라이브러리
scikit-image는 이미지 처리하기 위한 파이썬 라이브러리이다.<br>
장점 : 오픈 소스라 사용하기 쉬움.<br>
단점 : 일반적으로 얇은 책이나 책에 직전이 있을때 구분이 잘되지 않음. 실제 책의 수와 결과의 수가 차이가 심함. 파라미터를 일반화 시키기 어려움.<br>
정리 : 책이 아닌 바둑판이나 간단한 경우에는 활용하기 좋지만, 책의 경우엔 오차가 심하여 사용이 불가능함.

Documentation : https://scikit-image.org/

기본적으로 이미지를 처리 할 수 있는 다양한 기능들을 제공한다.<br>
skimage는 numpy배열로 구성되어 쉽게 사용이 가능하다. <br>
기본적인 이미지처리에 사용되는 그리기, 색상 변경, 회전등을 지원하고 책장 분석에선 윤곽석을 활용을 하였다.<br>

아래의 링크에서 설명에는 Canny를 통해 선을 검출하는데 굵은 책들은 결과가 좋게 보이지만 <br>
책이 얇거나 책에 선이 있는 경우에는 결과가 존재 않다는 것을 확인 할 수 있다.<br>
![참고이미지](https://velog.velcdn.com/images%2Fsuminwooo%2Fpost%2F25be6f9b-02be-42c3-ac2c-5ae205b54b7a%2Fimage.png "참고이미지")

```python
from skimage import feature
from skimage.feature import canny
import matplotlib.pyplot as plt
import cv2

img_ori = cv2.imread(####) # 데이터 경로 입력
img_down_sampling = cv2.pyrDown(img_ori) # 다운 샘플링
gray_scale = cv2.cvtColor(img_down_sampling, cv2.COLOR_BGR2GRAY)
edges1 = feature.canny(gray_scale, sigma=0.7) # sigma가 hyper parameter 
plt.imshow(edges1)
plt.axis('off')

```
실제 데이터를 활용해보았을때, 여러가지 문제점이 존재한다.
(이미지1은 결과를 정리하지 않은 상태이며, 이미지 2,3은 정리를 해준 상태이다.)
![이미지1](https://images.velog.io/images/suminwooo/post/97cc84bb-960c-4a7c-99d8-89cee822e80c/image.png "이미지1")
문제점:
1 . 책 옆면 이미지 or 한줄로 구성되었을 경우
(이미지 1처럼 책의 중간부분이 잘리게 되고 나중에 후처리 하기도 까다로워짐.)
2. 얇은 책 혹은 같은 책 시리즈의 경우엔 두권이 하나로 인식된다. -> 같은 시리즈일경우는 큰 문제가 존재하지 않지만 서로 다른 책일 경우 추후에 검색에서 막히게 된다.


[참고]https://jessicastringham.net/2018/12/01/books/

### Mask-RCNN 활용
2021년도 vision관련 논문들이 많이 출시되고 있다. 그중 가장 많이 쓰이는 모델이 YoLo인데 활용한 논문이 공개되었다.
![논문 속 figure](https://velog.velcdn.com/images%2Fsuminwooo%2Fpost%2Ff64baab3-62b5-4e6f-8087-43d84bb57c91%2Fimage.png "논문figure")

논문에 관한 파라미터와 사용한 데이터, 결과 등을 Research of YOLO Architectue Models in Book Detection(https://www.researchgate.net/publication/346820617_Research_of_YOLO_Architecture_Models_in_Book_Detection)

**Model**
YOLO CONVOLUTIONAL NETWORK 활용
1. YOLOv3(3 outputs)
2. YOLOv3(2 outputs)
3. Tiny YOLOv3

**data**
- 500개 이미지(5,245권 책 포함)<br>
- augmentation : such as scaling, rotation, flip, cropping distortion of colours and so on.<br>
- train 60%, valid 10%, test 30%

**netword parameter**

800 epochs<br>
size of input image – 448x448x3;<br>
optimizer – Adam;<br>
the learning rate – 0,0001;<br>
batch-size – 2 samples;<br>
metric for assessing the correctness of the class definition – AP.<br>

**result**

![YOLO 결과](https://velog.velcdn.com/images%2Fsuminwooo%2Fpost%2F0f434bc9-6ce1-4423-bd47-b8962ca8c589%2Fimage.png "결과")

### 2. CNN 활용
