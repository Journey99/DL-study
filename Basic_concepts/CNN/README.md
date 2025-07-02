# 📍CNN이란?
CNN(Convolutional Neural Network)은 딥러닝에서 주로 이미지나 영상 데이터를 처리할 때 쓰인다. 이름에서 알 수 있듯이 Convolution(합성곱)을 사용하는 네트워크다. CNN은 이미지의 공간정보를 유지한 상태로 학습이 가능한 모델이다.

## Convolution 작동원리
### (1) Convolution
하나의 합성곱 계층에는 입력되는 이미지의 채널 개수만큼 필터가 존재하며, 각 채널에 할당된 필터를 적용함으로써 합성곱 계층의 출력 이미지가 생성된다. 

예를 들어, 높이X너비X채널이 4X4X1인 텐서 형태의 입력 이미지에 대해 3X3 크기의 필터를 적용하는 합성곱 계층에서는 아래 그림과 같이 이미지와 필터에 대한 합성곱 연산을 통해 2X2X1 텐서 형태의 이미지가 생성된다.
![cnn-1](./IMG/cnn-1.png)

### (2) Stride
이미지에 대해 필터를 적용할 때는 필터의 이동량을 의미하는 스트라이드(stride)를 설정해야한다. 아래 그림은 동일한 입력 이미지와 필터에 대해 스트라이드 값에 따른 출력 이미지의 변화를 보여준다.
왼쪽이 stride=1 인 경우이고 오른쪽이 stride=2 인 경우이다.
![cnn-2](./IMG/cnn-2.png)

### (3) Padding
Convolution을 위에서 처럼 수행하게 되면 4X4에서 2X2으로 차원이 축소되면서 데이터 손실이 발생한 것을 알 수 있다. filter가 input image의 끝에 도달해버려서 결과적으로 input image와 filter 간의 크기 차이만큼의 차원이 손실되는 것이다.

이러한 문제점을 해결하기 위해 이용되는 것이 패딩 (padding)이다.

패딩은 아래 그림과 같이 입력 이미지의 가장자리에 특정 값으로 설정된 픽셀들을 추가함으로써 입력 이미지와 출력 이미지의 크기를 같거나 비슷하게 만드는 역할을 수행한다. 이미지의 가장자리에 0의 값을 갖는 픽셀을 추가하는 것을 zero-padding이라고 하며, CNN에서는 주로 이러한 zero-padding이 이용된다.

아래 그림의 오른쪽처럼 1폭짜리 zero-padding을 적용하였더니 출력 이미지 크기가 입력 이미지 크기와 같이 4x4 로 유지된다.
![cnn-3](./IMG/cnn-3.png)

## CNN 종류
CNN은 특정 모델을 한정하는 것이 아니다. Convolution layer와 Pooling layer의 조합으로 만들어진 모델을 통틀어 CNN 모델이라고 할 수 있다. 즉, 블랙박스 모델인(결과값을 설명할 수 없음) layers들을 Conv와 Pooling의 조합을 이것저것 만들어 보면서 최적을 결과 값이 나올때, 이름을 붙여 발표를 한다고 할 수 있다.

예를 들어 CNN의 모델은 다음과 같다.
- LeNet-5 1998
- AlexNet - 2012
- VGG - 2014
- Network in Network - 2014
- ResNets
- Inception-v3
- GAN


# 📍CNN 구조
![cnn-0](./IMG/cnn-0.png)
CNN은 위 이미지와 같이 이미지의 특징을 추출하는 부분과 클래스를 분류하는 부분으로 나눌 수 있다. 
- 특징 추출 : Convolution Layer + Pooling Layer(선택적)
- 이미지 분류 : Fully Connected Layer

각 Layer의 동작방법은 아래와 같다.


## (1) Convolution Layer
Convolutional layer는 간단하게 convolution과정과 활성 함수(activation function)를 거치는 과정이다.

여기서 여러 개의 filter(kernel)을 이용하여 다수의 convolution 결과값 도출이 가능하다.
![cnn-4](./IMG/cnn-4.png)

## (2) Pooling layer
Convolutional layer를 거치고 나면 다수의 결과값이 생성되어 너무 많은 데이터를 다루게 되는 문제점이 생긴다. 이를 해결하기 위한 방법으로 pooling layer를 거치게 된다.

pooling layer는 각 결과값(feature map)의 차원을 축소해 적당히 크기도 줄이고, 특정 feature를 강조할 수 있게 해준다.

pooling layer에는 크게 두 가지 방법으로 Max pooling과 Average pooling방법이 있다.
- Max pooling : feature map을 특정 크기로 잘라낸 후, 그 안에서 가장 큰 값을 뽑아내는 방법
- Average pooling : feature map을 특정크기로 잘라낸 후, 평균을 취해 뽑아내는 방법
![cnn-5](./IMG/cnn-5.png)

CNN에서 Pooling layer 를 이용함으로써 다음과 같은 이점들을 얻을 수 있다.
- 선택 영역 내부에서는 픽셀들이 이동 및 회전 등에 의해 위치가 변경되더라도 출력값은 동일하다. 따라서 이미지를 구성하는 요소들의 이동 및 회전 등에 의해 CNN의 출력값이 영향을 받는 문제를 완화할 수 있다.
- CNN이 처리해야하는 이미지의 크기가 크게 줄어들기 때문에 파라미터 또한 감소한다. Pooling layer를 이용함으로써 학습 시간을 절약할 수 있으며 오버피팅 문제 또한 완화할 수 있다.

## (3) Fully Connected Layer
이미지 특징을 위의 두 단계를 통해 추출했으면 이것이 무엇을 의미하는 데이터인지를 분류하는 작업이 필요하다. Fully Connected Layer가 그 작업을 진행한다. 
모든 처리를 거친 이미지 데이터를 1D array로 변환하여(flatten) softmax 함수를 적용할 수 있게끔 변환한다.
![cnn-6](./IMG/cnn-6.png)

# 📍 추가
CNN과정을 통해 사용되는 하이퍼파라미터는 다음과 같다.
- Convolution layers: 필터의 갯수, 필터의 크기, stride값, zero-padding의 유무
- Pooling layers: Pooling방식 선택(MaxPool or AvgPool), Pool의 크기, Pool stride 값(overlapping)
- Fully-connected layers: 넓이(width)
- 활성 함수 : ActivationReLU(가장 주로 사용되는 함수), SoftMax(multi class classification), Sigmoid(binary classification)
- Loss function: Cross-entropy for classification, L1 or L2 for regression
- 최적화(Optimization) 알고리즘과 이것에 대한 hyperparameter(보통 learning rate): SGD(Stochastic gradient descent), SGD with momentum, AdaGrad, RMSprop, Adam
- Random initialization: Gaussian or uniform, Scaling