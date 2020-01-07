---
title:  "Born-Again Neural Networks 논문 리뷰"
excerpt: "Born-Again Neural Networks 논문 리뷰"

mathjax: true
tags:
  - [Knowledge Distillation, Deep learning, BAN]
date: 2019-12-10T18:00:00+09:00
last_modified_at: 2019-12-30T13:44:00+09:00
---

#Born-Again Neural Networks

## Introduction
&nbsp;&nbsp;"Statistical Modeling: The Two Culture"에서는 서로 다른 stochastic, algorithmic procedure은 다양한 모델을 비슷한 validation 성능을 갖게 만들 수 있다. 
더해서 이러한 모델을 각 구성 모델보다 더 우수한 성능을 갖는 앙상블로 구성할 수 있다고 언급했다. 
앙상블이 강력함에도 불구하고 그 앙상블을 모방하고 비슷한 성능을 달성하는 더 단순한 (앙상블의 구성 요소 중 하나보다 더 복잡하지 않은) 모델을 종종 발견 할 수 있다. 
이전에 "Born-Again Trees"에서 이러한 아이디어를 처음 제시했다. 
multiple-tree predictor의 성능을 달성하는 single tree 학습이 존재한다는 것이다. 
Born-again Trees는 앙상블과 유사하지만 개별 decision tree의 속성인 해석에 대한 편의성을 추가로 제공한다. 
이후에 "model compression", "knowledge distillation"이라는 이름의 비슷한 아이디어가 발표되었다. 
둘 모두 고성능을 가진 high-capacity teacher가 compact student에게 knowledge transfer하는 방식이다. 
또한 student가 data를 직접 학습할 때는 teacher와 성능이 같아질 수 없지만 distillation process를 통해 student는 teacher의 성능과 가까워질 수 있게 된다. 
teacher가 동일한 용량을 가진 student에게 knowledge transfer하는 실험에서 student가 master가 되어 teacher의 성능을 큰 차이로 능가했다. 
본 논문은 "Sequence of Teaching Selves"와 유사하게 간단한 re-training procedure를 사용한다. 
teacher model이 수렴한 후 새로운 student model을 초기화한 다음 정확한 label을 예측하고 teacher의 output distribution과 일치하도록 훈련을 진행한다. 
이러한 student를 Born-Again Networks(BANs)라고 정의했다.    

## Related Literature
### Knowledge Distillation
&nbsp;&nbsp;여러가지 목적으로 모델의 knowledge transfer를 위해 많은 연구가 진행되었다. 
압축을 위해서 더 많은 공간을 차지하거나 예측을 위해 더 많은 연산을 필요로 하는 큰 모델의 정확도를 유지하는 compact 모델을 만들었다. 
최근에는 transparency 또는 interpretability를 높이기 위해 decision tree나 generalized additive model 같은 단순한 모델로 근사화하여 knowledge transfer하는 방법을 제안했다. 
또한 의사 결정을 설명하기 위해 deep network를 decision tree로 distilling 할 것을 제안했다. 
모델을 압축하려는 논문들 중에서 knowledge transfer의 목표는 간단하다. 
직접 훈련하는 것보다 teacher model을 통한 knowledge transfer로 인해 더 나은 성능을 가지는 student model을 생성하는 것이다.    

## Born-Again Networks
&nbsp;&nbsp;classical image classification setting에서 우리는 이미지와 label의 tuple $$(x,y) \in \mathcal{X}\times\mathcal{Y}$$로 구성된 training dataset을 갖고, 처음 보는 data에 대해 일반화 할 수 있도록 함수 $$f(x): \mathcal{X}\mapsto\mathcal{Y}$$를 최적화 한다. 
일반적으로 $$f(x)$$를 mapping하는 것은 신경망 $$f(x,\theta_1)$$에 의해 parameterized 된다. 
여기서 $$\theta_1$$은 어떤 공간 $$\Theta_1$$의 parameter이다. 
Empirical Risk Minimization (ERM)를 사용해 parameter를 학습하고 아래의 loss function을 최소화하는 model $$\theta_1^*$$을 만든다.  
$$\theta_1^*=\text{arg}_{\theta_1}\text{min}\mathcal{L}(y,f(x,\theta_1))$$  
Stochastic Gradient Descent (SGD)를 사용해 최적화한다.  
&nbsp;&nbsp;Born-Again Networks (BANs)은 knowledge distillation 또는 model compression 논문에서 입증된 loss function을 수정하여 generalization error를 줄일 수 있다는 경험적 발견에 기초하고 있다. 
BAN은 teacher model의 output distribution $$f(x,\theta_1^*)$$에 포함된 정보가 풍부한 training signal source를 제공하여 더 좋은 일반화 기능을 갖는 second solution $$f(x,\theta_2^*),\theta_2\in\Theta_2$$로 이어질 수 있다는 KD에서 입증된 아이디어를 이용한다. 
새로운 model의 출력과 original model의 출력 사이의 cross-entropy를 기반으로 하는 KD term을 사용해 original loss function을 수정하거나 대체, 정규화 하는 기술을 연구한다.  
$$\mathcal{L}(f(x,\text{arg}_{\theta_1}\text{min}\mathcal{L}(y,f(x,\theta_1))),f(x,\theta_2))$$  
기존의 KD와 다르게 teacher와 student network가 identical 구조를 갖는 경우를 다룬다. 
또한 teacher와 student network가 비슷한 용량을 갖지만 다른 구조를 갖는 경우를 다루는 실험을 제시한다. 
예를 들어 DensNet teacher에서 유사한 수의 prameter를 갖는 ResNet student로의 knowledge transfer를 수행한다.    

### Sequence of Teaching Selves Born-Again Networks Ensemble
&nbsp;&nbsp;CIFAR100에 대한 SGDR WideResnet과 Coupled-DenseNet ensemble의 놀라운 성과에 영감을 받아 여러 세대의 knowledge transfer를 통해 BANs를 순차적으로 학습시킨다. 
각각의 경우에 k-1번째 student에서 knowledge transfer를 통해 k번째 model이 학습된다.  
$$\mathcal{L}(f(x,\text{arg}_{\theta_{k-1}}\text{min}\mathcal{L}(y,f(x,\theta_{k-1}))),f(x,\theta_k))$$  
마지막으로 BAN의 모든 결과값을 평균내어 Born-Again Network Ensembles (BANE)을 만든다.  
$$\hat{f}^k(x)=\sum_{i=1}^kf(x,\theta_i)/k$$    

### Dark Knowledge Under the Light
&nbsp;&nbsp;Hinton은 KD의 성공이 output 카테고리의 유사성에 대한 정보를 전달하는 wrong response의 logit 분포에 숨겨진 dark knowledge에 의한 것이라고 주장했다. 
distillation과 normal supervised training의 correct class에 대응되는 ouput node를 통해 흐르는 gradient를 비교하는 것으로 dark knowledge를 설명할 수도 있다. 
Knowledge distillation은 정확한 예측에 대한 teacher의 confidence에 해당되는 importance-weighting과 유사하다.
논문에서는 dark knowledge의 성공이 teacher의 nonargmax outputs에 포함된 정보에 의한 것인지, dark knowledge가 그저 importance weighting의 역할만 하는 것인지 알아보기 위해서 두가지 처리를 진행했다. 
첫번째 처리는 Confidence Weighted by Teacher Max (CWTM)이다. 
student loss function 안에 있는 example에 teacher model의 confidence에 의해 weight를 준다. 
BAN model을 다음 식의 근사를 이용해 훈련시킨다.  
$$\sum_{s=1}^b\frac{w_s}{\sum_{u=1}^bw_u}(q_{\ast,s}-y_{\ast,s})=\sum_{s=1}^b\frac{p_{\ast,s}}{\sum_{u=1}^bp_{\ast,u}}(q_{\ast,s}-y_{\ast,s})$$  
위 식에서 정답 $$p_{\ast,s}$$를 teacher $$\text{max}p_{.,s}$$의 max output으로 대체한다. 
$$\sum_{s=1}^b\frac{\text{max}\ p_{.,s}}{\sum_{u=1}^b\text{max}\ p_{.,s}}(q_{\ast,s}-y_{\ast,s})$$  
두번째로 dark knowledge with Permuted Predictions (DKPP)는 teacher의 예측 분포에 대한 non-argmax output을 permute한다.  
$$\sum_{s=1}^b\sum_{i=1}^n\frac{\partial\mathcal{L}_{i,s}}{\partial z_{i,s}}=\sum_{s=1}^b(q_{\ast,s}-p_{\ast,s})+\sum_{s=1}^b\sum_{i=1}^{n-1}(q_{i,s}-p_{i,s})$$  
위 식에서 $$\ast$$를 $$\text{max}$$로 대체하고, teacher dimension의 dark knowledge 부분을 permute하여 아래와 같이 표현한다.  
$$\sum_{s=1}^b\sum_{i=1}^n\frac{\partial\mathcal{L}_{i,s}}{\partial z_{i,s}}=\sum_{s=1}^b(q_{\ast,s}-\text{max}p_{.,s})\ +\sum_{s=1}^b\sum_{i=1}^{n-1}q_{i,s}-\phi(p_{j,s})$$  
위 식에서 $$phi(p_{j,s})$$는 teacher의 permuted output이다. 
DKPP에서는 dark knowledge의 정확한 attribution을 각각의 non-argmax output dimension으로 뿌리고 original output 공분산 행렬의 쌍방향 유사성을 파괴한다.    

### BANs Stability to Depth and Width Variations
&nbsp;&nbsp;DenseNet은 depth, growth, compression으로 parametrized 된다. 
depth는 dense block의 수에 해당된다. 
growth는 각각의 새로운 dense block에서 연결되는 새로운 feature의 수를 정의한다. 
compression은 각 stage의 끝에서 감소되는 feature의 수를 결정한다.  
&nbsp;&nbsp;hyper-parameter의 변형은 parameter의 수, 메모리 사용, 각 pass에 대한 순차 작업의 수 사이의 trade-off를 일으킨다. 
논문에서는 다른 hyperparameter를 갖는 구조로 DenseNet teacher와 같은 기능을 표현할 가능성에 대해 실험했다. 
공정한 비교를 위해 각 spatial transition에서의 output dimensionality가 DenseNet-90-60 teacher와 일치하는 DenseNet을 만든다. 
hidden state의 크기를 일정하게 유지하면서 block의 수를 선택하여 간접적으로 growth factor를 조정한다. 
추가적으로 spatial transition의 전후에 compression factor를 줄임으로써 growth factor를 대폭 줄일 수 있다.    

### DenseNets Born-Again as ResNets
BAN-DenseNet은 여러 parameter를 사용하여 plain DenseNet과 동일한 수준으로 작동한다. 
논문에서는 BAN procedure가 ResNet을 향상시킬 수 있는지 실험한다. 
ResNet teacher 대신 DenseNet-90-60을 teacher로 사용한다. 
그리고 Dense Block을 Wide Residual Block과 Bottleneck Residual Block으로 전환하여 유사한 ResNet student를 구성한다.    

## Experiments
모든 실험에서 CIFAR-100 dataset을 사용했으며 Mean-Std normalization을 제외하고 *Wide-ResNet*과 같은 전처리 과정, training setting을 사용했다. 
정규화를 위해 KD loss, weight decay, WideResNet drop-out을 사용했다.    

### 4.1. CIFAR-10/100
**Baselines**  
기존 architecture의 prohibitive memory usage 없이 강한 teacher baseline을 얻기 위해 DenseNet의 여러 height, growth factor를 탐색하였다. 
논문에서는 growth factor를 높이고 original paper에서 가장 큰 구성과 비슷한 parameter 수를 갖는, 상대적으로 얕은 architecture에서 좋은 구성을 찾는다. 
Classical ResNet baselines은 *Wide residual networks*에 따라 train된다. 
최종적으로 DenseNet teacher 실험을 통한 BAN-ResNet의 baseline으로 각 block에서 DenseNet-90-60의 output shape와 일치하는 Wide-ResNet과 bottleneck-ResNet을 구성한다.  
**BAN-DenseNet and ResNet**  
수렴 후 teacher network를 훈련하는데 사용된 것과 동일한 training schedule을 사용해 BAN retraining을 수행한다. 
DenseNet-(116-33, 90-60, 80-80, 80-120)을 사용했고, 각 설정에 대한 BAN train을 시행하였다. 
두세개의 BAN을 ensemble한 성능을 시험하였다. 
BANs를 학습시키기위한 다른 knowledge transfer 기법도 조사하였다. 
구체적으로 BAN을 teacher와 더 유사해지도록 점진적으로 제한하고, student와 teacher가 첫번째와 마지막 layer를 공유하거나 student와 teacher간 activation의 L2 거리에 따른 페널티를 주는 loss를 추가한다. 
그러나 cross entropy를 통해 이러한 변화들이 간단한 KD보다 약간만 좋지 않게 동작한다는 것을 찾았다. 
ResNet teacher를 사용하는 BAN-ResNet 실험은 Wide-ResNet-(28-1, 28-2, 28-5, 28-10)을 사용한다.  
**BAN without Dark Knowledge**  
CWTM에서 argmax dimension을 제외한 모든 teacher output의 영향을 받지 않도록 했다. 
이를 위해 sample의 중요도에 따라 가중치가 부여되는 일반적인 label loss를 student에게 학습시킨다. 
각 sample에 대한 teacher output의 최댓값을 importance weight로 해석하고 이를 사용하여 student loss에 대한 각 sample을 재조정하는데 사용한다.  
DKPP에서는 teacher output이 전체적으로 높은 순간을 유지하지만 argmax를 제외한 각 output dimension을 무작위로 permute한다. 
나머지 training scheme와 architecture는 변경하지 않고 유지한다.  
두 방법 모두 output 사이의 공분산을 변경하여 어떠한 개선이라도 고전적인 dark knowledge의 결과로 보지 않도록 한다.  
**BAN-Resnet with DenseNet teacher**  
모든 DenseNet teacher를 사용한 BAN-ResNet 실험에서 student는 teacher의 첫번째와 마지막 layer를 공유했다. 
성공적인 Wide-ResNet28의 깊이부터 시작하여 block당 하나의 residual unit이 남을 때까지 unit의 수를 감소시키면서 ResNet의 복잡성을 조절한다. 
각 block당 channel의 수는 모든 residual unit에 대해 동일하므로, spatial down-sampling 이전에 1x1 convolution 후 dense block output의 비율과 일치 시킨다. 
논문에서는 주로 1의 비율로 architecture를 찾지만 network의 너비를 절반으로 줄이는 효과도 보인다.  
**BAN-DenseNet with ResNet teacher**  
이 실험에서는 ResNet teacher가 DenseNet-90-60 students를 성공적으로 훈련시킬 수 있는지 본다. 
우리는 여러 설정을 갖는 Wide-ResNet teacher를 사용하고 다른 DenseNet 실험과 동일한 hyper parameter를 갖는 Ban-DenseNet student를 훈련시킨다.    

### 4.2. Penn Tree Bank
BAN framework가 computer vision에만 국한되지 않는다는 것을 검증하기 위해 language model에 적용시켜 Penn Tree Bank (PTB) dataset으로 평가해보았다. 
논문에서는 두개의 BAN language model을 사용한다. 
하나는 single layer LSTM, 다른 하나는 2-layer LSTM(CNN-LSTM)이다.  
LSTM model에서는 *Using the output embedding to improve language models*의 weight와 65% dropout을 사용하고 mini-batch 크기가 32인 SGD를 사용하여 40 epoch 학습시켰다. 
초기 learning rate는 1에서 시작하여 epoch 이후에 validation perplexity가 감소하지 않으면 0.25배 된다.  
CNN-LSTM은 40epoch, mini-batch 크기가 20인 SGD로 훈련된다. 
초기 learning rate는 2에서 시작하여 적어도 0.5 epoch 이후에 validation perplexity가 감소하지 않으면 0.5배 된다.    

## Results
BAN student model이 teacher model보다 더 향상되었다.    

### 5.1. CIFAR-10
아래 표에서 알 수 있듯 CIFAR-10의 test error는 동일한 teacher에게 훈련받은 Wide-ResNet과 DenseNet student에서 보두 낮거나 같아졌다. 
BAN-DenseNet이 다른 복잡성의 architecture 간 격차를 빠르게 줄여 parameter와 error rate에 대한 implicit gain을 얻는 방법에 주목할 필요가 있다.  

| Network | Parameters | Teacher | BAN |
|:----|:---:|:---:|:---:|
| Wide-ResNet-28-1 | 0.38 M | 6.69 | **6.64** |
| Wide-ResNet-28-2 | 1.48 M | 5.06 | **4.86** |
| Wide-ResNet-28-5 | 9.16 M | 4.13 | **4.03** |
| Wide-ResNet-28-10 | 36 M | **3.77** | 3.86 |
| DenseNet-112-33 | 6.3 M | 3.84 | **3.61** |
| DenseNet-90-60 | 16.1 M | 3.81 | **3.5** |
| DenseNet-80-80 | 22.4 M | **3.48** | 3.49 |
| DenseNet-80-120 | 50.4 M | **3.37** | 3.54 |

### 5.2. CIFAR-100
CIFAR-100에서는 모든 BAN-DenseNet model이 향상되었다. 
따라서 born-again 현상을 이해하고 조사하기위해 이 dataset에서 대부분의 실험을 진행하였다.    

**BAN-DenseNet and BAN-ResNet**  
![그림1](/assets/images/BAN_figure1.png "그림1"){: .align-center}  
위 그림에서 label과 teacher output(BAN+L)을 사용하거나 후자(BAN)만을 사용하여 test error rate를 조사했다. 
label supervision을 완전히 제거하는 개선은 양식 전반에 걸쳐 체계적이며
가장 작은 student인 BAN-DenseNet-112-33이 6.5M개의 parameter만으로 16.95%의 error를 달성한 반면  DenseNet-80-120 teacher는 8배 많은 parameter를 사용해 16.87%의 error를 기록했다.  
아래 그림에서는 하나를 제외한 모든 Wide-ResNnet student가 그들의 teacher보다 향상되었다.  
![그림2](/assets/images/BAN_figure2.png "그림2"){: .align-center}    

**Sequence of Teaching Selves**  
BAN을 여러 세대 학습 시키는 것은 몇세대 후 포화상태에 도달하여 일관성을 잃게 하지만 성능이 향상된다. 
BAN-DenseNet-80-80의 3번째 세대(BAN-3)는 CIFAR-100에서 가장 좋은 single model이다(그림1 참조).    

**BAN-Ensemble**  
가장 큰 ensemble 모델인 BAN-3-DenseNet-BC-80-120은 14.9%의 error를 기록하여 같은 setting에서 가장 좋은 성능을 보였다. 

**Effect of non-argmax Logits**  
