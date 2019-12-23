---
title:  "Born-Again Neural Networks 논문 리뷰"
excerpt: "Born-Again Neural Networks 논문 리뷰"

mathjax: true
tags:
  - [Knowledge Distillation, Deep learning, BAN]
date: 2019-12-10T18:00:00+09:00
last_modified_at: 2019-12-10T13:44:00+09:00
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
BAN은 teacher model의 output distribution $$f(x,\theta_1^*)$$에 포함된 정보가 풍부한 training signal source를 제공하여 더 좋은 일반화 기능을 갖는 second solution $$f(x,\theta_2^*),\theta_2\in\Theata_2$$로 이어질 수 있다는 KD에서 입증된 아이디어를 이용한다. 
새로운 model의 출력과 original model의 출력 사이의 cross-entropy를 기반으로 하는 KD term을 사용해 original loss function을 수정하거나 대체, 정규화 하는 기술을 연구한다.  
$$\mathcal{L}(f(x,\text{arg}_{\theta_1}\text{min}\mathcal{L}(y,f(x,\theta_1))),f(x,\theta_2))$$  
기존의 KD와 다르게 teacher와 student network가 identical 구조를 갖는 경우를 다룬다. 
또한 teacher와 student network가 비슷한 용량을 갖지만 다른 구조를 갖는 경우를 다루는 실험을 제시한다. 
예를 들어 DensNet teacher에서 유사한 수의 prameter를 갖는 ResNet student로의 knowledge transfer를 수행한다.    

### Sequence of Teaching Selves Born-Again Networks Ensemble
CIFAR100에 대한 SGDR WideResnet과 Coupled-DenseNet ensemble의 놀라운 성과에 영감을 받아 여러 세대의 knowledge transfer를 통해 BANs를 순차적으로 학습시킨다. 
각각의 경우에 k-1번째 student에서 knowledge transfer를 통해 k번째 model이 학습된다.  
$$\mathcal{L}(f(x,\text{arg}_{\theta_{k-1}}\text{min}\mathcal{L}(y,f(x,\theta_{k-1}))),f(x,\theta_k))$$  
마지막으로 BAN의 모든 결과값을 평균내어 Born-Again Network Ensembles (BANE)을 만든다.  
$$\hat{f}^k(x)=\sum_{i=1}^kf(x,\theta_i)/k$$    

### Dark Knowledge Under the Light
KD의 저자는 KD의 성공이 output 카테고리의 유사성에 대한 정보를 전달하는 wrong response의 logit 분포에 숨겨진 dark knowledge에 의한 것이라고 주장했다. 
Knowledge distillation은 