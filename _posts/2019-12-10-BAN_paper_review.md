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
여러가지 목적으로 모델의 knowledge transfer를 위해 많은 연구가 진행되었다. 
압축을 위해서 더 많은 공간을 차지하거나 예측을 위해 더 많은 연산을 필요로 하는 큰 모델의 정확도를 유지하는 compact 모델을 만들었다. 
최근에는 transparency 또는 interpretability를 높이기 위해 decision tree나 generalized additive model 같은 단순한 모델로 근사화하여 knowledge transfer하는 방법을 제안했다. 
또한 의사 결정을 설명하기 위해 deep network를 decision tree로 distilling 할 것을 제안했다. 
모델을 압축하려는 논문들 중에서 knowledge transfer의 목표는 간단하다. 
직접 훈련하는 것보다 teacher model을 통한 knowledge transfer로 인해 더 나은 성능을 가지는 student model을 생성하는 것이다.    

## Born-Again Networks
