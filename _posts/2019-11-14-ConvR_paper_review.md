---
title:  "Adaptive Convolution for Multi-Relational Learning 논문 리뷰"
excerpt: "Adaptive Convolution for Multi-Relational Learning 논문 리뷰"

mathjax: true
categories:
  - NLP
tags:
  - [NLP, RelationPrediction, LinkPrediction, ConvR]
date: 2019-11-14T14:24:00+09:00
last_modified_at: 2019-11-14T14:24:00+09:00
---

# Adaptive Convolution for Multi-Relational Learning
+ [ConvR 논문](https://www.aclweb.org/anthology/N19-1103/)에 대한 해석 및 정리입니다.
+ 

## Introduction
&nbsp;&nbsp;ConvE는 입력 개체와 관계의 상호작용을 포착하기에는 불충분하다. 
ConvE는 입력 개체와 관계를 단순히 결합해 convolution layer의 입력으로 사용했다. 
또한 global filter를 사용하는 2D convolution은 concatenate line에서만 입력 개체와 관계의 상호작용을 얻을 수 있다. 

| ![그림1](/assets/images/ConvR_figure1.png "그림1"){: .align-center} |
|:---:|
| 그림1: ConvE와 ConvR이 reshape, convolution 된다. 오렌지색은 개체, 파란색은 관계를 나타낸다. 흰색 블럭은 입력 개체와 관계에 적용되는 global filter이다. |
  
그림 1(a)는 3x3 크기의 두 행렬을 reshape, stack하여 convolution layer에 입력한다. 
입력을 2x2 크기의 global filter로 연산하면 두 행렬이 인접한 부분에서만 상호작용이 일어난다. 
이로 인해 output convolutional features의 아주 작은 부분에서만 개체와 관계의 상호작용을 포착한다. 
대다수의 개체와 관계는 독립적으로 계산된다. 
(그림 1(a)에서는 파란색과 오렌지색으로 빗금쳐진 20% 범위에서만 일어난다.)  
&nbsp;&nbsp;이 논문에서는 입력 개체와 관계 사이의 상호작용을 최대화하는데 초점을 두었다. 
그림1(b)는 ConvR의 핵심 아이디어를 나타낸다. 
이 모델의 핵심 아이디어는 관계에 대해 적응적으로(adaptively) 구성된 필터를 사용하여 개체의 convolution을 용이하게 하는 것이다. 
관계로 구성된 filter를 사용하는 adaptive convolution으로 개체의 상호작용을 많이 포착하도록 한다. 
트리플이 주어지면 주어 개체를 나타내는 벡터는 convolution layer의 입력으로, 관계를 나타내는 벡터는 filter로 설정된다. 
이 방식은 생성된 모든 feature가 개체와 관계의 상호작용을 포착할 수 있다(그림 1(b)에서 파란색과 오렌지색으로 빗금쳐진 부분). 
이 convolution feature는 마지막으로 목적어 개체와 projected, matched 된다.  
&nbsp;&nbsp;Adapdive convolution을 사용하면 parameter 측면에서 효율적인 모델을 구성할 수 있다. 
그림 1의 두 모델을 비교하면 ConvR은 global filter가 필요하지 않고, ConvE의 것보다 반 정도 작은 feature map을 생성한다. 
adaptive convolution 아이디어는 2D convolution에 국한되지 않고, 1D 또는 3D로 쉽게 확장 가능하다. 
다만 ConvE와의 비교를 용이하게 하기 위해 이 논문은 2D convolution을 사용해 ConvR의 우수성을 보인다.    

## Background
앞에서 작성한 ConvE 논문의 Related Work와 큰 차이가 없어 생략하겠습니다.  
[ConvE 논문](https://arxiv.org/abs/1707.01476)
[ConvE 논문 리뷰 Related Work](https://dladustn95.github.io/nlp/ConvE_paper_review)    

## Adaptive Convolution on Multi-relational Data
### The ConvR model
주어진 트리플 $$(s,r,o)$$에 대해 ConvR은 두 개체 $$s,o$$를 벡터 $$\text{s,o}\in \mathcal(R)^{d_e}$$, 관계 $$r$$을 벡터 $$\text{r} \in \mathcal(R)^{d_r}로 변환했다. 
