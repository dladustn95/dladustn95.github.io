---
title:  "Adaptive Convolution for Multi-Relational Learning 논문 리뷰"
excerpt: "Adaptive Convolution for Multi-Relational Learning 논문 리뷰"

mathjax: true
categories:
  - NLP
tags:
  - [NLP, RelationPrediction, LinkPrediction, KBAT]
date: 2019-11-16T15:40:00+09:00
last_modified_at: 2019-11-19T17:18:00+09:00
---

# Learning Attention-based Embeddings for Relation Prediction in Knowledge Graphs
+ KBAT는 지식 그래프에 attention embedding을 적용한 모델이다. 

## Introduction
&nbsp;&nbsp;Relation prediction(knowledge base completion)에서 state-of-the-art 모델은 대부분 *knoledge embedding* 기반 모델이다. 
크게 *translational* 모델과 *convolutional neural network* 모델로 나눌 수 있다. 
translational model은 간단한 연산과 제한된 parameter를 사용해 임베딩을 학습한다. 
따라서 낮은 수준의 임베딩을 생성한다. 
반대로 CNN 기반 모델은 parameter의 효율성과 더 복잡한 관계를 고려하는 특성 때문에 더 풍부한 임베딩을 학습한다. 
그러나 두 종류의 모델 모두 각각의 트리플을 독립적으로 고려하기 때문에 주어진 개체 근처에 내재된 관계를 포착하지 못하는 단점이 있다.  
&nbsp;&nbsp;앞서 언급한 관찰에 의거하여 이 논문에서는 relation prediction을 위한 일반화된 attention 기반 그래프 임베딩 기법을 제안한다. 
node classification에서 *graph attention networks*(GATs)는 그래프의 가장 관련성있는 부분인 1-hop 이웃의 node feature에 집중하였다. 
이 논문은 attention mechanism을 일반화하고 확장시켜 주어진 개체(node)의 multi-hop 이웃 안에 있는 개체(node)와 관계(edge) 모두에 attention을 적용한다.  
&nbsp;&nbsp;논문에서 제시한 아이디어는 다음과 같다.

1. 주어진 node 주변의 multi-hop 관계를 포착한다.
2. 여러 관계에서 개체가 수행하는 역할의 다양성을 캡슐화한다. 
3. 의미적으로 유사한 관계 클러스터에 존재하는 기존 지식을 통합한다.

그러나 모델이 깊어질수록 거리가 먼 개체의 영향은 기하급수적으로 줄어든다. 
이를 해결하기 위해 n-hop 이웃 사이에 auxiliary edge를 도입하여 개체 간 지식 흐름을 용이하게 한다. 
논문에서는 새로운  *generalized graph attention model*과 *ConvKB*가 encoder, decoder로 동작하는 encoder-decoder 모델을 제안한다.  
&nbsp;&nbsp;이 논문은 지식그래프에서 relation prediction을 위한 graph attention based embedding을 최초로 제안한다. 
graph attention mechanism 일반화하고 확장하여 주어진 개체의 multi-hop 이웃에 존재하는 개체와 관계 feature를 동시에 포착한다. 
모델을 다양한 실제 dataset을 사용해 까다로운 relation prediction task에 대해 평가한다.    

## Related Work
&nbsp;&nbsp;최근에 relation prediction을 위한 다양한 지식 그래프 임베딩이 제안되고 있다. 
이것들은 크게 compositional, translational, CNN based, graph based 모델로 나눌 수 있다. 
앞의 세 종류의 모델은 생략하고 graph based 모델에 대해서만 설명하려 한다. 
graph based 모델 중 R-GCN이 존재한다. 
이것은 *graph convolutional networks*(GCNs)를 관계 data에 대해서도 적용하도록 확장한 것이다. 
개체의 이웃에 대해서 convolution 연산을 적용하고 이를 같은 weight로 할당한다. 
그러나 CNN based 모델보다 성능이 좋지 않다.  
&nbsp;&nbsp;주어진 예시들은 개체 feature만 집중하거나 개체와 관계를 분리하여 지식그래프 embedding을 학습한다. 
논문이 제안한 모델은 multi-hop과 주어진 개체의 n-hop 이웃에서 의미적으로 유사한 관계를 전체적으로 포착한다.    

## Our Approach
### Graph Attention Networks (GATs)
&nbsp;&nbsp;Graph convolutional networks (GCNs)는 개체의 이웃에서 정보를 모으고 이를 동일한 가중치로 전달한다. 
GATs는 GCN과 달리 node의 이웃에 대해 중요도를 다르게 할당하여 학습한다. 
&nbsp;&nbsp;layer로 node의 입력 집합은 $$\text{x}= \{ \overrightarrow{x}_1,\overrightarrow{x}_2,\dotsb ,\overrightarrow{x}_N\}$$이다. 
layer는 node feature의 집합을 벡터 $$\text{x}^\prime= \{ \overrightarrow{x}^\prime_1,\overrightarrow{x}^\prime_2,\dotsb ,\overrightarrow{x}^\prime_N\}$$로 변환한다. 
여기서 $$\overrightarrow{x}_i, \overrightarrow{x}^\prime_i$$는 개체 $$e_i$$의 입력과 출력 embedding이고 $$N$$은 개체(node)의 수이다. 
하나의 GAT layer는 $$e_{ij}=a(\text{W}\overrightarrow{x}_i,\text{W}\overrightarrow{x}_j)$$로 표현할 수 있다. 
$$e_{ij}$$는 그래프 $$\mathcal{G}$$에 있는 edge $$(e_i,e_j)$$의 attention value이다. 
$$\text{W}$$는 입력 feature를 높은 차원의 출력 feature로 선형변환하기 위한 parameter이다. 
$$a$$는 임의로 선택한 attention function이다.  
&nbsp;&nbsp;각 edge가 갖는 attention value는 source node $$e_i$$에 대한 edge$$(e_i,e_j)^\prime$$ feature의 중요도이다. 
relative attention $$\alpha_{ij}$$는 모든 이웃 값에 대해서 softmax function을 사용해 계산한다. 
아래는 layer의 출력을 나타낸다.  
$$\overrightarrow{x}^\prime_i=\sigma(\sum_{j\in\mathcal{N}_i}\alpha_{ij}\text{W}\overrightarrow{x}_j)$$  
GAT은 학습 과정을 안정화하기 위해 multi-head attention을 사용한다. 
K개의 attention head를 concatenate하는 multihead attention process는 다음과 같이 수행된다. 
$$\overrightarrow{x}^\prime_i=\|^K_{k=1}\sigma(\sum_{j\in\mathcal{N}_i}\alpha_{ij}^k\text{W}^k\overrightarrow{x}_j)$$  
$$\|$$는 concatenation을 의미하고, $$\sigma$$는 non-linear function을 나타낸다. 
$$\alpha_{ij}^k$$는 k번째 attention mechanism에서 계산된 edge $$(e_i,e_j)$$의 attention 계수를 나타낸다. 
$$\text{W}^k$$는 k번째 attention mechanism에서 linear transformation matrix를 나타낸다. 
마지막 layer에서 embedding 출력은 평균을 사용하여 multi-head attention을 얻는다.  
$$\overrightarrow{x}^\prime_i=\sigma(\frac{1}{K}\sum_{k=1}^K\sum_{j\in\mathcal{N}_i}\alpha_{ij}^k\text{W}^k\overrightarrow{x}_j)$$
  
### Relations are Important
&nbsp;&nbsp;GATs는 지식그래프에서 중요한 부분인 관계(edge) feature를 무시하기 때문에 부적절하다. 
지식그래프에서 개체는 그들이 속한 relation에 따라 다양한 역할을 가진다. 
![그림1](/assets/images/KBAT_figure1.png "그림1"){: .align-center}
예를 들어 그림 1에서 개체 "Christopher Nolan"은 "brother", "director"의 역할로 두개의 다른 트리플에서 나타난다. 
attention mechanism에서 관계와 이웃 node feature를 모두 사용하는 새로운 embedding 접근을 제안한다. 
모델에서 building block이 되는 single attention layer를 정의한다. 
GAT와 유사하게 이 framework는 attention mechanism의 특정 선택과 무관하다.  
&nbsp;&nbsp;모델의 각 layer에서는 입력으로 두개의 embedding matrix를 가진다. 
entity embedding은 행렬 $$\text{H}\in\Bbb R^{N_e \times T}$$로 표현된다. 
i번째 행은 개체 $$e_i$$의 embedding이고, $$N_e$$는 개체의 수, $$T$$는 각 개체 embedding의 feature 차원이다. 
이와 비슷하게 관계 embedding 또한 행렬 $$\text{G}\in\Bbb R^{N_r \times P}$$로 표현된다. 
layer는 $$\text{H}^\prime\in\Bbb R^{N_e \times T^\prime},\ \text{G}^\prime\in\Bbb R^{N_r \times P^\prime}$$을 출력한다.  
&nbsp;&nbsp;개체 $$e_i$$에 대한 새로운 embbeding을 얻기 위해 $$e_i$$가 속한 모든 트리플 표현을 학습한다. 
트리플 $$t^k_{ij}=(e_i,r_k,e_j)$$에 해당하는 개체와 관계 feature 벡터를 concatenate하고 선형 변환을 거쳐 이 embedding을 구한다.  
$$\overrightarrow{c}_{ijk} = \text{W}_1 |\overrightarrow{h_i}| |\overrightarrow{h_j}| |\overrightarrow{g_k}|$$  
$$\overrightarrow{c}_{ijk}$$는 트리플 $$t_{ij}^k$$의 벡터 표현이다. 
벡터 $$\overrightarrow{h_i}, \overrightarrow{h_j}, \overrightarrow{g_k}$$는 각각 개체와 관계 embedding $$e_i, e_j, r_k$$를 나타낸다. 
$$\text{W}_1$$은 선형 변환 행렬을 나타낸다. 
각각의 트리플 $$t^k_{ij}$$의 중요도를 학습한 값을 $$b_{ijk}$$로 나타낸다. 
weight matrix $$\text{W}_2$$를 parameter로 갖는 선형 변환을 수행하고, LeakryRelu의 non-linearity를 사용해 absolute attention value를 얻는다.  
$$b_{ijk}=\text{LeakyReLU}(\text{W}_2c_{ijk})$$  
![그림2](/assets/images/KBAT_figure2.png "그림2"){: .align-center}
relative attention value를 얻기 위해 softmax를 $$b_{ijk}$$에 적용한다. 
그림 2는 하나의 트리플에 대한 relative attention value $$\alpha_{ijk}$$를 얻는 과정을 보여준다.  
$$\alpha_{ijk}=\text{softmax}_{jk}(b_{ijk})\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ 
  \\=\frac{\text{exp}(b_{ijk})}{\sum_{n\in\mathcal{N}_i}\sum_{r\in\mathcal{R}_{in}}\text{exp}(b_{inr})}$$  
$$\mathcal{N}_i$$는 $$e_i$$의 이웃을 의미하고 $$\mathcal{R}_{in}$$는 $$e_i, e_j$$와 연결된 관계 집합을 의미한다. 
$$e_i$$의 새로운 embedding은 각각 트리플의 표현에 relative attention value를 곱한 값을 모두 합쳐 얻을 수 있다.  
$$\overrightarrow{h_i^\prime}=\sigma\left(\sum_{j\in\mathcal{N}_i} \sum_{k\in\mathcal{R}_{ij}} \alpha_{ijk} \overrightarrow{c}_{ijk}\right)$$
multi-head attnetion은 학습 과정을 안정화하고, 이웃으로부터 더 많은 정보를 캡슐화한다. 
M개의 독립된 attention mechanism은 embedding을 계산하고 concatenate한다.  
$$\overrightarrow{h_i^\prime}=\|^M_{m=1}\sigma\left(\sum_{j\in\mathcal{N}_i}  \alpha_{ijk}^m c_{ijk}^m)\right)$$
