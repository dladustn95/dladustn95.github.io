---
title:  "Convolutional 2D Knowledge Graph Embeddings 논문 리뷰"
excerpt: "Convolutional 2D Knowledge Graph Embeddings 논문 리뷰"

mathjax: true
categories:
  - NLP
tags:
  - NLP RelationPrediction LinkPrediction ConvE
last_modified_at: 2019-11-11T11:11:00+09:00
---

### 서론
 지식그래프(Knowledge graphs)는 지식베이스(Knowledge bases)에서 관계(relation)를 edge로 개체(entity)를 node로 바꾸어 만든 것이다. 
지식그래프는 search, analytics, recommendation, data integration에서 중요한 자원이다. 
그러나 그래프의 link가 누락되어 있는 불완전성으로 인해 사용에 어려움이 있다. 
예를 들어 DBpedia의 Freebase는 66%의 사람 개체의 출생지가 연결되어 있지 않다. 
이러한 link의 누락을 식별하는 것을 link prediction이라 부른다. 
지식그래프는 수백만개의 사실을 포함할 수 있다. 
따라서 link predictor는 파라미터의 수와 컴퓨팅 비용을 현실적으로 관리하는 방식으로 확장해야한다.  
 이러한 문제를 해결하기 위해 link prediction 모델은 주로 제한된 수의 매개변수를 사용하고, 임베딩 공간에서의 행렬 곱같은 간단한 연산으로 구성되었다. 
DistMult는 parameter당 하나의 feature를 생성하는 embedding parameter간의 3방향 상호작용이 특징이다. 
빠르고 간단한 얕은 모델을 사용해 덜 표현적인 feature를 배우는 대신 큰 지식그래프로 확장할 수 있다.  
 얕은 모델의 feature 수를 늘리는 방법(표현력을 높이는 방법)은 embedding 사이즈를 증가시키는 것뿐이다. 
그러나 이러한 방법은 더 큰 지식그래프로 확장되지는 않는데, embedding parameter의 수는 그래프의 개체와 관계 수에 비례하기 때문이다. 
예를 들어 DistMult와 같은 얕은 모델의 embedding 사이즈가 200이라 가정하고 Freebase에 적용하면 이것의 parameter 용량은 33GB이다. 
embedding 사이즈와 관계 없이 feature의 수를 늘리려면 feature의 여러 레이어를 사용해야한다. 
그러나 feature가 완전연결 레이어로 이뤄진 이전의 다층 지식그래프 임베딩구조는 overfit되기 쉬웠다.
얕은 구조의 크기 조절 문제와 완전연결된 깊은 구조의 overfit문제를 해결하기 위해서는 parameter를 효율적으로 사용하고, Deep network로 구성된 빠른 연산자가 필요하다.  
 컴퓨터비전에서 주로 사용되는 convolution 연산자는 위의 속성을 만족한다. 
게다가 다층 convolution 신경망을 학습할때 overfitting을 제어할 수 있는 강력한 방법론이 존재한다.  
 이 논문에서는 지식그래프에서 embedding위에서 2D convolution 연산을 사용해 지식그래프의 손실된 link를 예측하는 모델, ConvE를 소개한다. 
ConvE는 단일 convolution layer, embedding dimension에 대한 projection layer, 내부 product layer로 구성된 간단한 다층 convolution 구조이다.  
+ 간단하지만 경쟁력있는 2D convolutional link prediction 모델 ConvE를 소개한다. 
+ 3배 빠른 훈련과 300배 빠른 평가를 가능하게 하는 1-N scoring 기법 개발.
+ parameter 효율적인 모델을 만듬, DistMult와 R-GCNs에 비해 각각 8배, 17배 적은 parameter로 FB15k-237에서 더 나은 점수를 달성.
+ ConvE와 다른 얕은 모델간의 성능 차이가 indegre와 PageRank로 측정한 지식그래프의 복잡도에 비례하여 증가한다.
+ 주로 사용되는 link prediction dataset의 inverse relation으로 인한 평가의 어려움을 조사하였고 inverse relation을 제거한 버전의 dataset을 소개하여 간단한 규칙 기반 모델로 해결할 수 없도록 함.
+ ConvE와 이전에 제안된 모델을 위에서 소개한 dataset으로 평가한 결과 MRR에서 state-of-the-art를 달성    

### 관련 연구
 여러 link prediction 모델이 제안되었다. Translating Embedding 모델(TransE), Bilinear Diagonal 모델(DistMult)등. 
본 논문의 모델과 가장 유사한 모델은 Holographic Embedding 모델(HolE)이다. HolE는 cross-correlation을 사용한다. 
그러나 HolE는 비선형 feature의 여러 layer를 학습하지 않아서 이론적으로 ConvE보다 표현력이 떨어진다.
 ConvE는 link prediction에서 2D convolutional layer를 사용한 최초의 모델이다. 
*Graph Convolutional Networks*(GCNs)는 convolution 연산자가 그래프의 locality 정보를 생성하기 위해 사용되었다. 
그러나 GCN framework는 undirected 그래프에 한정되어 있고, 지식그래프는 directed 그래프이다.  
 여러 convolutional 모델들이 여러 자연어처리 task 해결을 위해 사용되고 있다. 
그러나 대부분의 연구들은 embedding 공간 안에서 단어의 순서와 같이 embedding의 시간적 순서에 작동하는 1D convolution을 사용한다. 
이 논문에는 embedding에 대해 공간적으로 작용하는 2D convolution을 사용한다.    

### Number of Interactions for 1D vs 2D Convolutions
 1D convolution 대신 2D convolution을 사용하면 embedding 사이의 추가 상호 작용 지점을 통해 모델의 표현성이 향상된다. 
예를 들어 $$( a^2 )$$