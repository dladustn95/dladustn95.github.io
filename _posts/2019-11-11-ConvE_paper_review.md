---
title:  "Convolutional 2D Knowledge Graph Embeddings 논문 리뷰"
excerpt: "Convolutional 2D Knowledge Graph Embeddings 논문 리뷰"

categories:
  - NLP
tags:
  - NLP RelationPrediction LinkPrediction ConvE
last_modified_at: 2019-11-11T11:11:00+09:00
---

### 서론
 Knowledge graphs는 Knowledge bases에서 관계를 edge로 개체를 node로 바꾸어 만든 것이다. 
KG는 search, analytics, recommendation, data integration에서 중요한 자원이다. 
그러나 그래프의 link가 누락되어 있는 불완전성으로 인해 사용에 어려움이 있다. 
예를 들어 DBpedia의 Freebase는 66%의 사람 개체의 출생지가 연결되어 있지 않다. 
이러한 link의 누락을 식별하는 것을 link prediction이라 부른다. 
지식그래프는 수백만개의 사실을 포함할 수 있다. 
따라서 link predictor는 파라미터의 수와 컴퓨팅 비용을 현실적으로 관리하는 방식으로 확장해야한다.  
 이러한 문제를 해결하기 위해 link prediction 모델은 주로 제한된 수의 매개변수를 사용하고, 임베딩 공간에서의 행렬 곱같은 간단한 연산으로 구성되었다. 
DistMult는 
