---
title:  "Attention Is All You Need 논문 리뷰"
excerpt: "Attention Is All You Need 논문 리뷰"

mathjax: true
categories:
  - NLP
tags:
  - [NLP, Attention, Transformer]
date: 2019-11-26T13:44:00+09:00
last_modified_at: 2019-11-26T13:44:00+09:00
---

# Attention is all you need
+
+
+

## Introduction
&nbsp;&nbsp;RNN, 특히 LSTM과 GRU는 sequence modeling과 language modeling이나 machine translation 같은 transduction problem에서 state-of-the-art로 자리 잡았다. 
Recurrent 모델의 순차적인 특성은 학습에서 병렬화를 제한하는데 메모리의 제약으로 batch에도 제한이 생겨 sequence의 길이가 길어질수록 큰 문제가 된다. 
다른 연구에서 factorization trick이나 conditional computation을 통해 연산 효율을 향상시켰으나 sequence 연산의 기본 제약은 여전히 남아있다.  
Attention mechanism은 input, output sequence의 길이에 상관없이 dependency를 모델링 해줄 수 있어 sequence modeling이나 transduction model에서 중요한 요소가 되었다. 
그러나 대부분 RNN에 결합되어 사용된다.  
이 논문에서는 recurrence를 피하는 대신 input과 output 사이의 global dependecy를 찾는 attention mechanism만 사용하는 Transformer 구조를 제안한다. 
Transformer는 더 많은 병렬처리가 가능하며 state-of-the-art 수준을 보인다.    

## Background
&nbsp;&nbsp;Sequential 연산을 줄이기 위해 Extended Neural GPU, ByteNet, ConvS2S 등 다양한 모델이 제안되었다. 
이 모델들은 CNN을 사용하여 모든 input, output 위치에 대한 hidden representation을 병렬로 계산한다. 
그러나 두 개의 임의의 input 또는 output을 연결하는데 필요한 number of operation은 거리에 따라 증가한다. 
따라서 모델이 distant position에 있는 dependency를 학습하는데 어려움이 있다. 
Transformer에서는 attention-weighted positions을 평균화함에 따라 number of operation이 일정한 수로 줄어든다. 
하지만 effective가 감소하는데 이는 Multi-Head Attention으로 극복한다.  
&nbsp;&nbsp;self-attention은 sequence의 representation을 계산하기 위해 단일 sequence의 서로 다른 위치들을 관련시키는 attention mechanism이다. 
self-attention은 reading comprehension, abstractive summarization, textual entailment, learning task-independent sentence representations 등 다양한 작업에서 사용된다.  
&nbsp;&nbsp;End-to-end memory network는 sequencealigned recurrence 대신 recurrent attention mechanism을 기반으로  simple-language question answering과 language modeling task에서 좋은 성능을 내었다.    

## Model Architecture
&nbsp;&nbsp;뛰어난 성능을 가지는 neural sequence transduction model은 encoder-decoder 구조로 이루어져 있다. 
encoder는 input sequence $$(x_1,...,x_n)$$을 continuous representation인 $$\text{z} = (z_1,...,z_n)$$으로 변환한다. 
주어진 $$\text{z}$$에 대해 decoder는 output seqeunce $$(y_1,...,y_m)$$을 하나씩 생성한다. 
각각의 step에서 모델은 다음 symbol을 생성하기 위해 이전에 생성한 symbol을 추가적인 input으로 사용하는데 이를 auto-regressive라고 한다.  
Transformer는 아래 그림과 같이 self-attention과 point-wise fully connected layer를 쌓아 만든 encoder와 decoder로 구성되어 있다. 
![그림1](/assets/images/Transformer_figure1.png "그림1"){: .align-center}

###  Encoder and Decoder Stacks
**Encoder:** encoder는 $$N=6$$개의 동일한 layer로 구성되어있다. 
각 layer는 multi-head self-attention mechanism과 간단한 position-wise fully connected feed-forward로 구성된 두개의 sub-layer를 갖는다. 
각 sublayer에 다음과 같이 $$\text{LayerNorm}(x+\text{Sublayer}(x))$$ residual connection을 사용한다. 
이를 적용하기 위해 sublayer의 output dimension과 embedding dimension이 동일해야하는데 여기서는 512로 설정하였다. 
그 후 layer normalization을 적용한다.  
**Decoder:** decoder 또한 $$N=6$$개의 동일한 layer로 구성되어있다. 
decoder는 encoder에 존재하는 두 개의 sub-layer와 추가로 encoder stack의 output에 대한 multi-head attention을 수행하는 세번째 sub-layer를 가진다. (그림에서 가운데 sub-layer) 
encoder와 유사하게 각 sublayer에 residual connection을 사용하고 layer normalization을 적용한다. 
decoder의 self-attention sub-layer에서는 현재 위치보다 뒤에 있는 요소에 attention을 적용하지 못하도록 masking을 추가해준다. 
position $$i$$에 대한 예측을 위해 $$i$$ 이전에 있는 정보들만 사용하도록 하는 것이다.    

### Attention
&nbsp;&nbsp;Attention function은 query와 key-value 쌍을 output에 mapping 하는 것으로 설명할 수 있다. 
여기서 query, key, value, output은 모두 벡터로 이뤄진다. 
output은 weighted sum으로 계산되는데, weight는 해당 key와 query의 compatibility function으로 계산된다.  
![그림2](/assets/images/Transformer_figure2.png "그림2"){: .align-center}  

#### Scaled Dot-Product Attention
&nbsp;&nbsp;Scaled Dot-Product Attention의 구조는 위 그림의 왼쪽에 나타나있다. 
입력으로는 $$d_k$$ 차원의 query, key와 $$d_v$$ 차원의 value를 가진다. 
query와 key의 dot product를 계산하고 $$\sqrt{d_k}$$로 나눈 값에 softmax 연산을 적용해 value에 대한 weight를 얻는다.  
$$\text{Attention}(Q,K,V)=\text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$$  
가장 자주 쓰이는 attention function은 additive attention과 dot-product(multiplicative) attention이다. 
dot-product attention은 scaling factor $$\frac{1}{\sqrt{d_k}}$$만 제외하면 논문의 알고리즘과 동일하다. 
additive attention은 compatibility function을 단일 hidden layer의 feed-forward network를 사용해 계산한다. 
두 개의 theoretical complexity는 비슷하지만, matrix multiplication에 대한 최적화된 코드가 많기 때문에 dot-product attention이 더 빠르고 공간 효율성이 좋다.  
작은 값의 $$d_k$$에 대해서는 두 mechanism이 비슷하게 동작하지만 scaling이 없는 큰 $$d_k$$에 대해서는 additive attention이 더 좋은 성능을 보인다. 
큰 값의 $$d_k$$에 대해서 dot product가 크게 증가하여 softmax 함수에서 gradient가 아주 작은 부분으로 가게 된다. 
이를 방지하기 위해 $$\frac{1}{\sqrt{d_k}}$$로 dot product에 scale을 해준다. 