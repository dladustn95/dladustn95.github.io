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

## 1 Introduction
&nbsp;&nbsp;RNN, 특히 LSTM과 GRU는 sequence modeling과 language modeling이나 machine translation 같은 transduction problem에서 state-of-the-art로 자리 잡았다. 
Recurrent 모델의 순차적인 특성은 학습에서 병렬화를 제한하는데 메모리의 제약으로 batch에도 제한이 생겨 sequence의 길이가 길어질수록 큰 문제가 된다. 
다른 연구에서 factorization trick이나 conditional computation을 통해 연산 효율을 향상시켰으나 sequence 연산의 기본 제약은 여전히 남아있다.  
Attention mechanism은 input, output sequence의 길이에 상관없이 dependency를 모델링 해줄 수 있어 sequence modeling이나 transduction model에서 중요한 요소가 되었다. 
그러나 대부분 RNN에 결합되어 사용된다.  
이 논문에서는 recurrence를 피하는 대신 input과 output 사이의 global dependecy를 찾는 attention mechanism만 사용하는 Transformer 구조를 제안한다. 
Transformer는 더 많은 병렬처리가 가능하며 state-of-the-art 수준을 보인다.    

## 2 Background
&nbsp;&nbsp;Sequential 연산을 줄이기 위해 Extended Neural GPU, ByteNet, ConvS2S 등 다양한 모델이 제안되었다. 
이 모델들은 CNN을 사용하여 모든 input, output 위치에 대한 hidden representation을 병렬로 계산한다. 
그러나 두 개의 임의의 input 또는 output을 연결하는데 필요한 number of operation은 거리에 따라 증가한다. 
따라서 모델이 distant position에 있는 dependency를 학습하는데 어려움이 있다. 
Transformer에서는 attention-weighted positions을 평균화함에 따라 number of operation이 일정한 수로 줄어든다. 
하지만 effective가 감소하는데 이는 Multi-Head Attention으로 극복한다.  
&nbsp;&nbsp;self-attention은 sequence의 representation을 계산하기 위해 단일 sequence의 서로 다른 위치들을 관련시키는 attention mechanism이다. 
self-attention은 reading comprehension, abstractive summarization, textual entailment, learning task-independent sentence representations 등 다양한 작업에서 사용된다.  
&nbsp;&nbsp;End-to-end memory network는 sequencealigned recurrence 대신 recurrent attention mechanism을 기반으로  simple-language question answering과 language modeling task에서 좋은 성능을 내었다.    

## 3 Model Architecture
&nbsp;&nbsp;뛰어난 성능을 가지는 neural sequence transduction model은 encoder-decoder 구조로 이루어져 있다. 
encoder는 input sequence $$(x_1,...,x_n)$$을 continuous representation인 $$\text{z} = (z_1,...,z_n)$$으로 변환한다. 
주어진 $$\text{z}$$에 대해 decoder는 output seqeunce $$(y_1,...,y_m)$$을 하나씩 생성한다. 
각각의 step에서 모델은 다음 symbol을 생성하기 위해 이전에 생성한 symbol을 추가적인 input으로 사용하는데 이를 auto-regressive라고 한다.  
Transformer는 아래 그림과 같이 self-attention과 point-wise fully connected layer를 쌓아 만든 encoder와 decoder로 구성되어 있다. 
![그림1](/assets/images/Transformer_figure1.png "그림1"){: .align-center}

### 3.1 Encoder and Decoder Stacks
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

### 3.2 Attention
&nbsp;&nbsp;Attention function은 query와 key-value 쌍을 output에 mapping 하는 것으로 설명할 수 있다. 
여기서 query, key, value, output은 모두 벡터로 이뤄진다. 
output은 weighted sum으로 계산되는데, weight는 해당 key와 query의 compatibility function으로 계산된다.  
![그림2](/assets/images/Transformer_figure2.png "그림2"){: .align-center}  

### 3.2.1 Scaled Dot-Product Attention
&nbsp;&nbsp;Scaled Dot-Product Attention의 구조는 위 그림의 왼쪽에 나타나있다. 
입력으로는 $$d_k$$ 차원의 query, key와 $$d_v$$ 차원의 value를 가진다. 
query와 key의 dot product를 계산하고 $$\sqrt{d_k}$$로 나눈 값에 softmax 연산을 적용해 value에 대한 weight를 얻는다.  
$$\text{Attention}(Q,K,V)=\text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$$  
가장 자주 쓰이는 attention function은 additive attention과 dot-product(multiplicative) attention이다. 
dot-product attention은 scaling factor $$\frac{1}{\sqrt{d_k}}$$만 제외하면 논문의 알고리즘과 동일하다. 
additive attention은 compatibility function을 단일 hidden layer의 feed-forward network를 사용해 계산한다. 
두 개의 theoretical complexity는 비슷하지만, matrix multiplication에 대한 최적화된 코드가 많기 때문에 dot-product attention이 더 빠르고 공간 효율성이 좋다.  
작은 값의 $$d_k$$에 대해서는 두 mechanism이 비슷하게 동작하지만 scaling이 없는 큰 $$d_k$$에 대해서는 additive attention이 더 좋은 성능을 보인다. 
큰 값의 $$d_k$$에 대해서는 dot product가 크게 증가하여 softmax 함수에서 gradient가 아주 작은 부분으로 가게 된다. 
이를 방지하기 위해 $$\frac{1}{\sqrt{d_k}}$$로 dot product에 scale을 해준다.    

### 3.2.2 Multi-Head Attention
&nbsp;&nbsp;$$d_{\text{model}}$$ 차원의 key, value, query로 하나의 attention을 계산하는 대신 query, key, value에 대해서 서로 다르게 학습된 $$d_k, d_k, d_v$$ 차원의 linearly project를 h번 수행하는 것이 더 효과적이다. 
project된 query, key, value에 대해 attention을 병렬적으로 계산해 $$d_v$$차원의 output을 산출한다. 
이 값들은 concatenate된 뒤에 다시 project 연산을 거쳐 최종 값을 얻게 된다. 
Multi-head attention을 사용해 다른 위치에 있는 representation subspace 정보를 attention 할 수 있다. 
Single attention head는 이것을 억제한다. 
이를 수식으로 표현하면 다음과 같다.  
$$\text{MultiHead}(Q,K,V)=\text{Concat}(\text{head}_1,...,\text{head}_h)W^O\\\text{where} \ \text{head}_i=\text{Attention}(QW_i^Q,KW_i^K,VW_i^V)$$  
여기서 parameter matrix는 다음과 같다.  
$$W_i^Q\in \Bbb R^{d_{\text{model}}\times d_k}, W_i^K\in \Bbb R^{d_{\text{model}}\times d_k}, W_i^V\in \Bbb R^{d_{\text{model}}\times d_v}$$  
이 논문에서는 $$h=8$$개의 parallel attention layer(head)를 사용했다. 
각 벡터는 다음과 같이 나눠진다. $$d_k=d_v=d_{\text{model}}/h=64$$ 
각 head에서 사용하는 dimension이 줄어들었기 때문에 총 연산량은 single-head attention과 비슷하다.    

### 3.2.3 Applications of Attention in our Model
&nbsp;&nbsp;Transformer는 multi-head attention을 다음과 같은 방법으로 사용했다. 
+ "encoder-decoder attention" layer에서 query는 이전 decoder layer에서 오고, key, value는 encoder의 output에서 온다. 
decoder의 모든 position에서 input sequence(encoder output)의 모든 position에 attention을 적용할 수 있도록 한다. 
+ encoder는 self-attention layer를 포함한다. 
self-attention layer에서 모든 key, value, query는 encoder의 이전 layer output에서 온다. 
encoder의 각 position에서 encoder의 이전 layer에 대한 모든 position에 대해 attention을 적용할 수 있다. 
+ decoder의 self-attention layer에서는 decoder의 각 position에 대해 그 position 이전까지의 position에 대해서만 attention을 적용할 수 있도록 한다. 
auto-regressive한 속성을 유지하기 위해 scaled dot-product attention에 masking out을 적용했다. 
masking out은 $$i$$번째 position에 대한 attention을 계산할 때 $$i+1$$번째 이후의 모든 position은 softmax의 input을 $$−\infty$$로 설정하여 attention을 얻지 못하도록 하는 것이다.  

### 3.3 Position-wise Feed-Forward Networks
&nbsp;&nbsp;encoder, decoder의 각 attention sub-layer는 fully connected feed-forward network를 포함한다. 
각 position 마다 독립적으로 적용되므로 postion-wise다. 
다음과 같이 두 개의 선형 변환과 ReLU activation function으로 구성되어 있다.  
$$\text{FFN}(x)=\text{max}(0,xW_1+b_1)W_2+b_2$$  
$$x$$에 선형 변환을 적용한 후 ReLU를 거쳐 다시 선형 변환을 한다. 
선형 변환은 position마다 같은 parameter를 사용하지만 layer가 달라지면 다른 parameter를 사용한다. 
input과 output의 차원은 $$d_{\text{model}}=512$$이고 inner-layer의 차원은 $$d_{ff}=2048$$이다.    

### 3.4 Embeddings and Softmax
&nbsp;&nbsp;다른 sequence transduction model과 유사하게 input과 output token을 $$d_{text{model}}$$차원을 갖는 벡터로 변환하여 embedding을 학습한다. 
일반적으로 학습된 선형 변환과 softmax합수를 사용해 decoder의 output을 다음 token을 예측하기 위한 확률로 변환한다. 
논문의 모델에서는 두 embedding layer와 softmax 이전의 linear transformation에서 동일한 weight 행렬을 공유한다. 
embedding layer에서는 weight에 $$\sqrt{d_{\text{model}}}$$을 곱한다.    

### 3.5 Positional Encoding
&nbsp;&nbsp;Transformer 모델은 recurrence나 convolution을 포함하지 않는다. 
따라서 sequence의 순서성을 이용하기 위해서는 position에 대한 정보를 sequence의 token에 주입하여야 한다. 
이를 위해 encoder와 decoder의 input embedding에 "positional encoding"을 더해준다. 
positional encoding은 embedding과 더해질 수 있게 $$d_{text{model}}$$로 같은 차원을 갖는다. 
논문에서는 다른 frequency를 갖ㄴ는 sine, cosine 함수를 사용했다.  
$$PE_{(pos,2i)}=sin(pos/10000^{2i/d_{\text{model}}})\\PE_{(pos,2i+1)}=cos(pos/10000^{2i/d_{\text{model}}})$$  
$$pos$$는 position, $$i$$는 차원을 의미한다. 
이 함수를 통해 모델이 relative position에 대해 attention을 더 쉽게 학습할 수 있다. 
고정된 offset $$k$$에 대해서 $$PE_{pos+k}$$는 $$PE_{pos}$$의 선형 변환으로 나타낼 수 있다.    

## 4 Why Self-Attention
&nbsp;&nbsp;self-attention을 사용하면 얻는 이점으로는 3가지가 있다. 
첫번재로 layer의 계산 복잡도가 감소한다. 두번째는 더 많은 양을 병렬 처리할 수 있다. 
마지막으로 긴 거리의 dependency를 잘 학습할 수 있다. 
추가적으로 self-attention은 더 해석 가능한 모델을 만든다.    

## 5 Training
### 5.1 Training Data and Batching
&nbsp;&nbsp;4.5M개의 문장 쌍이 있는 WMT 2014 English-German dataset을 사용했다. 
36M개의 문장 쌍을 포함하는 WMT 2014 English-French dataset도 사용했다. 
각 batch마다 약 25000개의 source token과 target token을 포함하는 문장 쌍을 가지도록 하였다.    

### 5.3 Optimizer
