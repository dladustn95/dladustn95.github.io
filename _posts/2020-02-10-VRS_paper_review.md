---
title:  "Select and Attend: Towards Controllable Content Selection in Text Generation 논문 리뷰"
excerpt: "Select and Attend: Towards Controllable Content Selection in Text Generation 논문 리뷰"

mathjax: true
tags:
  - [DeepLearning, TextGeneration, NLP]
date: 2020-02-10T18:00:00+09:00
last_modified_at: 2020-02-10T18:00:00+09:00
---

# Select and Attend: Towards Controllable Content Selection in Text Generation

## 1. Introduction
&nbsp;&nbsp;data-to-text, summarization, image captioning과 같은 text generation task는 두 개의 과정으로 나눌 수 있다. 
content selection과 surface realization이다. 
generaion은 두가지 수준의 다양성을 가져야 한다. 

1) content-level diversity는 content selection의 다양한 가능성을 의미한다. (무엇을 말할 것인가) 
2) surface-level diversity는 선택한 내용을 말로 표현하기 위한 언어적 다양성을 의미한다. (어떻게 말할 것인가) 

최근의 neural network 모델들은 위 두가지 과정을 black-box로 처리하는 encoder-decoder 구조를 통해 task를 해결한다. 
따라서 두가지 다양성들은 생성 과정에서 얽혀있다. 
그러나 이러한 얽힘은 controllability와 interpretability을 희생시켜서 content가 생성되는 텍스트에 전달되었는지 판별하기 어렵게 만든다.    
&nbsp;&nbsp;이 논문에서는 Enc-Dec famework에서 content selection을 분리하여 generation을 보다 세세하게 제어할 수 있도록 한다. 
아래 그림은 그 예를 보여준다. content selection을 수정하여 다양한 관점으로 생성하거나, 여러가지 구문으로 sampling 할 수 있다. 

![그림1](/assets/images/VRS_figure1.png "그림1"){: .align-center} 

&nbsp;&nbsp;Enc-Dec에서 content selection을 사용하기 위한 연구들이 있었지만 모두 위의 문제를 완벽히 해결하지 못했다. 
현재 방법은 세가지로 분류할 수 있고 각각의 한계를 지닌다. 

1) **Bottom-up:** source token에 대한 attention을 제한하기 위해 별도의 content selector를 학습시키지만 분리되어 교육된 selector와 generator는 통합할 때 불일치가 발생할 수 있다. 
2) **Soft-select:** 불필요한 정보를 거르는 soft mask filter를 학습한다. 그러나 mask가 확률적 변동이 없이 결정론적이므로 content-level diversity를 만들기 어렵다. 
3) **Reinforce-select:** 강화 학습으로 selector를 학습한다. training variance가 높고, content selection에서 낮은 다양성을 가진다. 

이 논문에서는 content selection을 latent variable로 취급하고 Amortized variational inference로 학습한다. 
이 방법은 Reinforce-select보다 낮은 training variance를 갖는다. 
selector와 generator는 같은 objective로 동시에 학습되기 때문에 generation은 Bottom-up 방식보다 selected content에 더 충실하다. 
이 모델은 task에 구애받지 않고, end-to-end 학습이 가능하고, 모든 encoder-decoder architecture에 완벽하게 삽입이 가능하다.  
&nbsp;&nbsp;이 논문에서는 Enc-Dec text generation을 위한 content selection의 문제를 연구하고, 좋은 성능을 보이는 task에 구애받지 않는 training framework를 제안하고, performance와 controllability간의 trade-off를 효과적으로 달성하는 방법을 제시한다.    

## 3. Content Selection
논문의 목표는 추가적인 content selector를 사용하는 방식으로 content selection을 decoder에서 분리하는 것이다. 
보다 더 해석가능하고 제어 가능한 generation을 위해 content selector가 content-level diversity를 완전히 포착하기를 바랬다. 
content selection을 sequence labeling task로 정의했다. 
$$\beta_i=1$$일 때 $$h_i$$가 선택되고 0이면 otherwise이다. 
$$\beta_i=1$$는 각각 독립적이고, bernoulli 분포$$B(\gamma_i)$$를 따른다. 
$$\gamma_i$$는 bernoulli parameter로 source encoder 상단의 2 layer feedforward network를 사용해 추정한다. 
문장은 $$\beta$$로 sampling되어 어떤 content를 선택할 지 결정한 뒤 조건부확률 $$p_\theta(Y|X,\beta)$$로 decode 되어 생성된다. 
문장은 모든 선택된 content를 전달받고, 선택되지 않은 것들은 버려진다. decoder는 임의의 문장을 생성하는 대신 선택된 token을 연결하여 기존의 정보를 잘 포함하도록 했다. 
아래 그림은 생성 과정을 도식화 한다. 

![그림2](/assets/images/VRS_figure2.png "그림2"){: .align-center} 

각각의 source target 쌍에서 ground-truth selection mask를 알 수 없기 때문에 훈련시키기 어렵다. 
다음 장에서는 여러 훈련 방법에 대해 이야기하고, 세부적인 모델을 제안한다.    

### 3.1 Bottom-up
