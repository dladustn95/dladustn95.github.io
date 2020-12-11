---
title:  "Generating Informative Responses with Controlled Sentence Function 논문 리뷰"
excerpt: "Generating Informative Responses with Controlled Sentence Function 논문 리뷰"

mathjax: true
tags:
  - [DeepLearning, DialogueGeneration NLP]
date: 2020-02-10T18:00:00+09:00
last_modified_at: 2020-02-10T18:00:00+09:00
---

# Generating Informative Responses with Controlled Sentence Function

## 1. Introduction
&nbsp;&nbsp;*sentence function*은 화자의 목적을 크게 질문, 선언, 명령, 감탄 4가지로 분류한다. 
따라서 sentence function은 대화 중에 대화의 목적을 나타내는 중요한 요소이다. 
질문은 user에게서 추가적인 정보를 얻기 위해 사용된다. 
명령은 추가 상호작용을 유도하기 위한 지시, 명령, 초대를 하는데 사용된다. 
선언은 무언가를 진술하거나 설명하기 위해 사용된다. 
질문과 명령은 대화에서 교착 상태를 방지하기 위해 사용된다. 
따라서, sentence function을 제어 할 수 있는 conversational system은 여러 상황에서 다양한 목적으로 전략을 조정할 수 있고,보다 적극적으로 행동하며, 대화가 더 진행될 수 있게 한다.  
&nbsp;&nbsp;sentence function을 사용한 text generation은 다른 controllable text generation과 큰 차이가 있다. 
sentiment polarity, emotion, or tense를 통제하는 연구는 local control에 해당하는데 controllable variable가 local variable-related word를 decoding 하여 locally reflected 되기 때문이다. 
반대로 sentence function은 문장의 global attribute이다.
sentence function을 제어하는 것은 단어의 순서와 패턴을 바꾸는 등 전체 문장의 global structure를 조정해야 하기 때문에 어렵다.  
&nbsp;&nbsp;의미있는 대답을 만들기 위해서 sentence function과 content 사이의 적합성을 조정해야 한다. 
