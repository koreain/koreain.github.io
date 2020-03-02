---
title: "LaserTagger"
excerpt: "인코딩, 태깅, 인지 : 제어가 가능하고 효율적인 텍스트 생성을 위한 접근"
toc: true
toc_stick: true
use_math: truef
categories:
  - ML
tags:
  - 자연어처리
  - GooglAI
  - LaserTagger
last_modified_at: 2020-03-02T18:21:00-00:00
---

# Encode, Tag and Realize: A Controllable and Efficient Approach for Text Generation
---
posted date: 2020-1-31 (금)
posted by Eric Malmi and Sebastian Krause, Software Engineers, Google Research

[seq2seq](https://en.wikipedia.org/wiki/Seq2seq) 모델들은 기계번역 분야에서 혁신을 주도하고 있고, 다양한 텍스트 생성 문제들([요약](https://www.microsoft.com/en-us/research/publication/dataset-evaluation-metrics-abstractive-sentence-paragraph-compression/), [문장융합](https://ai.google/tools/datasets/discofuse/), 문법오류 수정 등)을 해결하는 도구로서 선택되어 왔다. 모델 아키텍처(즉, [Transformer](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)) 개선과 비지도-사전 학습을 통해 얻어진 주석이 없는 대규모 말뭉치를 활용할 수 있는 능력이 최근 몇 년간 신경망 접근 방식에서 성능 향상을 가능하게 했다.  

하지만 텍스트 생성 문제에 seq2seq 모델을 활용하는 것은 사용 사례에 따라 여러 가지 결함들이 발생할 수 있다.
1. 입력 텍스트에 의해 지원되지 않는 출력을 생산해내는 문제(hallucination이라고 불린다; 환각; 존재하지 않는 무언가에 대한 감각적 경험)가 있다.
2. 좋은 성능에 도달하기 위해 많은 양의 학습 데이터를 요구하는 문제가 있다.
3. 더 나아가, seq2seq 모델은 일반적으로 단어 별로 텍스트 생성 결과를 생성해내기 때문에 추론 시간이 매우 느리다.

["Encode, Tag, Realize: High-Precision Text Editing"](https://ai.google/research/pubs/pub48542/) 포스트에서 우리는 특별히 위에서 언급한 세가지 문제점을 해결하기 위해 설계된 [오픈소스](http://lasertagger.page.link/code)의 새로운 텍스트 생성 방법론을 제안했다. 이 방법론은 속도와 정확도 때문에 LaserTagger 라고 부른다. 처음부터 결과 텍스트를 생성하는 대신, LaserTagger는 예측된 편집 연산을 활용하여 단어들 태깅을 통해 결과 텍스트를 생성하고, 이 결과물들은 별도로 분리된 인지(realization) 단계에서 입력 단어들에 적용된다. 오류가 발생할 가능성이 낮은 텍스트 생성 방법이며, 이는 학습 작업에서 쉽게 처리될 수 있으며 모델 아키텍처들을 더 빠르게 실행할 수 있다.  

## 1. LaserTagger 모델의 구조와 기능  
---
많은 텍스트 생성 작업의 뚜렷한 특징은 입력과 출력 텍스트가 자주 겹치게 된다는 것이다. (overlap) 예를 들면, 문법적 실수를 발견하고 고치거나 문장 융합 작업을 할 경우 대부분의 입력 텍스트는 변화시킬 필요가 없고, 단지 몇몇 단어의 수정이 필요한 경우가 많다. 이러한 이유로, LaserTagger는 실제 단어들 대신 편집 연산들의 시퀀스를 생성한다.  
우리는 4가지 유형의 편집 연산을 정의했다.  

1. Keep (입력 텍스트의 단어를 그대로 복사)  
2. Delete (입력 텍스트의 단어를 삭제)  
3. Keep-AddX (태그된 단어 이전에 구$^{phrase}$ X를 추가)  
4. Delete-AddX (태그된 단어를 지우고 그 자리에 구$^{phrase}$ X를 추가)  

이러한 편집 과정은 아래 그림에서 확인할 수 있다.  
(문장 융합 작업에서의 LaserTagger 적용 사례)  


![LaserTagger_to_sentence_fusion](https://github.com/koreain/koreain.github.io/blob/master/assets/images/Laser_Tagger_Sentence_Fusion.png?raw=true "LaserTagger to sentence fusion")  


위 사레에서 보면, 두번째 문장의 Turing 단어를 삭제하고 "and he"라는 구문을 그 이전에 추가했다.  
(Delte-AddX 연산이 실행된 것을 확인할 수 있다.)  

모든 추가된 구들은 한정된 사전으로 부터 도출된다. 이 사전은 사전 크기를 최소화하고, 대상 텍스트에 추가하는 데 필요한 단어만 어휘에서 나오는 학습 예제 수를 최대화하는 최적화 프로세스의 결과이다. 한정된 어휘 사전을 활용하는 것은 출력 결정 범위를 축소시키고, 모델이 임의의 단어를 추가하는 것을 방지하여 환각(hallucination) 문제를 완화할 수 있다. 입력 및 출력 텍스트의 높은 오버랩 속성에 대한 결과는 필요한 텍스트 수정이 지역(국소)적이고 독립적인 경향이 있다는 것이다.  

이는 이전 예측 결과를 조건부로 순차적으로 예측을 수행하는 자기회귀방식의 seq2seq 모델과 비교했을때, 입력 텍스트에 필요한 편집 연산이 높은 정확도를 가지면서 병렬적으로 예측될 수 있고, end-to-end 과정에 걸쳐 상당한 속도 향상을 가져올 수 있음을 의미한다.  

## 2. 결론
---
우리는 LaserTagger 모델을 4개의 작업에 대해서 평가했다.  
1. sentence fusion (문장융합)
2. split and rephrase (문장 분해 및 재구성?)
3. astractive summarization (요약)
4. grammer correction (문법보정)

모든 작업에 대해서,  LaserTagger는 대규모 학습 데이터를 사용한 strong BERT-based seq2seq baseline 모델과 유사한 성능을 보였다. 나아가 학습 데이터에 제한을 두었을 경우에는 명확하게 더 뛰어난 성능을 보였다. (아래 그림 참고)

![LaserTagger Performance](https://github.com/koreain/koreain.github.io/blob/master/assets/images/Laser_Tagger_Performance.png?raw=true "LaserTagger Performance")

*100만개의 전체 데이터 셋에 대해 학습할 경우, LaserTagger와 BERT 기반의 seq2seq baseline 모델과 유사한 성능을 보이나,*  
*1만개 이하의 데이터 셋에 대해 학습할 경우, LaserTagger가 명확하게 더 뛰어난 성능을 보였다.*


## 3. LaserTagger의 주요 장점
---

전통적인 seq2seq 방법론과 비교했을 때, LaserTagger는 다음과 같은 강점들을 가진다.  

1. 제어가능 : 우리가 직접 편집하고 관리할수 있는 **출력 구문 어휘** 의 조절 및 제어를 통해, 환각$^{hallucination}$이슈에 덜 취약하도록 한다.  
2. 추론속도 : seq2seq baseline 모델보다 최대 100배 더 빠른 예측 연산을 수행하고, 이는 모델을 실시간 서비스에 적용 가능하도록 해준다.  
3. 데이터효율성 : 수 백개 또는 수 천개의 데이터만으로 학습한 경우에도 합리적인 수준의 결과를 도출한다. 기존 seq2seq base라인 모델이 유사한 수준의 결과를 도출하기 위해서는 수 백, 수 천이 아닌 수 만개의 데이터를 학습해야 했다.  

## 4. 왜 LaserTagger와 같은 접근방식이 중요한가?
---

응답 길이를 줄이고 반복 횟수를 줄임으로써 일부 서비스에서 음성 응답의 공식화를 향상시키는 등 LaserTagger의 장점은 대규모로 적용 할 때 더욱 두드러진다. 매우 빠른 추론 속도는 사용자 측면에서 어떠한 눈에 띄는 서비스 지연 없이, 기존에 존재하는 기술 스택에 모델을 활용될 수 있도록 한다. 또한 개선된 데이터 효율성은 다양한 언어들의 학습 데이터 수집을 가능하게 하고, 이는 서로 다른 언어적 배경의 사용자들에게 헤택으로 돌아갈 수 있다.  


해당 소스는 오픈소스로서 [여기](https://github.com/google-research/lasertagger)에서 확인할 수 있다.  


## 5. 개인적으로.. 
---
현재 회사에서 진행중인 프로젝트에서 만약 문장을 생성해야 한다면, 생성된 문장의 문법적 오류를 바로바로 잡아서 서비스에 노출시켜 줄 수 있는 기능 구현에 LaserTagger를 활용해 보면 좋을 것 같다.  

블로그 원문 바로가기 [Original Blog link](https://ai.googleblog.com/2020/01/encode-tag-and-realize-controllable-and.html)