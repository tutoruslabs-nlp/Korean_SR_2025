# 한국어 문장 간 논리 판별 과제 Baseline
본 리포지토리는 '2025년 국립국어원 인공지능의 한국어 능력 평가' 상시 과제 중 '한국어 문장 간 논리 판별 과제'에 대한 베이스라인 모델의 추론을 재현하기 위한 코드를 포함하고 있습니다.  

추론의 실행 방법(How to Run)은 아래에서 확인하실 수 있습니다.

## 리포지토리 구조 (Repository Structure)
```
# 추론에 필요한 리소스들을 보관하는 디렉토리
resource
└── data

# 실행 가능한 python 스크립트를 보관하는 디렉토리
run
└── test.py

# 추론에 사용될 함수들을 보관하는 디렉토리
src
└── data.py
```

## 데이터 형태 (Data Format)
```
[
   {
      "id": "nikluge-2025-문장_간_논리_관계_판별-dev-1",
      "input": {
         "front": "공매도는 주식을 빌려 파는 만큼 나중에 이를 갚기 위해 다시 주식을 사들여야 한다.",
         "back": "파는 가격에 비해 사는 가격이 높으면 공매도한 투자자는 손실을 보게 된다."
      },
      "output": "순접"
   }
]
```

## 실행 방법 (How to Run)
### 추론 (Inference)
```
(실제 코드는 25년 7월 중순에 업데이트 예정)
```


## Reference
국립국어원 인공지능 (AI)말평 (https://kli.korean.go.kr/benchmark)  
transformers (https://github.com/huggingface/transformers)  
Qwen3-8B (https://huggingface.co/Qwen/Qwen3-8B)  
HyperCLOVAX-SEED-Text-Instruct-1.5B (https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B)
