# 홈 CCTV 특화 객체 인식 모델

YOLOv5 기반의 홈 CCTV 환경에 최적화된 객체 인식 모델 개발 프로젝트입니다.

## 📋 프로젝트 개요

이 프로젝트는 일반적인 객체 인식 모델과 달리 홈 CCTV 환경에 특화된 객체들만을 인식하도록 설계되었습니다. COCO 데이터셋에서 필요한 클래스만 선별하여 불필요한 객체(자동차, 자전거 등)를 제외하고 가정 내에서 중요한 객체들만 감지합니다.

## 🎯 인식 대상 객체

- **사람** (person) - 가족/침입자 구분 가능
- **반려동물** (cat, dog) - 고양이, 개
- **위험물체** (knife, scissors) - 칼, 가위
- **택배상자** (backpack, handbag, suitcase) - 배낭, 핸드백, 여행가방
- **귀중품** (laptop, mouse, remote, keyboard, cell phone) - 전자기기
- **가구** (chair, couch, bed, dining table, toilet, tv) - 주요 가구

## 🚀 시작하기

### 필요 조건

- Python 3.8 또는 3.9
- CUDA 지원 GPU (권장)
- 최소 16GB RAM

### 설치

1. 저장소 클론
```bash
git clone https://github.com/your-username/home-cctv-object-detection.git
cd home-cctv-object-detection
```

2. 가상환경 생성 및 활성화
```bash
python -m venv makeModel
# Windows
makeModel\Scripts\activate
# Linux/Mac
source makeModel/bin/activate
```

3. 의존성 설치
```bash
pip install -r requirements.txt
```

### 데이터셋 준비

1. COCO 데이터셋 다운로드
```bash
mkdir coco
cd coco
# Annotations 다운로드
curl -o annotations_trainval2017.zip http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip

# 이미지 다운로드 (선택사항 - 약 19GB)
curl -o train2017.zip http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip
```

2. 홈 CCTV 특화 데이터셋 생성
```bash
# COCO에서 필요한 클래스만 필터링
python make_model.py

# YOLOv5 형식으로 변환
python coco_to_yolo.py
```

### 모델 학습

1. YOLOv5 설치
```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
```

2. 학습 시작
```bash
python train.py --img 640 --batch 16 --epochs 100 --data ../HomeCCTV_dataset/data.yaml --weights yolov5s.pt --cache
```

3. 학습 모니터링
```bash
tensorboard --logdir runs/train
```

## 📁 프로젝트 구조

```
home-cctv-object-detection/
├── make_model.py              # COCO 데이터셋 필터링 스크립트
├── coco_to_yolo.py           # YOLO 형식 변환 스크립트
├── check_coco_classes.py     # COCO 클래스 확인 스크립트
├── requirements.txt          # Python 의존성
├── 객체인식모델_개발계획.txt    # 프로젝트 개발 계획서
├── README.md                 # 프로젝트 설명서
├── .gitignore               # Git 무시 파일 목록
├── coco/                    # COCO 데이터셋 (다운로드 후)
├── HomeCCTV_dataset/        # 변환된 YOLOv5 데이터셋
└── yolov5/                  # YOLOv5 저장소 (클론 후)
```

## 🔧 사용법

### 모델 평가
```bash
cd yolov5
python val.py --weights runs/train/exp/weights/best.pt --data ../HomeCCTV_dataset/data.yaml --img 640
```

### 실시간 감지 (웹캠)
```bash
python detect.py --weights runs/train/exp/weights/best.pt --source 0
```

### 이미지/비디오 감지
```bash
python detect.py --weights runs/train/exp/weights/best.pt --source path/to/image_or_video
```

## 📊 성능 목표

- **mAP (mean Average Precision)**: 0.85 이상
- **실시간 처리**: 30 FPS 이상 (GPU 환경)
- **정확도**: 홈 환경에서 95% 이상

## 🛠️ 커스터마이징

### 새로운 클래스 추가
`make_model.py` 파일에서 `홈_CCTV_클래스` 딕셔너리를 수정하여 원하는 COCO 클래스를 추가/제거할 수 있습니다.

### 하이퍼파라미터 조정
YOLOv5 학습 시 다음 파라미터들을 조정할 수 있습니다:
- `--img`: 입력 이미지 크기
- `--batch`: 배치 크기
- `--epochs`: 학습 에폭 수
- `--lr0`: 초기 학습률

## 📈 개발 계획

- [x] COCO 데이터셋 필터링
- [x] YOLOv5 형식 변환
- [ ] 기본 모델 학습
- [ ] 홈 환경 데이터 추가 수집
- [ ] 화재/연기 감지 기능 추가
- [ ] 모델 최적화 및 경량화
- [ ] 실시간 알림 시스템 구현

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 📞 연락처

프로젝트 관련 문의사항이 있으시면 이슈를 생성해주세요.

## 🙏 감사의 말

- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
- [COCO Dataset](https://cocodataset.org/)
- [PyTorch](https://pytorch.org/) 