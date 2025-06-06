#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔥 홈 CCTV 커스텀 모델 훈련 스크립트 (28개 클래스)

기능:
- 기본 19개 클래스 + 화재/연기/창문문상태/어린이행동 9개 추가
- 긴급 상황 감지 특화 (화재, 연기, 어린이 위험)
- 전이학습 + 커스텀 데이터 fine-tuning

사용법:
    python 커스텀훈련.py

주의사항:
    - 커스텀 데이터 (화재/연기/창문/어린이) 필요
    - 라벨링 도구로 정확한 바운딩박스 생성 필수
"""

import os
import sys
import torch
import subprocess
from pathlib import Path
from datetime import datetime

def check_environment():
    """환경 확인 및 설정"""
    print("🔧 환경 확인 중...")
    
    # Python 버전 확인
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 이상이 필요합니다.")
        return False
    
    # GPU 확인
    if torch.cuda.is_available():
        print(f"✅ GPU 사용 가능: {torch.cuda.get_device_name(0)}")
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"💾 GPU 메모리: {gpu_memory:.1f}GB")
        device = 'cuda'
    else:
        print("⚠️ GPU를 사용할 수 없습니다. CPU로 훈련합니다.")
        device = 'cpu'
    
    return device

def install_dependencies():
    """필요한 패키지 설치"""
    print("📦 필수 패키지 설치 중...")
    
    packages = [
        "ultralytics",
        "torch>=1.7.0",
        "torchvision",
        "matplotlib",
        "opencv-python",
        "pillow",
        "pyyaml",
        "requests",
        "tqdm",
        "seaborn",
        "plotly"
    ]
    
    for package in packages:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                         capture_output=True, check=True)
            print(f"✅ {package} 설치 완료")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ {package} 설치 실패: {e}")

def download_yolov5():
    """YOLOv5 다운로드"""
    if not os.path.exists("yolov5"):
        print("📥 YOLOv5 다운로드 중...")
        subprocess.run(["git", "clone", "https://github.com/ultralytics/yolov5.git"])
        print("✅ YOLOv5 다운로드 완료")
    else:
        print("✅ YOLOv5 이미 존재")

def create_custom_dataset_yaml():
    """커스텀 데이터셋 설정 파일 생성"""
    print("📄 커스텀 데이터셋 설정 파일 생성 중...")
    
    # 전체 28개 클래스 (기존 19 + 커스텀 9)
    all_classes = {
        # 기존 홈 CCTV 클래스 (0-18)
        0: 'person',
        1: 'cat', 2: 'dog',
        3: 'knife', 4: 'scissors',
        5: 'backpack', 6: 'handbag', 7: 'suitcase',
        8: 'laptop', 9: 'mouse', 10: 'remote', 11: 'keyboard', 12: 'cell phone',
        13: 'chair', 14: 'couch', 15: 'bed', 16: 'dining table', 17: 'toilet', 18: 'tv',
        
        # 새로운 커스텀 클래스 (19-27)
        19: 'fire', 20: 'smoke', 21: 'window_open', 22: 'door_open',
        23: 'window_closed', 24: 'door_closed', 25: 'child_standing',
        26: 'child_running', 27: 'child_falling'
    }
    
    # 우선순위 클래스 정의
    priority_classes = {
        'emergency': [19, 20, 27],  # fire, smoke, child_falling
        'security': [21, 22],       # window_open, door_open  
        'monitoring': [25, 26]      # child_standing, child_running
    }
    
    yaml_content = f"""# 홈 CCTV 커스텀 데이터셋 (28개 클래스)
path: custom_dataset
train: images/train
val: images/val

# 클래스 수
nc: {len(all_classes)}

# 클래스 이름
names:
"""
    
    for class_id, class_name in all_classes.items():
        yaml_content += f"  {class_id}: {class_name}\n"
    
    yaml_content += f"""
# 우선순위 클래스 (긴급도 순)
priority_classes:
  emergency: {priority_classes['emergency']}  # 화재, 연기, 어린이 위험
  security: {priority_classes['security']}    # 창문/문 열림
  monitoring: {priority_classes['monitoring']} # 어린이 행동
"""
    
    with open("custom_dataset.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content)
    
    print("✅ 커스텀 데이터셋 설정 파일 생성 완료")
    return "custom_dataset.yaml", all_classes, priority_classes

def check_custom_dataset():
    """커스텀 데이터셋 확인"""
    print("🔍 커스텀 데이터셋 확인 중...")
    
    dataset_path = Path("custom_dataset")
    
    if not dataset_path.exists():
        print("❌ 커스텀 데이터셋이 없습니다.")
        print("💡 커스텀 데이터셋 생성 방법:")
        print("   1. custom_dataset/images/train/ 폴더에 이미지 추가")
        print("   2. custom_dataset/labels/train/ 폴더에 YOLO 형식 라벨 추가")
        print("   3. 각 커스텀 클래스당 최소 100장 권장:")
        print("      - fire: 화재 이미지")
        print("      - smoke: 연기 이미지")
        print("      - window_open/closed: 창문 상태")
        print("      - door_open/closed: 문 상태") 
        print("      - child_standing/running/falling: 어린이 행동")
        
        # 샘플 데이터셋 디렉토리 생성
        create_sample_dataset()
        return False
    
    # 데이터 통계
    train_images = list((dataset_path / "images" / "train").glob("*.jpg"))
    train_labels = list((dataset_path / "labels" / "train").glob("*.txt"))
    val_images = list((dataset_path / "images" / "val").glob("*.jpg"))
    val_labels = list((dataset_path / "labels" / "val").glob("*.txt"))
    
    print(f"📊 훈련 이미지: {len(train_images)}개")
    print(f"📊 훈련 라벨: {len(train_labels)}개")
    print(f"📊 검증 이미지: {len(val_images)}개")
    print(f"📊 검증 라벨: {len(val_labels)}개")
    
    if len(train_images) < 100:
        print("⚠️ 훈련 데이터가 부족합니다. 커스텀 클래스 훈련을 위해 최소 500장 권장")
        print("💡 데이터 수집 방법:")
        print("   - Roboflow, LabelImg 등으로 라벨링")
        print("   - 공개 데이터셋 활용")
        print("   - 직접 촬영 (화재 시뮬레이션, 창문/문 상태 등)")
    
    return len(train_images) >= 50  # 최소 기준

def create_sample_dataset():
    """샘플 데이터셋 구조 생성"""
    print("📁 샘플 데이터셋 구조 생성 중...")
    
    directories = [
        "custom_dataset/images/train",
        "custom_dataset/images/val", 
        "custom_dataset/labels/train",
        "custom_dataset/labels/val"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # 샘플 README 생성
    readme_content = """# 커스텀 데이터셋 가이드

## 📁 폴더 구조
```
custom_dataset/
├── images/
│   ├── train/          # 훈련 이미지 (.jpg)
│   └── val/            # 검증 이미지 (.jpg)
└── labels/
    ├── train/          # 훈련 라벨 (.txt)
    └── val/            # 검증 라벨 (.txt)
```

## 🏷️ 라벨 형식 (YOLO)
각 .txt 파일은 다음 형식:
```
class_id x_center y_center width height
```
- 모든 값은 0-1 사이로 정규화
- 예시: `19 0.5 0.4 0.3 0.2` (fire 클래스)

## 🎯 커스텀 클래스 ID
- 19: fire (화재)
- 20: smoke (연기)  
- 21: window_open (창문 열림)
- 22: door_open (문 열림)
- 23: window_closed (창문 닫힘)
- 24: door_closed (문 닫힘)
- 25: child_standing (어린이 서있음)
- 26: child_running (어린이 뛰어다님)
- 27: child_falling (어린이 넘어짐)

## 📊 권장 데이터 수량
각 클래스당:
- 훈련: 100-500장
- 검증: 20-100장

## 🔧 라벨링 도구
- Roboflow (온라인, 추천)
- LabelImg (로컬)
- CVAT (온라인)
"""
    
    with open("custom_dataset/README.md", "w", encoding="utf-8") as f:
        f.write(readme_content)
    
    print("✅ 샘플 데이터셋 구조 생성 완료")
    print("📖 custom_dataset/README.md 참고하여 데이터 준비")

def train_custom_model(device="auto", classes_dict=None, priority_classes=None):
    """커스텀 모델 훈련"""
    print("🔥 커스텀 홈 CCTV 모델 훈련 시작!")
    print("=" * 50)
    print("🎯 목표: 화재/연기, 창문/문 상태, 어린이 행동 인식")
    
    # 배치 사이즈 설정 (커스텀 클래스로 메모리 사용량 증가)
    if device == "cuda":
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory > 20:
            batch_size = 12
        elif gpu_memory > 15:
            batch_size = 8
        elif gpu_memory > 10:
            batch_size = 6
        else:
            batch_size = 4
    else:
        batch_size = 2
    
    print(f"⚙️ 배치 사이즈: {batch_size} (커스텀 클래스용 조정)")
    print(f"🔧 디바이스: {device}")
    
    # 커스텀 훈련 설정
    train_config = {
        'img': 640,
        'batch': batch_size,
        'epochs': 120,              # 커스텀 클래스용 더 많은 epoch
        'data': '../custom_dataset.yaml',
        'weights': 'yolov5s.pt',
        'device': device,
        'project': '../runs',
        'name': 'home_cctv_custom',
        'exist_ok': True,
        
        # 전이학습 최적화 (커스텀 특화)
        'freeze': 15,               # 더 많은 레이어 고정
        'patience': 25,             # 긴 patience
        'save_period': 10,
        
        # 학습률 (커스텀 클래스용 낮은 학습률)
        'lr0': 0.0008,             # 낮은 초기 학습률
        'lrf': 0.001,              # 최종 학습률
        'momentum': 0.937,
        'weight_decay': 0.001,     # 정규화 강화
        'warmup_epochs': 5,
        
        # 데이터 증강 (커스텀 클래스 특화)
        'hsv_h': 0.03,             # 색상 변화 (화재/연기)
        'hsv_s': 0.7,              # 채도 변화
        'hsv_v': 0.4,              # 명도 변화 (낮/밤)
        'degrees': 10.0,           # 회전
        'translate': 0.1,          # 이동
        'scale': 0.5,              # 크기 변화
        'shear': 2.0,              # 전단 변형
        'flipud': 0.05,            # 상하 반전 (어린이 행동)
        'fliplr': 0.5,             # 좌우 반전
        'mosaic': 1.0,             # 모자이크 증강
        'mixup': 0.1,              # Mixup 증강
        'copy_paste': 0.1,         # Copy-paste 증강
        
        'cache': True,
        'workers': 2               # 커스텀 데이터로 안정성 우선
    }
    
    # 훈련 명령 생성
    cmd = ["python", "train.py"]
    for key, value in train_config.items():
        cmd.extend([f"--{key}", str(value)])
    
    print("🎯 커스텀 훈련 설정:")
    print(f"  - 총 클래스: {len(classes_dict)}개")
    print(f"  - 기존 홈 CCTV: 19개")
    print(f"  - 새로운 커스텀: 9개")
    print(f"  - 에폭 수: {train_config['epochs']}")
    print(f"  - 배치 크기: {train_config['batch']}")
    print(f"  - 학습률: {train_config['lr0']}")
    
    # 우선순위 클래스 표시
    print(f"\n🚨 우선순위 클래스:")
    for priority, class_ids in priority_classes.items():
        class_names = [classes_dict[cid] for cid in class_ids]
        print(f"  - {priority.upper()}: {class_names}")
    
    # 훈련 실행
    os.chdir("yolov5")
    try:
        print(f"\n🚀 커스텀 훈련 시작! (약 3-5시간 소요)")
        print("💡 긴급 클래스 성능을 우선적으로 모니터링하세요!")
        
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\n🎉 커스텀 훈련 완료!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 훈련 실패: {e}")
        return False
    finally:
        os.chdir("..")

def analyze_custom_results(classes_dict, priority_classes):
    """커스텀 훈련 결과 분석"""
    print("\n📊 커스텀 훈련 결과 분석:")
    print("=" * 50)
    
    model_path = Path("runs/home_cctv_custom/weights/best.pt")
    if model_path.exists():
        print(f"✅ 최종 모델: {model_path}")
        print(f"📊 모델 크기: {model_path.stat().st_size / 1024 / 1024:.1f}MB")
        
        # 3.완성모델로 복사
        import shutil
        shutil.copy(model_path, "3.완성모델/best_custom.pt")
        print("✅ 모델을 3.완성모델/에 복사했습니다")
    
    # 성능 리포트 생성
    create_performance_report(classes_dict, priority_classes)
    
    print("\n🎯 다음 단계:")
    print("1. python 3.완성모델/test_model.py (커스텀 모델 테스트)")
    print("2. 긴급 상황 탐지 시스템 구축")
    print("3. 실시간 알림 시스템 연동")

def create_performance_report(classes_dict, priority_classes):
    """성능 리포트 생성"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    report = f"""# 🔥 홈 CCTV 커스텀 모델 훈련 리포트

생성 시간: {timestamp}

## 📊 모델 정보
- 총 클래스 수: {len(classes_dict)}개
- 기존 홈 CCTV 클래스: 19개
- 새로운 커스텀 클래스: 9개

## 🏠 기존 홈 CCTV 클래스 (19개)
- 사람: person
- 반려동물: cat, dog
- 위험물체: knife, scissors
- 택배상자: backpack, handbag, suitcase
- 귀중품: laptop, mouse, remote, keyboard, cell phone
- 가구: chair, couch, bed, dining table, toilet, tv

## 🔥 새로운 커스텀 클래스 (9개)

### 🚨 긴급 클래스 (Emergency)
- fire (화재): 불꽃, 화염 감지
- smoke (연기): 화재 연기, 이상 연기 감지
- child_falling (어린이 위험): 넘어지는 위험 상황

### 🔒 보안 클래스 (Security)
- window_open (창문 열림): 보안 위험 상태
- door_open (문 열림): 보안 위험 상태

### 👀 모니터링 클래스 (Monitoring)
- window_closed (창문 닫힘): 정상 상태
- door_closed (문 닫힘): 정상 상태
- child_standing (어린이 서있음): 일반 상태
- child_running (어린이 뛰어다님): 활동 상태

## 📈 예상 성능 목표
- 전체 mAP@0.5: 0.70-0.85
- 긴급 클래스: 0.65-0.80 (높은 정확도 우선)
- 보안 클래스: 0.75-0.85
- 모니터링 클래스: 0.70-0.80

## 🚨 긴급 상황 감지 시스템
우선순위: {priority_classes['emergency']} (fire, smoke, child_falling)
- 신뢰도 임계값: 0.7 이상
- 실시간 알림 발송
- 이메일/SMS/푸시 알림 연동

## 🔧 성능 개선 권장사항
1. 긴급 클래스 데이터 보강 (각 클래스당 200+ 이미지)
2. 다양한 조명 조건에서 창문/문 데이터 수집
3. 어린이 행동 연속 프레임 데이터 추가
4. 하드 네거티브 마이닝으로 오탐지 감소

## 🚀 배포 준비사항
- [ ] 모델 경량화 (ONNX/TensorRT 변환)
- [ ] 실시간 스트리밍 연동
- [ ] 알림 시스템 구축
- [ ] 웹/모바일 인터페이스 개발
- [ ] 성능 모니터링 대시보드

훈련 완료! 커스텀 홈 CCTV 모델이 준비되었습니다. 🎉
"""
    
    with open("3.완성모델/performance_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    
    print("📋 성능 리포트 생성: 3.완성모델/performance_report.txt")

def main():
    """메인 실행 함수"""
    print("🔥 홈 CCTV 커스텀 모델 훈련")
    print("=" * 50)
    print("📋 대상: 28개 클래스 (기존 19 + 커스텀 9)")
    print("🎯 목표: 화재/연기, 창문/문 상태, 어린이 행동 인식")
    print("⏱️ 예상 시간: 3-5시간")
    print()
    
    try:
        # 1. 환경 확인
        device = check_environment()
        if not device:
            return
        
        # 2. 패키지 설치
        install_dependencies()
        
        # 3. YOLOv5 다운로드
        download_yolov5()
        
        # 4. 커스텀 데이터셋 설정
        dataset_yaml, classes_dict, priority_classes = create_custom_dataset_yaml()
        
        # 5. 커스텀 데이터셋 확인
        if not check_custom_dataset():
            print("\n⏸️ 커스텀 데이터셋을 준비한 후 다시 실행하세요.")
            print("📖 custom_dataset/README.md 참고")
            return
        
        # 6. 커스텀 모델 훈련
        success = train_custom_model(device, classes_dict, priority_classes)
        
        # 7. 결과 분석
        if success:
            analyze_custom_results(classes_dict, priority_classes)
        
        print("\n🎉 커스텀 홈 CCTV 모델 훈련 완료!")
        print("🔥 화재/연기, 창문/문 상태, 어린이 행동 인식 모델 준비!")
        
    except KeyboardInterrupt:
        print("\n⏹️ 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        print("💡 문제 해결:")
        print("  1. 커스텀 데이터셋이 올바르게 준비되었는지 확인")
        print("  2. 라벨링이 YOLO 형식에 맞는지 확인") 
        print("  3. GPU 메모리가 충분한지 확인 (배치 사이즈 조정)")

if __name__ == "__main__":
    main() 