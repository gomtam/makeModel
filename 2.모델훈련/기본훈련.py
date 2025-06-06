#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏠 홈 CCTV 기본 모델 훈련 스크립트 (19개 클래스)

기능:
- 홈 CCTV에 필요한 19개 클래스만 인식
- 전이학습으로 빠르고 안정적인 훈련
- 자동 환경 설정 및 오류 처리

사용법:
    python 기본훈련.py

결과:
    - 훈련된 모델: runs/train/exp/weights/best.pt
    - 성능 그래프: runs/train/exp/results.png
"""

import os
import sys
import torch
import subprocess
from pathlib import Path

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
        "tqdm"
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

def create_dataset_yaml():
    """데이터셋 설정 파일 생성"""
    print("📄 데이터셋 설정 파일 생성 중...")
    
    # 정확한 홈 CCTV 클래스 (19개)
    classes = {
        0: 'person',
        1: 'cat', 2: 'dog',
        3: 'knife', 4: 'scissors',
        5: 'backpack', 6: 'handbag', 7: 'suitcase',
        8: 'laptop', 9: 'mouse', 10: 'remote', 11: 'keyboard', 12: 'cell phone',
        13: 'chair', 14: 'couch', 15: 'bed', 16: 'dining table', 17: 'toilet', 18: 'tv'
    }
    
    yaml_content = f"""# 홈 CCTV 기본 데이터셋 (19개 클래스)
path: ../임시파일/HomeCCTV_dataset
train: images/train
val: images/val

# 클래스 수
nc: {len(classes)}

# 클래스 이름
names:
"""
    
    for class_id, class_name in classes.items():
        yaml_content += f"  {class_id}: {class_name}\n"
    
    with open("home_cctv_dataset.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content)
    
    print("✅ 데이터셋 설정 파일 생성 완료")
    return "home_cctv_dataset.yaml"

def check_dataset():
    """데이터셋 존재 확인"""
    dataset_path = Path("../임시파일/HomeCCTV_dataset")
    
    if not dataset_path.exists():
        print("❌ 데이터셋이 없습니다.")
        print("💡 먼저 1.데이터생성/ 폴더의 스크립트들을 실행하세요:")
        print("   1. python 1.데이터생성/make_model.py")
        print("   2. python 1.데이터생성/coco_to_yolo.py")
        return False
    
    # 이미지와 라벨 체크
    train_images = list((dataset_path / "images" / "train").glob("*.jpg"))
    train_labels = list((dataset_path / "labels" / "train").glob("*.txt"))
    
    print(f"📊 훈련 이미지: {len(train_images)}개")
    print(f"📊 훈련 라벨: {len(train_labels)}개")
    
    if len(train_images) < 100:
        print("⚠️ 훈련 이미지가 부족합니다. 최소 100개 권장")
        return False
    
    return True

def train_model(device="auto"):
    """모델 훈련 실행"""
    print("🚀 기본 홈 CCTV 모델 훈련 시작!")
    print("=" * 50)
    
    # 배치 사이즈 설정
    if device == "cuda":
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory > 15:
            batch_size = 16
        elif gpu_memory > 10:
            batch_size = 8
        else:
            batch_size = 4
    else:
        batch_size = 2
    
    print(f"⚙️ 배치 사이즈: {batch_size}")
    print(f"🔧 디바이스: {device}")
    
    # 훈련 설정
    train_config = {
        'img': 640,
        'batch': batch_size,
        'epochs': 80,
        'data': '../home_cctv_dataset.yaml',
        'weights': 'yolov5s.pt',
        'device': device,
        'project': '../runs',
        'name': 'home_cctv_basic',
        'exist_ok': True,
        
        # 전이학습 최적화
        'freeze': 10,         # backbone 고정
        'patience': 15,       # 조기 종료
        'save_period': 10,    # 주기적 저장
        
        # 데이터 증강 (홈 환경 특화)
        'hsv_h': 0.015,       # 색상 변화
        'hsv_v': 0.4,         # 명도 변화 (낮/밤)
        'flipud': 0.0,        # 상하반전 비활성화 (CCTV)
        'mosaic': 1.0,        # 모자이크 증강
        'cache': True,        # 캐시 사용
        'workers': 4
    }
    
    # 훈련 명령 생성
    cmd = ["python", "train.py"]
    for key, value in train_config.items():
        cmd.extend([f"--{key}", str(value)])
    
    print("🎯 훈련 설정:")
    print(f"  - 이미지 크기: {train_config['img']}")
    print(f"  - 배치 크기: {train_config['batch']}")
    print(f"  - 에폭 수: {train_config['epochs']}")
    print(f"  - 가중치: {train_config['weights']}")
    
    # 훈련 실행
    os.chdir("yolov5")
    try:
        print("\n🚀 훈련 시작! (약 1-2시간 소요)")
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\n🎉 훈련 완료!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 훈련 실패: {e}")
        return False
    finally:
        os.chdir("..")

def show_results():
    """훈련 결과 표시"""
    print("\n📊 훈련 결과:")
    print("=" * 50)
    
    model_path = Path("runs/home_cctv_basic/weights/best.pt")
    if model_path.exists():
        print(f"✅ 최종 모델: {model_path}")
        print(f"📊 모델 크기: {model_path.stat().st_size / 1024 / 1024:.1f}MB")
        
        # 3.완성모델로 복사
        import shutil
        shutil.copy(model_path, "3.완성모델/best_basic.pt")
        print("✅ 모델을 3.완성모델/에 복사했습니다")
    
    results_path = Path("runs/home_cctv_basic/results.png")
    if results_path.exists():
        print(f"📈 성능 그래프: {results_path}")
    
    print("\n🎯 다음 단계:")
    print("1. python 3.완성모델/test_model.py (모델 테스트)")
    print("2. python 2.모델훈련/커스텀훈련.py (화재/연기 감지 추가)")

def main():
    """메인 실행 함수"""
    print("🏠 홈 CCTV 기본 모델 훈련")
    print("=" * 50)
    print("📋 대상: 19개 홈 CCTV 클래스")
    print("🎯 목표: mAP@0.5 > 0.8")
    print("⏱️ 예상 시간: 1-2시간")
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
        
        # 4. 데이터셋 설정
        dataset_yaml = create_dataset_yaml()
        
        # 5. 데이터셋 확인
        if not check_dataset():
            return
        
        # 6. 모델 훈련
        success = train_model(device)
        
        # 7. 결과 표시
        if success:
            show_results()
        
        print("\n🎉 기본 홈 CCTV 모델 훈련 완료!")
        
    except KeyboardInterrupt:
        print("\n⏹️ 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        print("💡 문제 해결:")
        print("  1. 가상환경이 활성화되어 있는지 확인")
        print("  2. GPU 드라이버가 설치되어 있는지 확인")
        print("  3. 데이터셋이 올바르게 생성되었는지 확인")

if __name__ == "__main__":
    main() 