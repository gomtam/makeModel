#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚙️ 홈 CCTV 모델 원클릭 환경 설정 스크립트

기능:
- 필수 패키지 자동 설치
- 가상환경 확인 및 설정
- GPU/CPU 환경 자동 감지
- 모델 파일 다운로드
- 초기 설정 완료

사용법:
    python setup.py
"""

import os
import sys
import subprocess
import importlib
from pathlib import Path

def print_header():
    """헤더 출력"""
    print("🏠 홈 CCTV 모델 환경 설정")
    print("=" * 50)
    print("🎯 자동 설정: 패키지 설치, 환경 확인, 모델 준비")
    print()

def check_python_version():
    """Python 버전 확인"""
    print("🐍 Python 버전 확인 중...")
    
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 이상이 필요합니다.")
        print(f"현재 버전: {sys.version}")
        print("💡 Python 업그레이드 후 다시 시도하세요.")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def check_virtual_env():
    """가상환경 확인"""
    print("🔧 가상환경 확인 중...")
    
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ 가상환경이 활성화되어 있습니다.")
        return True
    else:
        print("⚠️ 가상환경이 활성화되지 않았습니다.")
        print("💡 권장사항:")
        print("   python -m venv venv")
        print("   venv\\Scripts\\activate  # Windows")
        print("   source venv/bin/activate  # Linux/Mac")
        return False

def install_packages():
    """필수 패키지 설치"""
    print("📦 필수 패키지 설치 중...")
    
    # 기본 패키지 목록
    base_packages = [
        "torch>=1.7.0",
        "torchvision>=0.8.0", 
        "ultralytics",
        "opencv-python",
        "matplotlib",
        "pillow",
        "pyyaml",
        "requests",
        "tqdm",
        "seaborn",
        "pandas"
    ]
    
    # 설치 카운터
    success_count = 0
    total_count = len(base_packages)
    
    for package in base_packages:
        try:
            print(f"  📥 {package} 설치 중...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package, "--quiet"],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"  ✅ {package} 설치 완료")
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"  ⚠️ {package} 설치 실패: {e}")
            if "torch" in package:
                print("  💡 PyTorch 수동 설치가 필요할 수 있습니다.")
                print("     방문: https://pytorch.org/get-started/locally/")
    
    print(f"📊 패키지 설치 완료: {success_count}/{total_count}")
    return success_count >= total_count * 0.8  # 80% 이상 성공

def check_gpu():
    """GPU 환경 확인"""
    print("🔧 GPU 환경 확인 중...")
    
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            
            print(f"✅ GPU 사용 가능!")
            print(f"  - GPU 수: {device_count}")
            print(f"  - GPU 이름: {device_name}")
            print(f"  - GPU 메모리: {gpu_memory:.1f}GB")
            
            # 추천 배치 사이즈
            if gpu_memory > 15:
                batch_size = "12-16"
            elif gpu_memory > 10:
                batch_size = "8-12"
            elif gpu_memory > 6:
                batch_size = "4-8"
            else:
                batch_size = "2-4"
            
            print(f"  - 추천 배치 사이즈: {batch_size}")
            return True
        else:
            print("⚠️ GPU를 사용할 수 없습니다.")
            print("  - CPU 모드로 실행됩니다.")
            print("  - 훈련 속도가 느려질 수 있습니다.")
            return False
    except ImportError:
        print("❌ PyTorch가 설치되지 않았습니다.")
        return False

def check_directories():
    """디렉토리 구조 확인"""
    print("📁 프로젝트 구조 확인 중...")
    
    required_dirs = [
        "1.데이터생성",
        "2.모델훈련", 
        "3.완성모델",
        "임시파일"
    ]
    
    missing_dirs = []
    for directory in required_dirs:
        if not Path(directory).exists():
            missing_dirs.append(directory)
        else:
            print(f"  ✅ {directory}/")
    
    if missing_dirs:
        print(f"⚠️ 누락된 디렉토리: {missing_dirs}")
        print("💡 자동으로 생성합니다...")
        for directory in missing_dirs:
            Path(directory).mkdir(exist_ok=True)
            print(f"  📁 {directory}/ 생성 완료")
    
    return True

def check_model_files():
    """모델 파일 확인"""
    print("🎯 모델 파일 확인 중...")
    
    model_files = [
        "3.완성모델/best.pt",
        "3.완성모델/dataset.yaml",
        "3.완성모델/test_model.py"
    ]
    
    existing_files = []
    missing_files = []
    
    for file_path in model_files:
        if Path(file_path).exists():
            existing_files.append(file_path)
            file_size = Path(file_path).stat().st_size / 1024 / 1024
            print(f"  ✅ {file_path} ({file_size:.1f}MB)")
        else:
            missing_files.append(file_path)
    
    if missing_files:
        print(f"⚠️ 누락된 파일: {len(missing_files)}개")
        for file_path in missing_files:
            print(f"  ❌ {file_path}")
        
        if "best.pt" in str(missing_files):
            print("💡 모델 파일이 없습니다. 다음 방법 중 선택:")
            print("   1. python 2.모델훈련/기본훈련.py (직접 훈련)")
            print("   2. 기존 모델 파일을 3.완성모델/에 복사")
    
    return len(existing_files) > 0

def create_quick_test():
    """빠른 테스트 스크립트 생성"""
    print("🧪 빠른 테스트 스크립트 생성 중...")
    
    test_script = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 홈 CCTV 모델 빠른 테스트 스크립트

사용법:
    python quick_test.py
"""

import os
import sys
from pathlib import Path

def test_environment():
    """환경 테스트"""
    print("🧪 환경 테스트 시작")
    print("=" * 30)
    
    # Python 버전
    print(f"Python: {sys.version}")
    
    # PyTorch 테스트
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("❌ PyTorch가 설치되지 않았습니다.")
        return False
    
    # 필수 패키지 테스트
    packages = ['cv2', 'matplotlib', 'PIL', 'yaml']
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"✅ {pkg}")
        except ImportError:
            print(f"❌ {pkg}")
            
    return True

def test_model():
    """모델 파일 테스트"""
    print("\\n🎯 모델 파일 테스트")
    print("=" * 30)
    
    model_path = Path("3.완성모델/best.pt")
    if not model_path.exists():
        print("❌ 모델 파일이 없습니다.")
        print("💡 다음 중 선택하세요:")
        print("   1. python 2.모델훈련/기본훈련.py")
        print("   2. 모델 파일을 3.완성모델/에 복사")
        return False
    
    try:
        from ultralytics import YOLO
        model = YOLO(str(model_path))
        print(f"✅ 모델 로드 성공: {model_path}")
        print(f"📊 모델 클래스 수: {len(model.names)}")
        print(f"📊 모델 크기: {model_path.stat().st_size / 1024 / 1024:.1f}MB")
        return True
    except Exception as e:
        print(f"❌ 모델 로드 실패: {e}")
        return False

def test_webcam():
    """웹캠 테스트 (선택사항)"""
    print("\\n📹 웹캠 테스트 (선택사항)")
    print("=" * 30)
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ 웹캠 사용 가능")
            ret, frame = cap.read()
            if ret:
                print(f"✅ 프레임 크기: {frame.shape}")
            cap.release()
            return True
        else:
            print("⚠️ 웹캠에 접근할 수 없습니다.")
            return False
    except Exception as e:
        print(f"⚠️ 웹캠 테스트 실패: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("🏠 홈 CCTV 모델 빠른 테스트")
    print("=" * 50)
    
    # 환경 테스트
    env_ok = test_environment()
    
    # 모델 테스트
    model_ok = test_model()
    
    # 웹캠 테스트
    webcam_ok = test_webcam()
    
    # 결과 요약
    print("\\n📊 테스트 결과 요약")
    print("=" * 30)
    print(f"환경 설정: {'✅' if env_ok else '❌'}")
    print(f"모델 파일: {'✅' if model_ok else '❌'}")
    print(f"웹캠 연결: {'✅' if webcam_ok else '⚠️'}")
    
    if env_ok and model_ok:
        print("\\n🎉 모든 테스트 통과!")
        print("🚀 다음 단계:")
        print("   python 3.완성모델/test_model.py --source 0  # 웹캠 테스트")
        print("   python 2.모델훈련/기본훈련.py              # 모델 훈련")
    else:
        print("\\n⚠️ 일부 테스트 실패")
        print("💡 setup.py를 다시 실행하거나 README.md를 참고하세요.")

if __name__ == "__main__":
    main()
'''
    
    with open("quick_test.py", "w", encoding="utf-8") as f:
        f.write(test_script)
    
    print("✅ quick_test.py 생성 완료")

def update_gitignore():
    """.gitignore 업데이트"""
    print("📝 .gitignore 업데이트 중...")
    
    gitignore_content = """# 🏠 홈 CCTV 모델 .gitignore

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# 가상환경
venv/
env/
ENV/
.venv/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# 모델 파일 (용량 큰 파일)
*.pt
*.pth
*.onnx
*.torchscript
*.engine

# 데이터셋 (용량 큰 파일)
coco/train2017.zip
coco/train2017/
*.zip
*.tar.gz

# 임시 파일들
임시파일/
runs/
wandb/
*.log

# 시스템 파일
.DS_Store
Thumbs.db
Desktop.ini

# 사용자 설정
config.yaml
settings.json
"""
    
    with open(".gitignore", "w", encoding="utf-8") as f:
        f.write(gitignore_content)
    
    print("✅ .gitignore 업데이트 완료")

def show_next_steps():
    """다음 단계 안내"""
    print("\n🎉 환경 설정 완료!")
    print("=" * 50)
    print("🚀 다음 단계를 선택하세요:")
    print()
    print("🧪 1. 빠른 테스트")
    print("   python quick_test.py")
    print()
    print("🎯 2. 완성된 모델 테스트 (웹캠)")
    print("   python 3.완성모델/test_model.py --source 0")
    print()
    print("🔧 3. 데이터 준비 (COCO 데이터셋 필요)")
    print("   python 1.데이터생성/make_model.py")
    print("   python 1.데이터생성/coco_to_yolo.py")
    print()
    print("🎓 4. 모델 훈련")
    print("   python 2.모델훈련/기본훈련.py      # 19개 클래스")
    print("   python 2.모델훈련/커스텀훈련.py    # 28개 클래스")
    print()
    print("📖 자세한 내용은 README.md를 참고하세요!")

def main():
    """메인 설정 함수"""
    try:
        # 헤더 출력
        print_header()
        
        # Python 버전 확인
        if not check_python_version():
            return
        
        # 가상환경 확인
        check_virtual_env()
        
        # 패키지 설치
        if not install_packages():
            print("⚠️ 일부 패키지 설치에 실패했습니다.")
            print("💡 수동으로 설치해주세요: pip install -r requirements.txt")
        
        # GPU 환경 확인
        check_gpu()
        
        # 디렉토리 구조 확인
        check_directories()
        
        # 모델 파일 확인
        check_model_files()
        
        # 테스트 스크립트 생성
        create_quick_test()
        
        # .gitignore 업데이트
        update_gitignore()
        
        # 다음 단계 안내
        show_next_steps()
        
    except KeyboardInterrupt:
        print("\n⏹️ 설정이 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 설정 중 오류 발생: {e}")
        print("💡 문제 해결:")
        print("  1. 인터넷 연결 확인")
        print("  2. 가상환경 활성화 확인")
        print("  3. 관리자 권한으로 실행")

if __name__ == "__main__":
    main() 