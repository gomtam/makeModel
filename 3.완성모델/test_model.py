# -*- coding: utf-8 -*-
"""
홈 CCTV 객체 인식 모델 테스트 스크립트
YOLOv5 기반 19개 클래스 객체 인식

사용법:
1. python test_model.py --image test_image.jpg
2. python test_model.py --video test_video.mp4
3. python test_model.py --webcam  # 웹캠 실시간 테스트
"""

import cv2
import torch
import numpy as np
import argparse
from pathlib import Path
import time
from ultralytics import YOLO

# 홈 CCTV 클래스 정의
HOME_CCTV_CLASSES = {
    0: 'person',        # 사람
    1: 'cat',           # 고양이
    2: 'dog',           # 개
    3: 'knife',         # 칼
    4: 'scissors',      # 가위
    5: 'backpack',      # 배낭
    6: 'handbag',       # 핸드백
    7: 'suitcase',      # 여행가방
    8: 'laptop',        # 노트북
    9: 'mouse',         # 마우스
    10: 'remote',       # 리모컨
    11: 'keyboard',     # 키보드
    12: 'cell phone',   # 휴대폰
    13: 'chair',        # 의자
    14: 'couch',        # 소파
    15: 'bed',          # 침대
    16: 'dining table', # 식탁
    17: 'toilet',       # 화장실
    18: 'tv'            # TV
}

# 위험 등급 분류
DANGER_LEVELS = {
    'HIGH': ['knife', 'scissors'],           # 높음 - 위험물체
    'MEDIUM': ['person'],                    # 중간 - 사람
    'LOW': ['cat', 'dog'],                   # 낮음 - 반려동물
    'VALUABLE': ['laptop', 'cell phone', 'backpack', 'handbag', 'suitcase'],  # 귀중품
    'NORMAL': ['chair', 'couch', 'bed', 'dining table', 'toilet', 'tv', 'mouse', 'remote', 'keyboard']  # 일반
}

def get_danger_level(class_name):
    """객체의 위험 등급을 반환합니다."""
    for level, classes in DANGER_LEVELS.items():
        if class_name in classes:
            return level
    return 'UNKNOWN'

def get_color_by_danger(danger_level):
    """위험 등급에 따른 색상을 반환합니다."""
    colors = {
        'HIGH': (0, 0, 255),      # 빨강
        'MEDIUM': (0, 165, 255),  # 주황
        'LOW': (0, 255, 0),       # 초록
        'VALUABLE': (255, 0, 255), # 자홍
        'NORMAL': (255, 255, 0),   # 청록
        'UNKNOWN': (128, 128, 128) # 회색
    }
    return colors.get(danger_level, (128, 128, 128))

class HomeCCTVDetector:
    def __init__(self, model_path='best.pt', conf_threshold=0.25):
        """
        홈 CCTV 객체 인식기 초기화
        
        Args:
            model_path: 모델 파일 경로
            conf_threshold: 신뢰도 임계값
        """
        print(f"🔍 모델 로딩 중: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        print(f"✅ 모델 로드 완료! (신뢰도 임계값: {conf_threshold})")
        
        # GPU 사용 가능 여부 확인
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🖥️  사용 장치: {self.device}")

    def detect_objects(self, image, draw_boxes=True):
        """
        이미지에서 객체를 인식합니다.
        
        Args:
            image: 입력 이미지 (numpy array)
            draw_boxes: 바운딩 박스 그리기 여부
            
        Returns:
            결과 이미지, 탐지된 객체 리스트
        """
        # 예측 수행
        results = self.model(image, conf=self.conf_threshold, device=self.device)
        
        detected_objects = []
        result_image = image.copy()
        
        # 결과 처리
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # 박스 정보 추출
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    # 클래스 이름 및 위험 등급
                    class_name = HOME_CCTV_CLASSES.get(class_id, 'Unknown')
                    danger_level = get_danger_level(class_name)
                    
                    # 탐지 정보 저장
                    detected_objects.append({
                        'class_name': class_name,
                        'confidence': confidence,
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'danger_level': danger_level
                    })
                    
                    if draw_boxes:
                        # 바운딩 박스 그리기
                        color = get_color_by_danger(danger_level)
                        
                        # 박스 그리기
                        cv2.rectangle(result_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        
                        # 라벨 텍스트
                        label = f"{class_name} {confidence:.2f} [{danger_level}]"
                        
                        # 텍스트 배경
                        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(result_image, (int(x1), int(y1) - text_height - 10), 
                                    (int(x1) + text_width, int(y1)), color, -1)
                        
                        # 텍스트 그리기
                        cv2.putText(result_image, label, (int(x1), int(y1) - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result_image, detected_objects

    def test_image(self, image_path, save_result=True):
        """이미지 파일 테스트"""
        print(f"📸 이미지 테스트: {image_path}")
        
        # 이미지 로드
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ 이미지를 로드할 수 없습니다: {image_path}")
            return
        
        # 객체 인식
        start_time = time.time()
        result_image, detections = self.detect_objects(image)
        inference_time = time.time() - start_time
        
        # 결과 출력
        print(f"⏱️  추론 시간: {inference_time:.3f}초")
        print(f"🎯 탐지된 객체 수: {len(detections)}")
        
        for i, obj in enumerate(detections, 1):
            print(f"  {i}. {obj['class_name']} (신뢰도: {obj['confidence']:.3f}, 위험도: {obj['danger_level']})")
        
        # 결과 이미지 저장
        if save_result:
            output_path = Path(image_path).stem + "_result.jpg"
            cv2.imwrite(output_path, result_image)
            print(f"💾 결과 저장: {output_path}")
        
        # 결과 이미지 표시
        cv2.imshow('홈 CCTV 객체 인식 결과', result_image)
        print("❌ 종료하려면 아무 키나 누르세요...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def test_video(self, video_path, save_result=True):
        """비디오 파일 테스트"""
        print(f"🎬 비디오 테스트: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"❌ 비디오를 열 수 없습니다: {video_path}")
            return
        
        # 비디오 정보
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📊 비디오 정보: {width}x{height}, {fps}FPS, {total_frames}프레임")
        
        # 결과 비디오 저장 설정
        if save_result:
            output_path = Path(video_path).stem + "_result.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 객체 인식
            result_frame, detections = self.detect_objects(frame)
            
            # 프레임 정보 표시
            info_text = f"Frame: {frame_count}/{total_frames} | Objects: {len(detections)}"
            cv2.putText(result_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 결과 저장
            if save_result:
                out.write(result_frame)
            
            # 화면 표시
            cv2.imshow('홈 CCTV 비디오 분석', result_frame)
            
            # ESC 키로 종료
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                break
            
            # 진행률 출력 (10프레임마다)
            if frame_count % 10 == 0:
                progress = (frame_count / total_frames) * 100
                elapsed = time.time() - start_time
                fps_current = frame_count / elapsed
                print(f"진행률: {progress:.1f}% ({fps_current:.1f} FPS)")
        
        # 정리
        cap.release()
        if save_result:
            out.release()
            print(f"💾 결과 비디오 저장: {output_path}")
        
        cv2.destroyAllWindows()
        
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time
        print(f"✅ 비디오 분석 완료!")
        print(f"⏱️  총 시간: {total_time:.1f}초, 평균 FPS: {avg_fps:.1f}")

    def test_webcam(self):
        """웹캠 실시간 테스트"""
        print("📹 웹캠 실시간 테스트 시작...")
        print("❌ 종료하려면 'q' 키를 누르세요")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ 웹캠을 열 수 없습니다")
            return
        
        # 웹캠 해상도 설정
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 객체 인식
            result_frame, detections = self.detect_objects(frame)
            
            # FPS 계산
            current_time = time.time()
            elapsed = current_time - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            # 정보 표시
            info_text = f"FPS: {fps:.1f} | Objects: {len(detections)}"
            cv2.putText(result_frame, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 탐지된 객체 리스트 표시
            y_offset = 60
            for obj in detections:
                obj_text = f"{obj['class_name']} ({obj['confidence']:.2f}) - {obj['danger_level']}"
                cv2.putText(result_frame, obj_text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 25
            
            # 화면 표시
            cv2.imshow('홈 CCTV 실시간 모니터링', result_frame)
            
            # 'q' 키로 종료
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ 웹캠 테스트 종료")

def main():
    parser = argparse.ArgumentParser(description='홈 CCTV 객체 인식 모델 테스트')
    parser.add_argument('--model', default='best.pt', help='모델 파일 경로')
    parser.add_argument('--image', help='테스트할 이미지 파일')
    parser.add_argument('--video', help='테스트할 비디오 파일')
    parser.add_argument('--webcam', action='store_true', help='웹캠 실시간 테스트')
    parser.add_argument('--conf', type=float, default=0.25, help='신뢰도 임계값')
    
    args = parser.parse_args()
    
    # 모델 초기화
    detector = HomeCCTVDetector(args.model, args.conf)
    
    # 테스트 실행
    if args.image:
        detector.test_image(args.image)
    elif args.video:
        detector.test_video(args.video)
    elif args.webcam:
        detector.test_webcam()
    else:
        print("사용법:")
        print("이미지 테스트: python test_model.py --image test.jpg")
        print("비디오 테스트: python test_model.py --video test.mp4")
        print("웹캠 테스트: python test_model.py --webcam")

if __name__ == "__main__":
    main() 