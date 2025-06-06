import json
import os
from tqdm import tqdm
import shutil

# 경로 설정
coco_annotation_file = '홈CCTV_annotations.json'
output_dir = 'HomeCCTV_dataset'
image_source_dir = 'coco/train2017'  # COCO 이미지가 저장된 경로

# 출력 디렉토리 구조 생성
os.makedirs(f'{output_dir}/images/train', exist_ok=True)
os.makedirs(f'{output_dir}/images/val', exist_ok=True)
os.makedirs(f'{output_dir}/labels/train', exist_ok=True)
os.makedirs(f'{output_dir}/labels/val', exist_ok=True)

# COCO JSON 파일 로드
with open(coco_annotation_file, 'r') as f:
    coco_data = json.load(f)

# 클래스 매핑 생성 (COCO ID -> YOLO 인덱스)
categories = coco_data['categories']
category_id_to_name = {category['id']: category['name'] for category in categories}
category_id_to_index = {category['id']: i for i, category in enumerate(categories)}

# YOLO 데이터셋 설정 파일 생성
with open(f'{output_dir}/data.yaml', 'w') as f:
    f.write(f'train: ./images/train\n')
    f.write(f'val: ./images/val\n\n')
    f.write(f'nc: {len(categories)}\n')
    f.write('names: [')
    for i, category in enumerate(categories):
        if i > 0:
            f.write(', ')
        f.write(f"'{category['name']}'")
    f.write(']\n')

# 이미지 ID -> 파일 이름 매핑
image_id_to_filename = {img['id']: img['file_name'] for img in coco_data['images']}

# 이미지와 주석 처리
images = coco_data['images']
annotations = coco_data['annotations']

# 검증 세트용 이미지 선택 (20%)
val_ratio = 0.2
val_image_ids = set(img['id'] for img in images[:int(len(images) * val_ratio)])

# 이미지와 라벨 처리
for img in tqdm(images, desc='이미지 및 라벨 처리'):
    img_id = img['id']
    file_name = img['file_name']
    img_width = img['width']
    img_height = img['height']
    
    # 훈련/검증 세트 구분
    if img_id in val_image_ids:
        target_img_dir = f'{output_dir}/images/val'
        target_label_dir = f'{output_dir}/labels/val'
    else:
        target_img_dir = f'{output_dir}/images/train'
        target_label_dir = f'{output_dir}/labels/train'
    
    # 이미지 복사 (이미지가 있는 경우에만)
    source_img_path = os.path.join(image_source_dir, file_name)
    if os.path.exists(source_img_path):
        shutil.copy(source_img_path, os.path.join(target_img_dir, file_name))
    
    # 해당 이미지의 모든 주석(annotations) 찾기
    img_annotations = [ann for ann in annotations if ann['image_id'] == img_id]
    
    if len(img_annotations) > 0:
        # YOLO 형식 라벨 파일 생성 (.txt)
        label_file = os.path.splitext(file_name)[0] + '.txt'
        with open(os.path.join(target_label_dir, label_file), 'w') as f:
            for ann in img_annotations:
                category_idx = category_id_to_index[ann['category_id']]
                
                # COCO 형식 (x, y, width, height)를 YOLO 형식으로 변환
                # YOLO 형식: [class_id] [center_x] [center_y] [width] [height] - 모두 0~1 사이 정규화
                x, y, w, h = ann['bbox']
                center_x = (x + w/2) / img_width
                center_y = (y + h/2) / img_height
                norm_width = w / img_width
                norm_height = h / img_height
                
                # 라벨 쓰기
                f.write(f"{category_idx} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}\n")

print(f'YOLO 형식 데이터셋 생성 완료: {output_dir}')

# 결과 요약
image_count = len(images)
annotation_count = len(annotations)
category_count = len(categories)

print(f'\n변환 결과:')
print(f'- 총 이미지: {image_count}개')
print(f'- 총 객체 주석: {annotation_count}개')
print(f'- 클래스 수: {category_count}개')
print(f'- 클래스 목록: {[cat["name"] for cat in categories]}')
print(f'- 훈련/검증 비율: {1-val_ratio:.1f}/{val_ratio:.1f}') 