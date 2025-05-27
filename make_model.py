import json
import os
from tqdm import tqdm

# COCO 클래스 ID (홈 CCTV용으로 선택) - 정확한 ID로 수정
홈_CCTV_클래스 = {
    '사람': [1],                     # person
    '반려동물': [16, 17],             # cat, dog  
    '위험물체': [44, 77],             # knife, scissors
    '택배상자': [25, 27, 29],         # backpack, handbag, suitcase
    '귀중품': [63, 64, 65, 67, 68],   # laptop, mouse, remote, keyboard, cell phone
    '가구': [56, 57, 59, 60, 61, 62] # chair, couch, bed, dining table, toilet, tv
}

# ID 목록 평면화
필요한_클래스_ID = []
for ids in 홈_CCTV_클래스.values():
    필요한_클래스_ID.extend(ids)

# 중복 제거
필요한_클래스_ID = list(set(필요한_클래스_ID))

print(f"선택된 클래스 ID: {필요한_클래스_ID}")

# 원본 annotations 파일 불러오기
원본_파일 = 'coco/annotations_trainval2017/annotations/instances_train2017.json'
출력_파일 = '홈CCTV_annotations.json'

print(f"COCO 데이터셋에서 {len(필요한_클래스_ID)}개 클래스 필터링 시작...")

with open(원본_파일, 'r') as f:
    data = json.load(f)

# 선택된 클래스 정보 출력
print("선택된 클래스:")
for 카테고리 in data['categories']:
    if 카테고리['id'] in 필요한_클래스_ID:
        print(f"- ID {카테고리['id']}: {카테고리['name']}")

# 필요한 클래스만 필터링
선별_주석 = []
선별_이미지_ID = set()

for 주석 in tqdm(data['annotations'], desc="주석 필터링"):
    if 주석['category_id'] in 필요한_클래스_ID:
        선별_주석.append(주석)
        선별_이미지_ID.add(주석['image_id'])

print(f"총 {len(선별_주석)}개 주석과 {len(선별_이미지_ID)}개 이미지 선별됨")

# 선별된 이미지만 포함
선별_이미지 = []
for 이미지 in tqdm(data['images'], desc="이미지 필터링"):
    if 이미지['id'] in 선별_이미지_ID:
        선별_이미지.append(이미지)

# 필요한 클래스 카테고리만 포함
선별_카테고리 = []
for 카테고리 in data['categories']:
    if 카테고리['id'] in 필요한_클래스_ID:
        선별_카테고리.append(카테고리)

# 새로운 주석 파일 생성
새_데이터 = {
    'images': 선별_이미지,
    'annotations': 선별_주석,
    'categories': 선별_카테고리
}

# 저장
with open(출력_파일, 'w') as f:
    json.dump(새_데이터, f)

print(f"필터링 완료: {출력_파일} 생성됨")