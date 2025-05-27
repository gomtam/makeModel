import json

# COCO annotations 파일 로드
with open('coco/annotations_trainval2017/annotations/instances_train2017.json', 'r') as f:
    data = json.load(f)

print("COCO 데이터셋의 모든 클래스:")
print("ID\t클래스명")
print("-" * 30)

for category in data['categories']:
    print(f"{category['id']}\t{category['name']}")

print(f"\n총 {len(data['categories'])}개 클래스")

# 홈 CCTV에 필요한 클래스 찾기
필요한_키워드 = ['person', 'cat', 'dog', 'knife', 'scissors', 'cell phone', 'laptop', 'suitcase', 'backpack', 'handbag']

print(f"\n홈 CCTV 관련 클래스:")
for category in data['categories']:
    if any(keyword in category['name'].lower() for keyword in 필요한_키워드):
        print(f"ID {category['id']}: {category['name']}") 