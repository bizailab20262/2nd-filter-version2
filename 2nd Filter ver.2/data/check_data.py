from src.step1 import load_training_data

records = load_training_data()

print(f"\n전체 학습 공고: {len(records)}개")
print("=" * 65)

for i, record in enumerate(records, start=1):
    print(f"\n[{i}] {record['title']}")
    print("Action 정답:", record["actions"])