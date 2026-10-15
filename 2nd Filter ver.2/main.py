from src.step1 import (
    normalize_title,
    train_da_model,
    classify_da,
    features,
)


def main():
    # 기존 학습 데이터로 모델 준비
    model = train_da_model()

    print("2차 필터링 Step 1 - Domain / Action 분석")
    print("종료하려면 q를 입력하세요.")

    while True:
        title = input("\n공고명 입력: ").strip()

        if title.lower() == "q":
            break

        if not title:
            continue

        # 1. 정규화
        cleaned = normalize_title(title)

        # 2. Domain / Action 분류
        domain = []
        action = []

        if model is not None:
            words = cleaned.split()

            if words:
                X = [
                    features(words, i)
                    for i in range(len(words))
                ]

                labels = model.predict(X)

                domain = [
                    word
                    for word, label in zip(words, labels)
                    if label == "DOMAIN"
                ]

                action = [
                    word
                    for word, label in zip(words, labels)
                    if label == "ACTION"
                ]

        # 3. 결과 출력
        print("\n" + "=" * 60)
        print("[Step 1 분석 결과]")
        print("=" * 60)

        print("\n정규화 결과:", cleaned)
        print("Domain:", domain if domain else "미분류")
        print("Action:", action if action else "미분류")

        print("=" * 60)


if __name__ == "__main__":
    main()