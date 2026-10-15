import json
import re
from pathlib import Path

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "training_data.jsonl"

# 초기 학습용 예시: Dictionary가 아닌 정답 라벨 데이터
SEED_DATA = [
    ("전산시스템 유지보수", ["유지보수"]),
    ("공공데이터 분석 플랫폼 구축", ["구축"]),
    ("정보시스템 유지관리", ["유지관리"]),
    ("AI 기반 예측모델 개발", ["개발"]),
    ("인재육성프로그램 운영", ["운영"]),
    ("병원 MRI 유지보수", ["유지보수"]),
    ("경기도의료원 수원병원 MRI 유지보수 견적제출 공고",
     ["유지보수"]),
    ("행정업무 자동화 시스템 개발 입찰공고", ["개발"]),
    ("데이터 분석 및 시각화", ["분석", "시각화"]),
    ("정보시스템 구축 및 운영지원 용역", ["구축", "운영지원"]),
    ("청사 시설물 유지관리 용역", ["유지관리"]),
    ("인공지능 기반 상담서비스 운영 사업", ["운영"]),
    ("교통정보 시스템 고도화 입찰 공고", ["고도화"]),
    ("연구 결과 보고서 작성 용역", ["작성"]),
]


def normalize_title(title):
    if not isinstance(title, str):
        raise TypeError("공고명은 문자열이어야 합니다.")

    text = re.sub(r"\[\s*재공고\s*\]", " ", title)
    text = re.sub(r"^\s*(?:19|20)\d{2}년\s*", "", text)
    return re.sub(r"\s+", " ", text).strip()


def features(words, i):
    word = words[i]

    return {
        "word": word.lower(),
        "prefix2": word[:2],
        "suffix2": word[-2:],
        "suffix3": word[-3:],
        "position": "last" if i == len(words) - 1 else "other",
        "prev": words[i - 1].lower() if i > 0 else "<START>",
        "next": words[i + 1].lower() if i + 1 < len(words) else "<END>",
    }


def load_training_data():
    records = [
        {"title": title, "actions": actions}
        for title, actions in SEED_DATA
    ]

    if DATA_FILE.exists():
        with DATA_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

    # 같은 공고를 다시 검토했다면 가장 최근 정답 우선
    deduplicated = {}

    for record in records:
        key = normalize_title(record["title"])
        deduplicated[key] = record

    return list(deduplicated.values())


def train_model():
    X, y = [], []

    for record in load_training_data():
        words = normalize_title(record["title"]).split()
        actions = record["actions"]

        for i, word in enumerate(words):
            X.append(features(words, i))
            y.append("ACTION" if word in actions else "OTHER")

    if len(set(y)) < 2:
        raise ValueError("ACTION과 OTHER 정답이 모두 필요합니다.")

    model = Pipeline([
        ("vectorizer", DictVectorizer()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    model.fit(X, y)
    return model


def classify_title(title, model):
    cleaned = normalize_title(title)
    words = cleaned.split()

    if not words:
        return {
            "cleaned_title": cleaned,
            "pairs": [],
            "status": "unresolved"
        }

    X = [features(words, i) for i in range(len(words))]
    predictions = model.predict(X)

    action_indices = [
        i for i, label in enumerate(predictions)
        if label == "ACTION"
    ]

    if not action_indices:
        return {
            "cleaned_title": cleaned,
            "pairs": [],
            "status": "unresolved"
        }

    first_action = min(action_indices)

    return {
        "cleaned_title": cleaned,
        "pairs": [{
            "domain": words[:first_action],
            "action": [words[i] for i in action_indices]
        }],
        "status": "predicted"
    }


def save_feedback(title, actions):
    """사람이 확인한 Action 정답을 누적 저장한다."""

    words = normalize_title(title).split()

    if not all(action in words for action in actions):
        raise ValueError("Action은 공고명에 있는 어절이어야 합니다.")

    record = {
        "title": title,
        "actions": actions
    }

    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)

    with DATA_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

        
DA_DATA_FILE = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "da_training.jsonl"
)


def load_da_data():
    """검증된 Domain/Action 정답 데이터 불러오기"""
    if not DA_DATA_FILE.exists():
        return []

    records = {}
    with DA_DATA_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                key = normalize_title(record["title"])
                records[key] = record

    return list(records.values())


def save_da_feedback(title, domain_indices, action_indices):
    """어절 위치를 이용한 Domain/Action 정답 저장"""
    words = normalize_title(title).split()

    d = set(domain_indices)
    a = set(action_indices)

    if d & a:
        raise ValueError("같은 어절을 Domain과 Action에 중복 지정할 수 없습니다.")

    if any(i < 0 or i >= len(words) for i in d | a):
        raise ValueError("어절 번호가 올바르지 않습니다.")

    labels = [
        "DOMAIN" if i in d
        else "ACTION" if i in a
        else "OTHER"
        for i in range(len(words))
    ]

    record = {
        "title": title,
        "words": words,
        "labels": labels
    }

    DA_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)

    with DA_DATA_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def train_da_model():
    """Domain / Action / Other 다중 분류 학습"""
    records = load_da_data()

    X, y = [], []

    for record in records:
        words = record["words"]
        labels = record["labels"]

        for i in range(len(words)):
            X.append(features(words, i))
            y.append(labels[i])

    if not X or len(set(y)) < 3:
        return None

    model = Pipeline([
        ("vectorizer", DictVectorizer()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    model.fit(X, y)
    return model


def classify_da(title, model):
    """학습된 모델로 Domain과 Action 예측"""
    words = normalize_title(title).split()

    if model is None or not words:
        return {"pairs": [], "status": "untrained"}

    X = [features(words, i) for i in range(len(words))]
    labels = model.predict(X)

    domain = [
        word for word, label in zip(words, labels)
        if label == "DOMAIN"
    ]

    action = [
        word for word, label in zip(words, labels)
        if label == "ACTION"
    ]

    return {
        "pairs": [{"domain": domain, "action": action}]
        if domain and action else [],
        "status": "predicted" if domain and action else "unresolved"
    }
