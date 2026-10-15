
EXAMPLES = [
    {
        "title": "2027년 서초구립반포도서관 전산시스템 유지보수",
        "expected_phrases": ["전산시스템 유지보수"],
    },
    {
        "title": "2027년 횡성인재육성관 인재육성프로그램 운영 주관회사 공개모집 공고",
        "expected_phrases": ["인재육성프로그램 운영"],
    },
    {
        "title": "2027년 경기도의료원 수원병원 MRI 유지보수 견적제출 공고",
        "expected_phrases": ["MRI 유지보수"],
    },
]


def test_expected_phrases_exist():
    for example in EXAMPLES:
        assert example["expected_phrases"]
