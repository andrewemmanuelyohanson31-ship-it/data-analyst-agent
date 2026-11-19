from agent.fuzzy import normalize

KNOWN = {"what", "is", "the", "total", "revenue", "which", "with", "city", "has", "highest", "weight", "height",
         "average", "for", "size", "berapa", "yang", "rata-rata", "products", "product", "orders"}
CONTENT = {"total", "revenue", "city", "highest", "weight", "height", "average", "product", "orders"}


def run(q):
    return normalize(q, KNOWN, CONTENT, use_embeddings=False)


def test_typos_and_swapped_letters():
    assert run("What is the totl revenue?")[0] == "What is the total revenue?"
    assert run("Wich city has the hihgest revenue?")[0] == "which city has the highest revenue?"
    assert run("average weigth")[0] == "average weight"  # a swap, not "with"
    assert run("prodcut")[0] == "product"  # not the plural


def test_indonesian_chat_abbreviations():
    q, notes = run("Brp rata2 yg tertinggi?")
    assert q == "berapa rata-rata yang tertinggi?" and len(notes) == 3


def test_unclear_words_are_left_alone():
    q, notes = run("What is the zebra revenue?")
    assert q == "What is the zebra revenue?" and notes == []


def test_real_words_are_never_corrected():
    assert run("more median")[1] == []  # "more"->"most" / "median"->"mean" once changed answers
