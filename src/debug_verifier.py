from src.verifier import Verifier


if __name__ == "__main__":
    print("Creating verifier...")
    verifier = Verifier()
    print("Verifier created")

    evidence_list = [
        {
            "page": "Roman_Atwood",
            "sentence_id": 0,
            "text": "Roman Bernard Atwood is an American YouTube personality, comedian, vlogger and pranker."
        },
        {
            "page": "Roman_Atwood",
            "sentence_id": 3,
            "text": "He also has another YouTube channel called RomanAtwood, where he posts pranks."
        }
    ]

    claims = [
        "Roman Atwood is an American YouTube personality.",
        "Roman Atwood is a content creator.",
        "Roman Atwood is a professional basketball player."
    ]

    for claim in claims:
        print("\n" + "=" * 70)
        print("Claim:", claim)
        print("Running verifier prediction...")
        prediction = verifier.predict(claim, evidence_list)
        print("Prediction:", prediction)