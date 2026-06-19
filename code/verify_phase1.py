"""Smoke-test Phase 1 data layer: loader + lookups."""

from config import CLAIMS_CSV, SAMPLE_CLAIMS_CSV
from data.loader import load_claims, load_sample_claims
from data.lookups import EvidenceRequirementsLookup, UserHistoryLookup


def main() -> None:
    claims = load_claims(CLAIMS_CSV)
    samples = load_sample_claims(SAMPLE_CLAIMS_CSV)
    history = UserHistoryLookup()
    evidence = EvidenceRequirementsLookup()

    print(f"Loaded {len(claims)} test claims, {len(samples)} sample claims")
    print(f"User history rows: {len(history)}")
    print(f"Evidence requirements: {len(evidence.all)}")

    first = claims[0]
    print(f"\nFirst claim: {first.user_id} ({first.claim_object})")
    print(f"  Images: {[img.image_id for img in first.images]}")
    print(f"  Paths exist: {[img.exists for img in first.images]}")

    hist = history.get(first.user_id)
    if hist:
        print(f"  History flags: {hist.history_flags}")

    reqs = evidence.lookup(first.claim_object, "dent")
    print(f"  Evidence rules for dent: {[r.requirement_id for r in reqs]}")

    labeled = samples[0]
    print(f"\nFirst sample labeled: claim_status={labeled.labels.get('claim_status')}")


if __name__ == "__main__":
    main()
