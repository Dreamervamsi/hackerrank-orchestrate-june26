"""Phase 2 smoke test: prompt build, JSON parser, optional live VLM call."""

from __future__ import annotations

import argparse

from config import SAMPLE_CLAIMS_CSV
from data.loader import load_sample_claims
from data.lookups import EvidenceRequirementsLookup, UserHistoryLookup
from models.hf_vlm import HFVLMClient
from pipeline.prompts import build_claim_user_text, build_system_prompt
from pipeline.reviewer import ClaimReviewer
from schema.json_parser import parse_prediction_json
from schema.columns import OUTPUT_COLUMNS


def test_prompt_and_parser() -> None:
    samples = load_sample_claims(SAMPLE_CLAIMS_CSV)
    history = UserHistoryLookup()
    evidence = EvidenceRequirementsLookup()
    record = samples[0]

    system_prompt = build_system_prompt()
    user_text = build_claim_user_text(
        record,
        history=history.get(record.user_id),
        evidence_requirements=evidence.lookup(record.claim_object, "dent"),
        issue_keyword="dent",
    )

    print("=== System prompt (first 500 chars) ===")
    print(system_prompt[:500], "...\n")

    print("=== User text (first 400 chars) ===")
    print(user_text[:400], "...\n")

    mock_json = (
        '{"evidence_standard_met":"true","evidence_standard_met_reason":"test",'
        '"risk_flags":"none","issue_type":"dent","object_part":"rear_bumper",'
        '"claim_status":"supported","claim_status_justification":"test",'
        '"supporting_image_ids":"img_1","valid_image":"true","severity":"medium"}'
    )
    parsed = parse_prediction_json(mock_json, OUTPUT_COLUMNS)
    print("=== JSON parser OK ===")
    print(parsed)


def test_live_call(limit: int = 1) -> None:
    samples = load_sample_claims(SAMPLE_CLAIMS_CSV)[:limit]
    reviewer = ClaimReviewer(vlm_client=HFVLMClient())
    for record in samples:
        print(f"\n=== Reviewing {record.user_id} ({record.claim_object}) ===")
        missing = [img.image_id for img in record.images if not img.exists]
        if missing:
            print(f"Warning: missing image files for {missing}")
        prediction = reviewer.review(record, issue_keyword="general")
        print(prediction)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Phase 2 VLM integration")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Call Hugging Face Inference Providers (uses HF_TOKEN)",
    )
    parser.add_argument("--limit", type=int, default=1, help="Sample claims to review live")
    args = parser.parse_args()

    test_prompt_and_parser()
    if args.live:
        test_live_call(limit=args.limit)


if __name__ == "__main__":
    main()
