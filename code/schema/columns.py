"""CSV column definitions matching problem_statement.md."""

INPUT_COLUMNS: list[str] = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
]

OUTPUT_COLUMNS: list[str] = [
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]

# Full row order for output.csv and labeled sample_claims.csv
SAMPLE_COLUMNS: list[str] = INPUT_COLUMNS + OUTPUT_COLUMNS
