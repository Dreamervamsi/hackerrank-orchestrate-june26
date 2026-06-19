"""Allowed output values from problem_statement.md."""

CLAIM_STATUSES: list[str] = [
    "supported",
    "contradicted",
    "not_enough_information",
]

ISSUE_TYPES: list[str] = [
    "dent",
    "scratch",
    "crack",
    "glass_shatter",
    "broken_part",
    "missing_part",
    "torn_packaging",
    "crushed_packaging",
    "water_damage",
    "stain",
    "none",
    "unknown",
]

CAR_OBJECT_PARTS: list[str] = [
    "front_bumper",
    "rear_bumper",
    "door",
    "hood",
    "windshield",
    "side_mirror",
    "headlight",
    "taillight",
    "fender",
    "quarter_panel",
    "body",
    "unknown",
]

LAPTOP_OBJECT_PARTS: list[str] = [
    "screen",
    "keyboard",
    "trackpad",
    "hinge",
    "lid",
    "corner",
    "port",
    "base",
    "body",
    "unknown",
]

PACKAGE_OBJECT_PARTS: list[str] = [
    "box",
    "package_corner",
    "package_side",
    "seal",
    "label",
    "contents",
    "item",
    "unknown",
]

OBJECT_PARTS_BY_CLAIM: dict[str, list[str]] = {
    "car": CAR_OBJECT_PARTS,
    "laptop": LAPTOP_OBJECT_PARTS,
    "package": PACKAGE_OBJECT_PARTS,
}

RISK_FLAGS: list[str] = [
    "none",
    "blurry_image",
    "cropped_or_obstructed",
    "low_light_or_glare",
    "wrong_angle",
    "wrong_object",
    "wrong_object_part",
    "damage_not_visible",
    "claim_mismatch",
    "possible_manipulation",
    "non_original_image",
    "text_instruction_present",
    "user_history_risk",
    "manual_review_required",
]

SEVERITIES: list[str] = [
    "none",
    "low",
    "medium",
    "high",
    "unknown",
]

BOOLEAN_STRINGS: list[str] = ["true", "false"]

PREDICTION_JSON_SCHEMA: dict[str, str] = {
    "evidence_standard_met": "true | false",
    "evidence_standard_met_reason": "short string",
    "risk_flags": "semicolon-separated flags or none",
    "issue_type": "one allowed issue_type",
    "object_part": "one allowed object_part for claim_object",
    "claim_status": "supported | contradicted | not_enough_information",
    "claim_status_justification": "concise image-grounded explanation",
    "supporting_image_ids": "semicolon-separated image IDs or none",
    "valid_image": "true | false",
    "severity": "none | low | medium | high | unknown",
}
