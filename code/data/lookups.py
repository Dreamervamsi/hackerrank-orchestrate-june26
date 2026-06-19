"""Lookups for user history and evidence requirements."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from config import EVIDENCE_REQUIREMENTS_CSV, USER_HISTORY_CSV


@dataclass(frozen=True)
class UserHistoryRow:
    user_id: str
    past_claim_count: int
    accept_claim: int
    manual_review_claim: int
    rejected_claim: int
    last_90_days_claim_count: int
    history_flags: str
    history_summary: str

    @property
    def flag_list(self) -> list[str]:
        if not self.history_flags or self.history_flags.strip().lower() == "none":
            return []
        return [f.strip() for f in self.history_flags.split(";") if f.strip()]


@dataclass(frozen=True)
class EvidenceRequirement:
    requirement_id: str
    claim_object: str
    applies_to: str
    minimum_image_evidence: str

    @property
    def issue_keywords(self) -> list[str]:
        """Tokenize applies_to into searchable issue keywords."""
        text = self.applies_to.lower()
        parts = re.split(r",|\bor\b|\band\b", text)
        return [p.strip() for p in parts if p.strip()]


# Maps normalized issue types / conversation terms to applies_to fragments.
ISSUE_ALIASES: dict[str, list[str]] = {
    "dent": ["dent"],
    "scratch": ["scratch"],
    "crack": ["crack"],
    "glass_shatter": ["crack", "glass", "shatter", "broken"],
    "broken_part": ["broken", "break", "breakage"],
    "missing_part": ["missing"],
    "torn_packaging": ["torn", "seal", "open"],
    "crushed_packaging": ["crushed", "crush"],
    "water_damage": ["water", "wet"],
    "stain": ["stain", "oil"],
    "general": ["general", "review", "multi-image", "reviewability", "identity", "orientation"],
}


class UserHistoryLookup:
    """user_id -> history row."""

    def __init__(self, csv_path: Path | None = None) -> None:
        path = csv_path or USER_HISTORY_CSV
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        self._by_user: dict[str, UserHistoryRow] = {}
        for _, row in df.iterrows():
            history = UserHistoryRow(
                user_id=str(row["user_id"]),
                past_claim_count=int(row["past_claim_count"]),
                accept_claim=int(row["accept_claim"]),
                manual_review_claim=int(row["manual_review_claim"]),
                rejected_claim=int(row["rejected_claim"]),
                last_90_days_claim_count=int(row["last_90_days_claim_count"]),
                history_flags=str(row["history_flags"]),
                history_summary=str(row["history_summary"]),
            )
            self._by_user[history.user_id] = history

    def get(self, user_id: str) -> UserHistoryRow | None:
        return self._by_user.get(user_id)

    def __contains__(self, user_id: str) -> bool:
        return user_id in self._by_user

    def __len__(self) -> int:
        return len(self._by_user)


class EvidenceRequirementsLookup:
    """claim_object + issue_keyword -> matching evidence requirement rows."""

    def __init__(self, csv_path: Path | None = None) -> None:
        path = csv_path or EVIDENCE_REQUIREMENTS_CSV
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        self._requirements: list[EvidenceRequirement] = [
            EvidenceRequirement(
                requirement_id=str(row["requirement_id"]),
                claim_object=str(row["claim_object"]),
                applies_to=str(row["applies_to"]),
                minimum_image_evidence=str(row["minimum_image_evidence"]),
            )
            for _, row in df.iterrows()
        ]

    @property
    def all(self) -> list[EvidenceRequirement]:
        return list(self._requirements)

    def for_object(self, claim_object: str) -> list[EvidenceRequirement]:
        """All requirements applicable to a claim object (includes claim_object=all)."""
        obj = claim_object.strip().lower()
        return [
            req
            for req in self._requirements
            if req.claim_object.lower() == "all" or req.claim_object.lower() == obj
        ]

    def for_issue(self, claim_object: str, issue_keyword: str) -> list[EvidenceRequirement]:
        """Requirements matching claim_object and an issue keyword or issue_type alias."""
        keyword = issue_keyword.strip().lower()
        if not keyword:
            return self.for_object(claim_object)

        search_terms = {keyword}
        if keyword in ISSUE_ALIASES:
            search_terms.update(ISSUE_ALIASES[keyword])

        matched: list[EvidenceRequirement] = []
        for req in self.for_object(claim_object):
            applies_lower = req.applies_to.lower()
            req_keywords = set(req.issue_keywords)
            if any(term in applies_lower or term in req_keywords for term in search_terms):
                matched.append(req)
        return matched

    def lookup(self, claim_object: str, issue_keyword: str) -> list[EvidenceRequirement]:
        """Primary lookup: claim_object + issue_keyword."""
        return self.for_issue(claim_object, issue_keyword)
