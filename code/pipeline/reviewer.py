"""End-to-end claim review using prompts, lookups, and the VLM client."""

from __future__ import annotations

from typing import Any

from data.loader import ClaimRecord
from data.lookups import EvidenceRequirementsLookup, UserHistoryLookup
from models.hf_vlm import HFVLMClient
from pipeline.prompts import build_claim_user_text, build_system_prompt


class ClaimReviewer:
    """Review one claim and return structured prediction fields."""

    def __init__(
        self,
        vlm_client: HFVLMClient | None = None,
        history_lookup: UserHistoryLookup | None = None,
        evidence_lookup: EvidenceRequirementsLookup | None = None,
    ) -> None:
        self.vlm = vlm_client or HFVLMClient()
        self.history_lookup = history_lookup or UserHistoryLookup()
        self.evidence_lookup = evidence_lookup or EvidenceRequirementsLookup()

    def review(
        self,
        record: ClaimRecord,
        issue_keyword: str = "general",
    ) -> dict[str, Any]:
        history = self.history_lookup.get(record.user_id)
        evidence = self.evidence_lookup.lookup(record.claim_object, issue_keyword)
        user_text = build_claim_user_text(
            record,
            history=history,
            evidence_requirements=evidence,
            issue_keyword=issue_keyword,
        )
        return self.vlm.complete_json(
            user_text=user_text,
            images=record.images,
            system_prompt=build_system_prompt(),
        )
