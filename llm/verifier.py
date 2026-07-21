"""
llm/verifier.py
───────────────
Cross-source news verification layer.

Problem: LLMs are susceptible to hallucination and can be misled by
single-source sensationalist headlines or misquoted figures. Before
feeding news to the sentiment encoder, we verify that key claims are
corroborated by at least N sources.

Verification approach:
  1. Group articles by semantic topic cluster (simple keyword hashing)
  2. A claim is "verified" if it appears in ≥ min_sources articles
  3. Unverified single-source articles are flagged with low confidence
     and passed through with a reduced weight, not discarded entirely
     (discarding could itself be a source of bias)

This is intentionally conservative. In a production system you would
use a proper NLP clustering approach or a dedicated fact-checking API.
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

from llm.news_fetcher import NewsArticle

logger = logging.getLogger(__name__)


@dataclass
class VerifiedArticle:
    article: NewsArticle
    confidence: float       # 0.0–1.0: how well corroborated this article is
    corroboration_count: int  # how many other sources cover a similar story

    def to_prompt_snippet(self) -> str:
        conf_str = f"[confidence: {self.confidence:.0%}]"
        return f"{conf_str}\n{self.article.to_prompt_snippet()}"


class NewsVerifier:
    """
    Filters and scores a list of NewsArticles for cross-source corroboration.

    Parameters
    ----------
    min_sources : articles corroborated by fewer sources get a confidence penalty
    """

    def __init__(self, min_sources: int = 2):
        self.min_sources = min_sources

    def verify(self, articles: List[NewsArticle]) -> List[VerifiedArticle]:
        """
        Score each article by how many other articles cover a similar story.

        Articles from the same source covering the same topic are not counted
        as corroboration (avoids single-source syndication inflation).

        Returns
        -------
        List of VerifiedArticle sorted by confidence DESC, then recency DESC
        """
        if not articles:
            return []

        # Build topic clusters by a rough fingerprint of key title words
        clusters: Dict[str, List[int]] = defaultdict(list)
        for i, article in enumerate(articles):
            key = self._topic_key(article.title)
            clusters[key].append(i)

        # Score each article
        verified: List[VerifiedArticle] = []
        for key, indices in clusters.items():
            # Count unique sources in this cluster
            sources_in_cluster = {articles[i].source for i in indices}
            corroboration = len(sources_in_cluster)

            for i in indices:
                confidence = self._confidence(corroboration)
                verified.append(VerifiedArticle(
                    article=articles[i],
                    confidence=confidence,
                    corroboration_count=corroboration,
                ))

        # Sort: high confidence first, then by recency
        verified.sort(
            key=lambda v: (
                v.confidence,
                v.article.published_utc.timestamp()
                if v.article.published_utc else 0,
            ),
            reverse=True,
        )

        n_high   = sum(1 for v in verified if v.confidence >= 0.7)
        n_medium = sum(1 for v in verified if 0.4 <= v.confidence < 0.7)
        n_low    = sum(1 for v in verified if v.confidence < 0.4)
        logger.info(
            "Verification complete: %d high / %d medium / %d low confidence articles",
            n_high, n_medium, n_low,
        )

        return verified

    def filter_high_confidence(
        self,
        verified: List[VerifiedArticle],
        threshold: float = 0.5,
    ) -> List[VerifiedArticle]:
        """Return only articles above the confidence threshold."""
        return [v for v in verified if v.confidence >= threshold]

    # ── Private ───────────────────────────────────────────────────────────────

    def _topic_key(self, title: str) -> str:
        """
        Produce a rough topic fingerprint from a title.
        Strips common stop words, sorts remaining words, and hashes.
        Two articles with the same dominant keywords map to the same cluster.
        """
        stop_words = {
            "the", "a", "an", "in", "on", "at", "of", "to", "and", "or",
            "for", "is", "are", "was", "were", "has", "have", "had", "with",
            "its", "as", "by", "from", "that", "this", "it",
        }
        words = [
            w.lower().strip(".,!?\"'")
            for w in title.split()
            if w.lower().strip(".,!?\"'") not in stop_words
            and len(w) > 2
        ]
        key_words = sorted(set(words))[:5]  # top 5 unique non-stop words
        raw = " ".join(key_words)
        return hashlib.md5(raw.encode()).hexdigest()[:8]

    def _confidence(self, corroboration: int) -> float:
        """
        Map corroboration count to a confidence score.
        1 source  → 0.3  (treat with scepticism)
        2 sources → 0.65
        3 sources → 0.85
        4+ sources → 1.0
        """
        mapping = {1: 0.30, 2: 0.65, 3: 0.85}
        return mapping.get(corroboration, 1.0)
