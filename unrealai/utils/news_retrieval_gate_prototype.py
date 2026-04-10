from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional


UTC = timezone.utc


@dataclass
class RetrievalQuery:
    symbol: str
    company_name: str
    as_of: datetime
    market_terms: list[str] = field(default_factory=list)
    company_terms: list[str] = field(default_factory=list)


@dataclass
class ArticleCandidate:
    title: str
    url: str
    source: str
    published_at: Optional[datetime]
    retrieved_at: datetime
    body_text: str
    symbol: str
    query_type: str
    author: str = ""
    summary: str = ""
    revised_at: Optional[datetime] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class GatedArticle:
    title: str
    url: str
    source: str
    published_at: datetime
    body_text: str
    symbol: str
    query_type: str
    summary: str = ""


@dataclass
class RejectedArticle:
    title: str
    url: str
    source: str
    reason: str
    published_at: Optional[datetime] = None
    revised_at: Optional[datetime] = None


@dataclass
class GateResult:
    as_of: datetime
    accepted: list[GatedArticle]
    rejected: list[RejectedArticle]


def ensure_utc(ts: Optional[datetime]) -> Optional[datetime]:
    if ts is None:
        return None
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)


def requires_confirmed_timestamp(article: ArticleCandidate) -> bool:
    return (
        bool(article.url)
        and bool(article.source)
        and article.published_at is not None
    )


def article_is_point_in_time_safe(
    article: ArticleCandidate,
    *,
    as_of: datetime,
    min_publication_lag: timedelta = timedelta(minutes=0),
    reject_revised_articles: bool = False,
) -> tuple[bool, str]:
    as_of = ensure_utc(as_of)
    published_at = ensure_utc(article.published_at)
    revised_at = ensure_utc(article.revised_at)

    if not requires_confirmed_timestamp(article):
        return False, "missing_confirmed_timestamp"
    if published_at is None:
        return False, "published_at_unavailable"
    if published_at > as_of - min_publication_lag:
        return False, "published_after_as_of_cutoff"
    if reject_revised_articles and revised_at is not None:
        return False, "revised_article_rejected"
    if revised_at is not None and revised_at <= as_of:
        return False, "article_revised_before_as_of"
    if not article.body_text or not article.body_text.strip():
        return False, "empty_body_text"
    return True, "accepted"


def gate_articles(
    articles: Iterable[ArticleCandidate],
    *,
    as_of: datetime,
    min_publication_lag: timedelta = timedelta(minutes=0),
    reject_revised_articles: bool = False,
) -> GateResult:
    accepted: list[GatedArticle] = []
    rejected: list[RejectedArticle] = []

    for article in articles:
        ok, reason = article_is_point_in_time_safe(
            article,
            as_of=as_of,
            min_publication_lag=min_publication_lag,
            reject_revised_articles=reject_revised_articles,
        )
        if ok:
            accepted.append(
                GatedArticle(
                    title=article.title,
                    url=article.url,
                    source=article.source,
                    published_at=ensure_utc(article.published_at),
                    body_text=article.body_text,
                    symbol=article.symbol,
                    query_type=article.query_type,
                    summary=article.summary,
                )
            )
        else:
            rejected.append(
                RejectedArticle(
                    title=article.title,
                    url=article.url,
                    source=article.source,
                    reason=reason,
                    published_at=ensure_utc(article.published_at),
                    revised_at=ensure_utc(article.revised_at),
                )
            )

    accepted.sort(key=lambda x: (x.published_at, x.source, x.title))
    return GateResult(as_of=ensure_utc(as_of), accepted=accepted, rejected=rejected)


def llm_retrieval_system_prompt() -> str:
    return (
        "You are a retrieval assistant for point-in-time market research. "
        "Find candidate articles about the named company and broad market context. "
        "Return candidates only; do not make the final safety decision. "
        "Prefer primary reporting with clear publication timestamps and stable URLs. "
        "Do not summarize using hindsight language."
    )


def build_retrieval_queries(
    symbol: str,
    company_name: str,
    *,
    as_of: datetime,
) -> list[RetrievalQuery]:
    return [
        RetrievalQuery(
            symbol=symbol,
            company_name=company_name,
            as_of=ensure_utc(as_of),
            company_terms=[symbol, company_name, f"{company_name} earnings", f"{company_name} guidance"],
            market_terms=["S&P 500", "Nasdaq", "Federal Reserve", "Treasury yields", "market selloff", "market rally"],
        )
    ]


def assemble_overwatch_news_context(
    gate_result: GateResult,
    *,
    max_articles: int = 8,
    max_chars_per_article: int = 2500,
) -> str:
    lines = [
        "Approved point-in-time news context.",
        f"As-of UTC: {gate_result.as_of.isoformat()}",
        f"Approved articles: {min(len(gate_result.accepted), max_articles)}",
    ]
    for idx, article in enumerate(gate_result.accepted[:max_articles], start=1):
        body = article.body_text.strip()
        if len(body) > max_chars_per_article:
            body = body[:max_chars_per_article].rstrip() + "..."
        lines.extend(
            [
                f"[{idx}] {article.title}",
                f"Source: {article.source}",
                f"Published: {article.published_at.isoformat()}",
                f"URL: {article.url}",
                f"Type: {article.query_type}",
                f"Summary: {article.summary.strip()}" if article.summary.strip() else "Summary:",
                f"Body: {body}",
            ]
        )
    return "\n".join(lines)


def save_gate_audit(gate_result: GateResult, path: str | Path) -> None:
    out = {
        "as_of": gate_result.as_of.isoformat(),
        "accepted": [
            {
                **asdict(article),
                "published_at": article.published_at.isoformat(),
            }
            for article in gate_result.accepted
        ],
        "rejected": [
            {
                **asdict(article),
                "published_at": article.published_at.isoformat() if article.published_at else None,
                "revised_at": article.revised_at.isoformat() if article.revised_at else None,
            }
            for article in gate_result.rejected
        ],
    }
    Path(path).write_text(json.dumps(out, indent=2), encoding="utf-8")


def prototype_pipeline(
    *,
    symbol: str,
    company_name: str,
    as_of: datetime,
    candidate_articles: Iterable[ArticleCandidate],
    min_publication_lag_minutes: int = 0,
    reject_revised_articles: bool = False,
) -> dict:
    """
    Prototype wiring for a safe-ish two-stage setup:

    1. Retrieval stage produces candidate articles.
    2. Python gate enforces point-in-time timestamp rules.
    3. Overwatch receives only the gated context.

    `candidate_articles` is intentionally injected here so the trust boundary
    remains in Python rather than in an LLM browsing step.
    """
    queries = build_retrieval_queries(symbol, company_name, as_of=as_of)
    gate_result = gate_articles(
        candidate_articles,
        as_of=as_of,
        min_publication_lag=timedelta(minutes=int(min_publication_lag_minutes)),
        reject_revised_articles=bool(reject_revised_articles),
    )
    overwatch_context = assemble_overwatch_news_context(gate_result)
    return {
        "queries": [asdict(q) for q in queries],
        "accepted_count": len(gate_result.accepted),
        "rejected_count": len(gate_result.rejected),
        "gate_result": gate_result,
        "overwatch_context": overwatch_context,
    }
