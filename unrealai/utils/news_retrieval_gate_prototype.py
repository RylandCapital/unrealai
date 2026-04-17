from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date as date_type
from datetime import datetime, time, timedelta, timezone
import os
from pathlib import Path
import re
from typing import Any, Iterable, Optional
import urllib.error
import urllib.parse
import urllib.request

from dateutil import parser as date_parser
from dotenv import load_dotenv


UTC = timezone.utc
load_dotenv(Path(__file__).resolve().parents[2] / ".env")
load_dotenv()
DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
DEFAULT_NEWS_RETRIEVAL_MODEL = os.getenv(
    "NEWS_RETRIEVAL_MODEL",
    os.getenv("OVERWATCH_MODEL", os.getenv("MODEL", "gpt-5-mini")),
)
DEFAULT_NEWS_RETRIEVAL_PROVIDER = os.getenv("NEWS_RETRIEVAL_PROVIDER", "gdelt").strip().lower()
DEFAULT_GDELT_LOOKBACK_DAYS = int(os.getenv("NEWS_RETRIEVAL_GDELT_LOOKBACK_DAYS", "14"))
DEFAULT_TAVILY_LOOKBACK_DAYS = int(os.getenv("NEWS_RETRIEVAL_TAVILY_LOOKBACK_DAYS", "14"))
DEFAULT_TAVILY_SEARCH_DEPTH = os.getenv("NEWS_RETRIEVAL_TAVILY_SEARCH_DEPTH", "basic").strip().lower()
DEFAULT_TAVILY_TOPIC = os.getenv("NEWS_RETRIEVAL_TAVILY_TOPIC", "news").strip().lower()
DEFAULT_TAVILY_INCLUDE_RAW_CONTENT = os.getenv("NEWS_RETRIEVAL_TAVILY_INCLUDE_RAW_CONTENT", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
    "on",
    "text",
}
GDELT_DOC_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


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
    image_urls: list[str] = field(default_factory=list)
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
    image_urls: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


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


@dataclass
class NewsRetrievalResult:
    symbol: str
    company_name: str
    as_of: datetime
    queries: list[RetrievalQuery]
    candidates: list[ArticleCandidate]
    gate_result: GateResult
    raw_response_text: str = ""
    response_sources: list[dict] = field(default_factory=list)
    overwatch_context: str = ""

    @property
    def accepted(self) -> list[GatedArticle]:
        return self.gate_result.accepted

    @property
    def rejected(self) -> list[RejectedArticle]:
        return self.gate_result.rejected


def ensure_utc(ts: Optional[datetime]) -> Optional[datetime]:
    if ts is None:
        return None
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)


def normalize_as_of(value: datetime | date_type | str) -> datetime:
    """
    Normalize a training/Overwatch as-of value to a UTC instant.

    Date-only values are interpreted as end-of-day UTC so an as-of like
    "2021-05-04" means "information available through that calendar date."
    Pass an explicit datetime if the decision point needs a market-close cutoff.
    """
    if isinstance(value, datetime):
        return ensure_utc(value)
    if isinstance(value, date_type):
        return datetime.combine(value, time.max, tzinfo=UTC)

    text = str(value).strip()
    if not text:
        raise ValueError("as_of date is required")
    if DATE_ONLY_RE.match(text):
        parsed_date = datetime.strptime(text, "%Y-%m-%d").date()
        return datetime.combine(parsed_date, time.max, tzinfo=UTC)
    return ensure_utc(date_parser.parse(text))


def parse_datetime_utc(value: Any, *, date_only_end: bool = False) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return ensure_utc(value)
    if isinstance(value, date_type):
        clock = time.max if date_only_end else time.min
        return datetime.combine(value, clock, tzinfo=UTC)

    text = str(value).strip()
    if not text or text.lower() in {"null", "none", "unknown", "n/a"}:
        return None
    if DATE_ONLY_RE.match(text):
        parsed_date = datetime.strptime(text, "%Y-%m-%d").date()
        clock = time.max if date_only_end else time.min
        return datetime.combine(parsed_date, clock, tzinfo=UTC)
    return ensure_utc(date_parser.parse(text))


def safe_parse_datetime_utc(value: Any, *, date_only_end: bool = False) -> Optional[datetime]:
    try:
        return parse_datetime_utc(value, date_only_end=date_only_end)
    except (TypeError, ValueError, OverflowError):
        return None


def parse_gdelt_datetime(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if re.match(r"^\d{14}$", text):
        return datetime.strptime(text, "%Y%m%d%H%M%S").replace(tzinfo=UTC)
    return safe_parse_datetime_utc(text)


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
    as_of = normalize_as_of(as_of)
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
    if revised_at is not None and revised_at > as_of:
        return False, "revised_after_as_of_cutoff"
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
    as_of = normalize_as_of(as_of)
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
                    image_urls=list(article.image_urls),
                    metadata=dict(article.metadata),
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
        "You are a retrieval assistant for point-in-time equity research. "
        "Act like an analyst gathering actionable news that could help a long-only "
        "shop decide whether to be LONG or FLAT as of a historical date. "
        "Find candidate articles about the named company, sector, and broad market context. "
        "Return candidates only; Python will make the final date-safety decision. "
        "Prefer primary reporting with clear publication timestamps and stable URLs. "
        "Never use hindsight language or facts that would not have been known at the as-of time."
    )


def build_retrieval_queries(
    symbol: str,
    company_name: str,
    *,
    as_of: datetime,
) -> list[RetrievalQuery]:
    symbol = str(symbol or "").strip().upper()
    company_name = str(company_name or symbol).strip()
    return [
        RetrievalQuery(
            symbol=symbol,
            company_name=company_name,
            as_of=normalize_as_of(as_of),
            company_terms=[
                symbol,
                company_name,
                f"{company_name} earnings",
                f"{company_name} guidance",
                f"{company_name} analyst rating",
                f"{company_name} product launch",
                f"{company_name} lawsuit investigation",
            ],
            market_terms=["S&P 500", "Nasdaq", "Federal Reserve", "Treasury yields", "market selloff", "market rally"],
        )
    ]


def build_llm_news_search_prompt(
    *,
    symbol: str,
    company_name: str,
    as_of: datetime,
    queries: list[RetrievalQuery],
    max_candidates: int,
) -> str:
    as_of = normalize_as_of(as_of)
    query_payload = [
        {
            "symbol": q.symbol,
            "company_name": q.company_name,
            "as_of_utc": normalize_as_of(q.as_of).isoformat(),
            "company_terms": q.company_terms,
            "market_terms": q.market_terms,
        }
        for q in queries
    ]
    schema = {
        "candidates": [
            {
                "title": "Article headline",
                "url": "Stable article URL",
                "source": "Publisher name",
                "published_at": "ISO-8601 timestamp or YYYY-MM-DD if only date is available",
                "revised_at": "ISO-8601 timestamp if visible, otherwise null",
                "author": "Author if visible, otherwise empty string",
                "summary": "1-3 sentences focused on long/flat relevance as known then",
                "body_text": "Relevant excerpt or condensed article text, without future hindsight",
                "symbol": symbol,
                "query_type": "company|sector|macro|market",
                "image_urls": ["Article image URLs if visible"],
                "metadata": {
                    "timestamp_evidence": "Short note describing where the timestamp came from",
                    "why_actionable": "Why it may matter for a long/flat decision",
                },
            }
        ]
    }
    return (
        f"As-of cutoff: {as_of.isoformat()} UTC.\n"
        f"Target symbol: {str(symbol or '').strip().upper()}.\n"
        f"Company name: {str(company_name or symbol).strip()}.\n\n"
        "Find up to "
        f"{int(max_candidates)} candidate news items that could have mattered to a point-in-time "
        "long/flat equity decision. Search for company-specific catalysts first, then sector and "
        "broad-market context only if it is likely to affect the target stock.\n\n"
        "Hard retrieval rules:\n"
        "- Include only items with a visible publication timestamp or publication date.\n"
        "- Do not decide whether an item passes the final as-of gate; Python will check timestamps.\n"
        "- Do not add later-known outcomes, later market moves, or hindsight framing.\n"
        "- If a source was revised after publication and the revised timestamp is visible, return revised_at.\n"
        "- If images are visible in the search result or article metadata, include their direct URLs.\n"
        "- Return JSON only, with no markdown fences.\n\n"
        f"Search plan:\n{json.dumps(query_payload, default=str, indent=2)}\n\n"
        f"Required JSON shape:\n{json.dumps(schema, indent=2)}"
    )


def _get_openai_client(api_key: str | None = None):
    key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("SECRET")
    if not key:
        raise RuntimeError("Missing API key. Set OPENAI_API_KEY or SECRET.")
    from openai import OpenAI

    return OpenAI(api_key=key)


def _response_output_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return str(text)
    data = _response_to_dict(response)
    chunks: list[str] = []
    for item in data.get("output", []) if isinstance(data, dict) else []:
        for part in item.get("content", []) if isinstance(item, dict) else []:
            if isinstance(part, dict) and part.get("type") in {"output_text", "text"}:
                chunks.append(str(part.get("text", "")))
    return "\n".join(c for c in chunks if c)


def _response_to_dict(response: Any) -> dict:
    if response is None:
        return {}
    if isinstance(response, dict):
        return response
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "to_dict"):
        return response.to_dict()
    return {}


def _collect_response_sources(response: Any) -> list[dict]:
    data = _response_to_dict(response)
    found: list[dict] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if isinstance(node.get("sources"), list):
                for item in node["sources"]:
                    if isinstance(item, dict):
                        found.append(dict(item))
            if node.get("type") == "url_citation":
                found.append(dict(node))
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(data)
    deduped: list[dict] = []
    seen: set[str] = set()
    for item in found:
        key = str(item.get("url") or item.get("uri") or item.get("title") or item)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _extract_json_payload(text: str) -> Any:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    starts = [idx for idx in (cleaned.find("{"), cleaned.find("[")) if idx >= 0]
    for start in sorted(starts):
        try:
            obj, _ = decoder.raw_decode(cleaned[start:])
            return obj
        except json.JSONDecodeError:
            continue
    raise ValueError("Could not parse JSON candidate payload from LLM response.")


def _coerce_str(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _coerce_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        text = _coerce_str(item)
        if text:
            out.append(text)
    return out


def _coerce_metadata(value: Any) -> dict:
    return dict(value) if isinstance(value, dict) else {}


def _candidate_from_mapping(
    item: dict,
    *,
    symbol: str,
    retrieved_at: datetime,
) -> ArticleCandidate:
    published_at = safe_parse_datetime_utc(item.get("published_at") or item.get("published"), date_only_end=True)
    revised_at = safe_parse_datetime_utc(
        item.get("revised_at") or item.get("updated_at") or item.get("modified_at"),
        date_only_end=True,
    )
    article_symbol = _coerce_str(item.get("symbol")) or str(symbol or "").strip().upper()
    metadata = _coerce_metadata(item.get("metadata"))
    timestamp_evidence = item.get("timestamp_evidence")
    if timestamp_evidence and "timestamp_evidence" not in metadata:
        metadata["timestamp_evidence"] = timestamp_evidence
    return ArticleCandidate(
        title=_coerce_str(item.get("title")),
        url=_coerce_str(item.get("url")),
        source=_coerce_str(item.get("source") or item.get("publisher")),
        published_at=published_at,
        retrieved_at=retrieved_at,
        body_text=_coerce_str(item.get("body_text") or item.get("text") or item.get("excerpt")),
        symbol=article_symbol,
        query_type=_coerce_str(item.get("query_type")) or "company",
        author=_coerce_str(item.get("author")),
        summary=_coerce_str(item.get("summary")),
        revised_at=revised_at,
        image_urls=_coerce_str_list(item.get("image_urls") or item.get("images")),
        metadata=metadata,
    )


def parse_llm_candidate_articles(
    raw_text: str,
    *,
    symbol: str = "",
    retrieved_at: datetime | None = None,
) -> list[ArticleCandidate]:
    payload = _extract_json_payload(raw_text)
    if isinstance(payload, dict):
        items = payload.get("candidates") or payload.get("articles") or payload.get("news") or []
    elif isinstance(payload, list):
        items = payload
    else:
        items = []

    retrieved_at = ensure_utc(retrieved_at or datetime.now(tz=UTC))
    candidates: list[ArticleCandidate] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        candidate = _candidate_from_mapping(item, symbol=symbol, retrieved_at=retrieved_at)
        dedupe_key = (candidate.url or f"{candidate.source}:{candidate.title}").strip().lower()
        if not dedupe_key or dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        candidates.append(candidate)
    return candidates


def _gdelt_datetime_arg(value: datetime) -> str:
    return ensure_utc(value).strftime("%Y%m%d%H%M%S")


def _gdelt_query_for_terms(terms: list[str]) -> str:
    cleaned = []
    for term in terms:
        text = str(term or "").strip()
        if not text:
            continue
        text = re.sub(r"\s+", " ", text)
        if " " in text and not (text.startswith('"') and text.endswith('"')):
            text = f'"{text}"'
        cleaned.append(text)
    return " OR ".join(cleaned[:8])


def _gdelt_queries_for_terms(terms: list[str], *, limit: int = 3) -> list[str]:
    queries = []
    seen = set()
    for term in terms:
        text = str(term or "").strip()
        if not text:
            continue
        text = re.sub(r"\s+", " ", text)
        if " " in text and not (text.startswith('"') and text.endswith('"')):
            text = f'"{text}"'
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        queries.append(text)
        if len(queries) >= int(limit):
            break
    return queries


def _read_json_url(url: str, *, timeout: int = 20) -> dict:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "unrealai-news-retrieval/1.0",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def gdelt_search_candidate_articles(
    *,
    symbol: str,
    company_name: str = "",
    as_of: datetime | date_type | str,
    max_candidates: int = 10,
    lookback_days: int | None = None,
) -> tuple[list[ArticleCandidate], str, list[dict]]:
    """
    Free local retrieval layer for Spark/Ollama runs.

    GDELT supplies timestamped article URLs; Python still applies the final
    point-in-time gate before any context reaches Overwatch.
    """
    as_of = normalize_as_of(as_of)
    symbol = str(symbol or "").strip().upper()
    company_name = str(company_name or symbol).strip()
    lookback_days = int(DEFAULT_GDELT_LOOKBACK_DAYS if lookback_days is None else lookback_days)
    start = as_of - timedelta(days=max(1, lookback_days))
    retrieved_at = datetime.now(tz=UTC)
    queries = build_retrieval_queries(symbol, company_name, as_of=as_of)

    search_queries: list[tuple[str, str]] = []
    for query in queries:
        if symbol == "SPY":
            for gdelt_query in _gdelt_queries_for_terms(query.market_terms, limit=3):
                search_queries.append(("market", gdelt_query))
        else:
            company_terms = [company_name, f"{company_name} earnings", f"{symbol} stock"]
            for gdelt_query in _gdelt_queries_for_terms(company_terms, limit=3):
                search_queries.append(("company", gdelt_query))
            for gdelt_query in _gdelt_queries_for_terms(query.market_terms, limit=2):
                search_queries.append(("market", gdelt_query))

    candidates: list[ArticleCandidate] = []
    response_sources: list[dict] = []
    raw_payloads: list[dict] = []
    seen: set[str] = set()
    per_query_records = max(5, int(max_candidates))

    for query_type, gdelt_query in search_queries:
        if not gdelt_query:
            continue
        params = {
            "query": gdelt_query,
            "mode": "ArtList",
            "format": "json",
            "sort": "HybridRel",
            "maxrecords": str(per_query_records),
            "startdatetime": _gdelt_datetime_arg(start),
            "enddatetime": _gdelt_datetime_arg(as_of),
        }
        url = f"{GDELT_DOC_API_URL}?{urllib.parse.urlencode(params)}"
        try:
            payload = _read_json_url(url)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raw_payloads.append({"query": gdelt_query, "error": str(exc)})
            continue

        articles = payload.get("articles") or []
        raw_payloads.append({"query": gdelt_query, "article_count": len(articles)})
        for item in articles:
            if not isinstance(item, dict):
                continue
            article_url = _coerce_str(item.get("url"))
            title = _coerce_str(item.get("title"))
            if not article_url or not title:
                continue
            key = article_url.lower()
            if key in seen:
                continue
            seen.add(key)

            domain = _coerce_str(item.get("domain") or item.get("source") or urllib.parse.urlparse(article_url).netloc)
            published_at = parse_gdelt_datetime(item.get("seendate"))
            image_url = _coerce_str(item.get("socialimage"))
            summary = f"GDELT matched this {query_type} item for query: {gdelt_query}"
            candidates.append(
                ArticleCandidate(
                    title=title,
                    url=article_url,
                    source=domain or "GDELT",
                    published_at=published_at,
                    retrieved_at=retrieved_at,
                    body_text=f"{title}. {summary}",
                    symbol=symbol,
                    query_type=query_type,
                    summary=summary,
                    image_urls=[image_url] if image_url else [],
                    metadata={
                        "provider": "gdelt",
                        "query": gdelt_query,
                        "language": _coerce_str(item.get("language")),
                        "source_country": _coerce_str(item.get("sourcecountry")),
                        "timestamp_evidence": "GDELT seendate",
                    },
                )
            )
            response_sources.append({"url": article_url, "title": title, "provider": "gdelt"})
            if len(candidates) >= int(max_candidates):
                return candidates, json.dumps(raw_payloads, default=str), response_sources

    return candidates[: int(max_candidates)], json.dumps(raw_payloads, default=str), response_sources[: int(max_candidates)]


def _get_tavily_client(api_key: str | None = None):
    key = api_key or os.getenv("TAVILY_API_KEY")
    if not key:
        raise RuntimeError("Missing Tavily API key. Set TAVILY_API_KEY.")
    from tavily import TavilyClient

    return TavilyClient(api_key=key)


def _tavily_image_urls(item: dict) -> list[str]:
    images = item.get("images")
    if not images:
        return []
    if not isinstance(images, list):
        images = [images]

    urls: list[str] = []
    for image in images:
        if isinstance(image, dict):
            text = _coerce_str(image.get("url"))
        else:
            text = _coerce_str(image)
        if text:
            urls.append(text)
    return urls


def tavily_search_candidate_articles(
    *,
    symbol: str,
    company_name: str = "",
    as_of: datetime | date_type | str,
    api_key: str | None = None,
    client: Any = None,
    max_candidates: int = 10,
    lookback_days: int | None = None,
) -> tuple[list[ArticleCandidate], str, list[dict]]:
    """
    Tavily-backed retrieval for local/Gemma Overwatch runs.

    Tavily does the web/news search; Python still owns the final timestamp gate.
    Results without a parseable publication date stay rejected by the gate.
    """
    as_of = normalize_as_of(as_of)
    symbol = str(symbol or "").strip().upper()
    company_name = str(company_name or symbol).strip()
    lookback_days = int(DEFAULT_TAVILY_LOOKBACK_DAYS if lookback_days is None else lookback_days)
    start = as_of - timedelta(days=max(1, lookback_days))
    retrieved_at = datetime.now(tz=UTC)
    queries = build_retrieval_queries(symbol, company_name, as_of=as_of)

    search_queries: list[tuple[str, str]] = []
    for query in queries:
        if symbol == "SPY":
            for term in _gdelt_queries_for_terms(query.market_terms, limit=3):
                search_queries.append(("market", term))
        else:
            company_terms = [company_name, f"{company_name} earnings", f"{symbol} stock"]
            for term in _gdelt_queries_for_terms(company_terms, limit=3):
                search_queries.append(("company", term))
            for term in _gdelt_queries_for_terms(query.market_terms, limit=2):
                search_queries.append(("market", term))

    tavily_client = client or _get_tavily_client(api_key=api_key)
    candidates: list[ArticleCandidate] = []
    response_sources: list[dict] = []
    raw_payloads: list[dict] = []
    seen: set[str] = set()

    search_depth = DEFAULT_TAVILY_SEARCH_DEPTH if DEFAULT_TAVILY_SEARCH_DEPTH else "basic"
    topic = DEFAULT_TAVILY_TOPIC if DEFAULT_TAVILY_TOPIC else "news"
    include_raw_content: bool | str = "text" if DEFAULT_TAVILY_INCLUDE_RAW_CONTENT else False
    per_query_records = min(20, max(5, int(max_candidates)))

    for query_type, tavily_query in search_queries:
        if not tavily_query:
            continue
        try:
            payload = tavily_client.search(
                query=tavily_query,
                max_results=per_query_records,
                search_depth=search_depth,
                topic=topic,
                start_date=start.date().isoformat(),
                end_date=as_of.date().isoformat(),
                include_answer=False,
                include_raw_content=include_raw_content,
                include_images=True,
            )
        except Exception as exc:
            raw_payloads.append({"query": tavily_query, "provider": "tavily", "error": str(exc)})
            continue

        results = payload.get("results") if isinstance(payload, dict) else []
        raw_payloads.append(
            {
                "query": tavily_query,
                "provider": "tavily",
                "result_count": len(results or []),
                "response_time": payload.get("response_time") if isinstance(payload, dict) else None,
                "request_id": payload.get("request_id") if isinstance(payload, dict) else None,
            }
        )
        for item in results or []:
            if not isinstance(item, dict):
                continue
            article_url = _coerce_str(item.get("url"))
            title = _coerce_str(item.get("title"))
            if not article_url or not title:
                continue
            key = article_url.lower()
            if key in seen:
                continue
            seen.add(key)

            published_at = safe_parse_datetime_utc(
                item.get("published_date") or item.get("published_at") or item.get("date"),
                date_only_end=True,
            )
            raw_content = _coerce_str(item.get("raw_content"))
            content = _coerce_str(item.get("content"))
            body_text = raw_content or content or title
            source = _coerce_str(urllib.parse.urlparse(article_url).netloc) or "Tavily"
            summary = f"Tavily matched this {query_type} item for query: {tavily_query}"
            candidates.append(
                ArticleCandidate(
                    title=title,
                    url=article_url,
                    source=source,
                    published_at=published_at,
                    retrieved_at=retrieved_at,
                    body_text=body_text,
                    symbol=symbol,
                    query_type=query_type,
                    summary=summary,
                    image_urls=_tavily_image_urls(item),
                    metadata={
                        "provider": "tavily",
                        "query": tavily_query,
                        "score": item.get("score"),
                        "timestamp_evidence": "Tavily published_date",
                        "search_depth": search_depth,
                        "topic": topic,
                    },
                )
            )
            response_sources.append({"url": article_url, "title": title, "provider": "tavily"})
            if len(candidates) >= int(max_candidates):
                return candidates, json.dumps(raw_payloads, default=str), response_sources

    return candidates[: int(max_candidates)], json.dumps(raw_payloads, default=str), response_sources[: int(max_candidates)]


def llm_search_candidate_articles(
    *,
    symbol: str,
    company_name: str = "",
    as_of: datetime | date_type | str,
    client: Any = None,
    api_key: str | None = None,
    model: str | None = None,
    reasoning_effort: str | None = None,
    max_candidates: int = 10,
    provider: str | None = None,
) -> tuple[list[ArticleCandidate], str, list[dict]]:
    as_of = normalize_as_of(as_of)
    symbol = str(symbol or "").strip().upper()
    company_name = str(company_name or symbol).strip()
    queries = build_retrieval_queries(symbol, company_name, as_of=as_of)
    prompt = build_llm_news_search_prompt(
        symbol=symbol,
        company_name=company_name,
        as_of=as_of,
        queries=queries,
        max_candidates=max_candidates,
    )

    provider = (provider or DEFAULT_NEWS_RETRIEVAL_PROVIDER).strip().lower()
    if provider == "tavily" and client is None:
        return tavily_search_candidate_articles(
            symbol=symbol,
            company_name=company_name,
            as_of=as_of,
            max_candidates=max_candidates,
        )
    if client is None and not api_key and provider in {"gdelt", "local", "ollama"}:
        return gdelt_search_candidate_articles(
            symbol=symbol,
            company_name=company_name,
            as_of=as_of,
            max_candidates=max_candidates,
        )

    client = client or _get_openai_client(api_key=api_key)
    request_kwargs = {
        "model": model or DEFAULT_NEWS_RETRIEVAL_MODEL,
        "tools": [{"type": "web_search"}],
        "tool_choice": "auto",
        "include": ["web_search_call.action.sources"],
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": llm_retrieval_system_prompt()},
                    {"type": "input_text", "text": prompt},
                ],
            }
        ],
    }
    if reasoning_effort:
        request_kwargs["reasoning"] = {"effort": str(reasoning_effort)}

    response = client.responses.create(**request_kwargs)
    raw_text = _response_output_text(response).strip()
    candidates = parse_llm_candidate_articles(raw_text, symbol=symbol, retrieved_at=datetime.now(tz=UTC))
    return candidates[: int(max_candidates)], raw_text, _collect_response_sources(response)[: int(max_candidates)]


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
                f"Images: {', '.join(article.image_urls)}" if article.image_urls else "Images:",
                f"Body: {body}",
            ]
        )
    return "\n".join(lines)


def _serialize_dataclass_with_datetimes(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, list):
        return [_serialize_dataclass_with_datetimes(item) for item in value]
    if isinstance(value, dict):
        return {key: _serialize_dataclass_with_datetimes(item) for key, item in value.items()}
    if hasattr(value, "__dataclass_fields__"):
        return _serialize_dataclass_with_datetimes(asdict(value))
    return value


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


def save_news_retrieval_audit(result: NewsRetrievalResult, path: str | Path) -> None:
    out = _serialize_dataclass_with_datetimes(result)
    Path(path).write_text(json.dumps(out, indent=2), encoding="utf-8")


def find_news(
    as_of: datetime | date_type | str,
    symbol: str = "",
    company_name: str = "",
    *,
    candidate_articles: Iterable[ArticleCandidate] | None = None,
    client: Any = None,
    api_key: str | None = None,
    model: str | None = None,
    reasoning_effort: str | None = None,
    provider: str | None = None,
    max_candidates: int = 10,
    max_context_articles: int = 8,
    min_publication_lag_minutes: int = 0,
    reject_revised_articles: bool = False,
    audit_path: str | Path | None = None,
) -> NewsRetrievalResult:
    """
    Search for candidate news, then return only articles passing the Python date gate.

    The LLM/web-search stage is allowed to be broad and opportunistic. The gate is
    the trust boundary: no candidate reaches Overwatch unless it has a URL, source,
    parseable publication timestamp, non-empty text, and a timestamp at or before
    the normalized as-of cutoff.
    """
    as_of = normalize_as_of(as_of)
    symbol = str(symbol or "").strip().upper()
    company_name = str(company_name or symbol).strip()
    queries = build_retrieval_queries(symbol, company_name, as_of=as_of)
    raw_response_text = ""
    response_sources: list[dict] = []

    if candidate_articles is None:
        candidates, raw_response_text, response_sources = llm_search_candidate_articles(
            symbol=symbol,
            company_name=company_name,
            as_of=as_of,
            client=client,
            api_key=api_key,
            model=model,
            reasoning_effort=reasoning_effort,
            provider=provider,
            max_candidates=max_candidates,
        )
    else:
        candidates = list(candidate_articles)

    gate_result = gate_articles(
        candidates,
        as_of=as_of,
        min_publication_lag=timedelta(minutes=int(min_publication_lag_minutes)),
        reject_revised_articles=bool(reject_revised_articles),
    )
    overwatch_context = assemble_overwatch_news_context(
        gate_result,
        max_articles=int(max_context_articles),
    )
    result = NewsRetrievalResult(
        symbol=symbol,
        company_name=company_name,
        as_of=as_of,
        queries=queries,
        candidates=candidates,
        gate_result=gate_result,
        raw_response_text=raw_response_text,
        response_sources=response_sources,
        overwatch_context=overwatch_context,
    )
    if audit_path:
        save_news_retrieval_audit(result, audit_path)
    return result


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
    result = find_news(
        as_of,
        symbol=symbol,
        company_name=company_name,
        candidate_articles=candidate_articles,
        min_publication_lag_minutes=min_publication_lag_minutes,
        reject_revised_articles=reject_revised_articles,
    )
    return {
        "queries": [asdict(q) for q in result.queries],
        "accepted_count": len(result.accepted),
        "rejected_count": len(result.rejected),
        "gate_result": result.gate_result,
        "overwatch_context": result.overwatch_context,
        "result": result,
    }
