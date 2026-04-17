from datetime import datetime, timezone
import json
import unittest

from unrealai.utils.news_retrieval_gate_prototype import ArticleCandidate, find_news


class NewsRetrievalGatePrototypeTest(unittest.TestCase):
    def test_gate_rejects_future_publication_and_future_revision(self):
        articles = [
            ArticleCandidate(
                title="Accepted",
                url="https://example.com/accepted",
                source="Example",
                published_at=datetime(2020, 1, 2, 12, tzinfo=timezone.utc),
                retrieved_at=datetime.now(timezone.utc),
                body_text="Known before the as-of date.",
                symbol="AMD",
                query_type="company",
            ),
            ArticleCandidate(
                title="Future",
                url="https://example.com/future",
                source="Example",
                published_at=datetime(2020, 1, 4, 12, tzinfo=timezone.utc),
                retrieved_at=datetime.now(timezone.utc),
                body_text="Published after the as-of date.",
                symbol="AMD",
                query_type="company",
            ),
            ArticleCandidate(
                title="Future revision",
                url="https://example.com/revision",
                source="Example",
                published_at=datetime(2020, 1, 1, 12, tzinfo=timezone.utc),
                revised_at=datetime(2020, 1, 5, 12, tzinfo=timezone.utc),
                retrieved_at=datetime.now(timezone.utc),
                body_text="Current body may include later edits.",
                symbol="AMD",
                query_type="company",
            ),
        ]

        result = find_news("2020-01-03", symbol="AMD", candidate_articles=articles)

        self.assertEqual(["Accepted"], [article.title for article in result.accepted])
        self.assertEqual(
            {
                "Future": "published_after_as_of_cutoff",
                "Future revision": "revised_after_as_of_cutoff",
            },
            {article.title: article.reason for article in result.rejected},
        )

    def test_date_only_candidate_is_conservative_for_intraday_cutoff(self):
        raw_response = json.dumps(
            {
                "candidates": [
                    {
                        "title": "Date only",
                        "url": "https://example.com/date-only",
                        "source": "Example",
                        "published_at": "2020-01-03",
                        "body_text": "No exact publication time.",
                        "symbol": "AMD",
                        "query_type": "company",
                    }
                ]
            }
        )

        result = find_news(
            datetime(2020, 1, 3, 12, tzinfo=timezone.utc),
            symbol="AMD",
            client=_FakeOpenAIClient(raw_response),
        )

        self.assertEqual([], [article.title for article in result.accepted])
        self.assertEqual(["published_after_as_of_cutoff"], [article.reason for article in result.rejected])

    def test_find_news_searches_with_openai_client_then_gates(self):
        raw_response = json.dumps(
            {
                "candidates": [
                    {
                        "title": "Usable search result",
                        "url": "https://example.com/usable",
                        "source": "Example",
                        "published_at": "2020-01-02T12:00:00Z",
                        "body_text": "Useful analyst context.",
                        "symbol": "AMD",
                        "query_type": "company",
                        "image_urls": ["https://example.com/image.jpg"],
                    }
                ]
            }
        )

        client = _FakeOpenAIClient(raw_response)
        result = find_news("2020-01-03", symbol="AMD", company_name="Advanced Micro Devices", client=client)

        self.assertEqual(["Usable search result"], [article.title for article in result.accepted])
        self.assertIn("https://example.com/image.jpg", result.overwatch_context)
        self.assertEqual([{"url": "https://example.com/usable", "title": "Usable search result"}], result.response_sources)
        self.assertEqual([{"type": "web_search"}], client.last_request["tools"])


class _FakeOpenAIClient:
    def __init__(self, output_text):
        self.responses = _FakeResponses(self, output_text)
        self.last_request = None


class _FakeResponses:
    def __init__(self, parent, output_text):
        self._parent = parent
        self._output_text = output_text

    def create(self, **kwargs):
        self._parent.last_request = kwargs
        return _FakeResponse(self._output_text)


class _FakeResponse:
    def __init__(self, output_text):
        self.output_text = output_text

    def model_dump(self):
        return {
            "output": [],
            "sources": [
                {
                    "url": "https://example.com/usable",
                    "title": "Usable search result",
                }
            ],
        }


if __name__ == "__main__":
    unittest.main()
