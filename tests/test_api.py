import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_analyze_email_endpoint(client: TestClient) -> None:
    response = client.post(
        "/analyze-email",
        json={
            "email_id": "12345",
            "sender": "client@company.com",
            "subject": "Meeting tomorrow",
            "body": "Can we schedule a meeting tomorrow at 4 PM?",
            "timestamp": "2026-05-16T10:30:00",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["spam"] is False
    assert payload["spam_label"] in {"ham", "suspicious"}
    assert payload["reply_needed"] is True
    assert payload["generated_reply"]


def test_spam_endpoint_blocks_reply(client: TestClient) -> None:
    response = client.post(
        "/analyze-email",
        json={
            "email_id": "spam-1",
            "sender": "promo@spam.com",
            "subject": "Free prize waiting",
            "body": "Click now to claim your free prize and unsubscribe later.",
            "timestamp": "2026-05-16T10:30:00",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["spam"] is True
    assert payload["spam_label"] == "spam"
    assert payload["spam_reasons"]
    assert payload["reply_needed"] is False
    assert payload["generated_reply"] is None


def test_newsletter_is_not_hard_spam(client: TestClient) -> None:
    response = client.post(
        "/analyze-email",
        json={
            "email_id": "newsletter-1",
            "sender": "newsletter@example.com",
            "subject": "Weekly Tech Updates",
            "body": "Here are this week's technology updates, product launches, and AI industry news. Thank you for subscribing to our newsletter.",
            "timestamp": "2026-05-16T10:30:00",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["spam"] is False
    assert payload["spam_label"] in {"ham", "suspicious"}
    assert payload["reply_needed"] is False


def test_phishing_spam_is_detected(client: TestClient) -> None:
    response = client.post(
        "/analyze-email",
        json={
            "email_id": "phish-1",
            "sender": "promo@cheapoffers.com",
            "subject": "WIN FREE GIFT NOW",
            "body": "Congratulations! You have won a free iPhone and shopping voucher. Click the link below immediately to claim your reward.",
            "timestamp": "2026-05-16T10:30:00",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["spam"] is True
    assert payload["spam_label"] == "spam"
    assert payload["reply_needed"] is False
