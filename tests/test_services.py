from app.models.schemas import EmailInput
from app.services.heuristics import predict_intent_rule
from app.services.spam_service import SpamClassifier
from app.services.reply_generator import ReplyGenerator
from app.services.rule_engine import RuleEngine


def test_intent_heuristic_detects_meeting_request() -> None:
    intent, confidence = predict_intent_rule("Meeting tomorrow", "Can we schedule a meeting tomorrow at 4 PM?")
    assert intent in {"meeting_request", "scheduling"}
    assert confidence > 0.2


def test_rule_engine_blocks_spam_replies() -> None:
    engine = RuleEngine()
    decision = engine.should_generate_reply(True, True, "meeting_request")
    assert not decision.allow_reply


def test_reply_generator_returns_template() -> None:
    generator = ReplyGenerator()
    reply = generator.generate("meeting_request", "client@company.com", "Meeting tomorrow", "Can we schedule a meeting tomorrow at 4 PM?", "12345")
    assert reply is not None
    assert "meeting" in reply.lower() or "available" in reply.lower()


def test_spam_classifier_flags_obvious_spam() -> None:
    classifier = SpamClassifier()
    result = classifier.predict(
        "promo@cheapoffers.com",
        "WIN FREE GIFT NOW",
        "Congratulations! You have won a free iPhone and shopping voucher. Click the link below immediately to claim your reward.",
    )
    assert result.spam is True
    assert result.spam_label == "spam"
    assert result.spam_confidence >= 0.75
    assert any(reason for reason in result.spam_reasons)


def test_spam_classifier_keeps_business_email_as_ham() -> None:
    classifier = SpamClassifier()
    result = classifier.predict(
        "alice@example.com",
        "Meeting schedule",
        "Lets have a meeting about our service and discount discussion",
    )
    assert result.spam is False
    assert result.spam_label in {"ham", "suspicious"}
