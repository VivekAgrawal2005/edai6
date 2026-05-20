class ToneDetector:

    def detect_tone(self, text: str):

        text = text.lower()

        if "issue" in text or "problem" in text:
            return "apologetic"

        if "meeting" in text:
            return "professional"

        if "thank you" in text:
            return "friendly"

        return "professional"