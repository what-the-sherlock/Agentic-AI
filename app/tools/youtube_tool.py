import re
from dataclasses import dataclass
from typing import Optional

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound


@dataclass
class YouTubeTranscriptResult:
    transcript: str
    confidence: float
    error: Optional[str] = None


class YouTubeService:
    YT_REGEX = re.compile(
        r"(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([\w-]{11})"
    )

    def _extract_video_id(self, text: str) -> Optional[str]:
        m = self.YT_REGEX.search(text)
        return m.group(1) if m else None

    def fetch_transcript_from_text(self, text: str) -> YouTubeTranscriptResult:
        vid = self._extract_video_id(text)
        if not vid:
            return YouTubeTranscriptResult(
                transcript="",
                confidence=0.0,
                error="No valid YouTube video ID found.",
            )

        try:
            transcript_list = YouTubeTranscriptApi.get_transcript(vid, languages=["en"])
        except (TranscriptsDisabled, NoTranscriptFound) as e:
            return YouTubeTranscriptResult(
                transcript="",
                confidence=0.0,
                error=f"No transcript available: {e}",
            )
        except Exception as e:
            return YouTubeTranscriptResult(
                transcript="",
                confidence=0.0,
                error=f"Failed to fetch transcript: {e}",
            )

        full_text = " ".join(item["text"] for item in transcript_list).strip()
        return YouTubeTranscriptResult(
            transcript=full_text,
            confidence=0.9,
            error=None,
        )
