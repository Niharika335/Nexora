"""Gateway normalizer for event key variations and cumulative vs delta text merging."""

from typing import Any, Dict, Optional, Tuple
from slrag.contracts.events import BaseEvent, TranscriptChunk, UtteranceEnd


class EventNormalizer:
    """Normalizes raw input dictionaries and handles cumulative/delta speech text streams."""

    def __init__(self):
        self._cumulative_text: str = ""

    def normalize_dict_keys(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize common key synonyms across different streaming clients."""
        normalized = dict(raw_data)

        # Map text synonyms
        if "text" not in normalized:
            for alias in ("chunk", "transcript", "content", "message", "utterance"):
                if alias in normalized:
                    normalized["text"] = str(normalized[alias])
                    break

        # Map stream timestamp synonyms (seconds from stream start)
        if "ts_s" not in normalized:
            for alias in ("timestamp_s", "timestamp", "t"):
                if alias in normalized:
                    try:
                        normalized["ts_s"] = float(normalized[alias])
                    except (ValueError, TypeError):
                        pass
                    break

        # Map seq synonyms
        if "seq" not in normalized:
            for alias in ("seq_no", "sequence", "seq_num", "index"):
                if alias in normalized:
                    try:
                        normalized["seq"] = int(normalized[alias])
                    except (ValueError, TypeError):
                        pass
                    break

        # Map utterance_id / turn_id synonyms
        if "turn_id" not in normalized:
            for alias in ("turnId", "turn", "session_turn"):
                if alias in normalized:
                    normalized["turn_id"] = str(normalized[alias])
                    break

        if "utterance_id" not in normalized:
            for alias in ("utteranceId", "utterance_no", "utt_id"):
                if alias in normalized:
                    normalized["utterance_id"] = str(normalized[alias])
                    break

        # Map is_final synonyms
        if "is_final" not in normalized:
            for alias in ("isFinal", "final", "complete"):
                if alias in normalized:
                    normalized["is_final"] = bool(normalized[alias])
                    break

        return normalized

    @staticmethod
    def _norm(text: str) -> str:
        """Lowercase, collapse whitespace, strip leading/trailing ellipses."""
        t = " ".join(text.lower().split())
        for ell in ("…", "..."):
            t = t.strip().removeprefix(ell).removesuffix(ell)
        return t.strip()

    def merge(self, new_text: str, is_cumulative: Optional[bool] = None) -> Tuple[str, str, str]:
        """Merge an incoming chunk into the utterance buffer.

        Cumulative if norm(new) starts with norm(buffer) and is longer (buffer is replaced);
        otherwise delta (appended with one space).

        Returns:
            (delta_text, current_full_text, merge_mode)
        """
        new_text = new_text.strip()

        if is_cumulative is None:
            norm_new, norm_buf = self._norm(new_text), self._norm(self._cumulative_text)
            is_cumulative = bool(norm_buf) and norm_new.startswith(norm_buf) and len(norm_new) > len(norm_buf)

        if is_cumulative:
            if new_text.startswith(self._cumulative_text):
                delta = new_text[len(self._cumulative_text):].strip()
            else:
                delta = new_text
            self._cumulative_text = new_text
            return delta, self._cumulative_text, "cumulative"

        delta = new_text
        self._cumulative_text = f"{self._cumulative_text} {delta}".strip() if self._cumulative_text else delta
        return delta, self._cumulative_text, "delta"

    def process_transcript_text(self, new_text: str, is_cumulative: Optional[bool] = None) -> Tuple[str, str]:
        """Process incoming speech text.

        Returns:
            (delta_text, current_full_text)
        """
        delta, full, _ = self.merge(new_text, is_cumulative)
        return delta, full

    @property
    def buffer(self) -> str:
        return self._cumulative_text

    def reset_utterance(self) -> str:
        """Reset accumulated text state at utterance boundary and return full utterance."""
        full = self._cumulative_text
        self._cumulative_text = ""
        return full
