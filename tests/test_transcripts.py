"""Tests for speaker/timestamp-aware transcript parsing."""

import pytest

from assistant.ingestion.transcripts import has_speaker_markers, parse_transcript

VTT = """\
WEBVTT

1
00:00:01.000 --> 00:00:04.000
<v Alex>Morning everyone, let's start.

2
00:00:09.500 --> 00:00:12.000
<v Priya>I pushed the migration last night.
"""

SRT = """\
1
00:00:01,000 --> 00:00:04,000
Alex: Morning everyone, let's start.

2
00:00:09,500 --> 00:00:12,000
Priya: I pushed the migration last night.
"""

BRACKETED = """\
[00:00:01] Alex: Morning everyone, let's start.
[00:00:09] Priya: I pushed the migration last night.
"""

PROSE = (
    "The quarterly review covered revenue, hiring, and infrastructure. "
    "Revenue grew faster than forecast."
)


class TestHasSpeakerMarkers:
    @pytest.mark.parametrize("text", [VTT, SRT, BRACKETED])
    def test_detects_marked_transcripts(self, text):
        assert has_speaker_markers(text) is True

    def test_detects_speaker_only_lines(self):
        assert has_speaker_markers("Alex: hello\nPriya: hi\n") is True

    def test_detects_bare_timestamp_lines(self):
        assert has_speaker_markers("00:01:02 Alex: hello\n") is True

    def test_rejects_prose(self):
        assert has_speaker_markers(PROSE) is False

    def test_rejects_empty(self):
        assert has_speaker_markers("") is False

    @pytest.mark.parametrize(
        "line",
        [
            "Note: revenue grew this quarter.",
            "Warning: the build is flaky.",
            "Agenda: three items today.",
            "See: the appendix for details.",
        ],
    )
    def test_rejects_prose_lead_ins(self, line):
        assert has_speaker_markers(line) is False

    def test_rejects_long_fragment_with_colon(self):
        text = "In the meeting we agreed on the following three things: ship it."
        assert has_speaker_markers(text) is False

    def test_rejects_url(self):
        assert has_speaker_markers("https://example.com/a/b") is False


class TestParseSpeakerPrefixed:
    def test_bracketed_timestamps(self):
        turns = parse_transcript(BRACKETED)

        assert len(turns) == 2
        assert turns[0].speaker == "Alex"
        assert turns[0].timestamp == "00:00:01"
        assert turns[0].text == "Morning everyone, let's start."
        assert turns[1].speaker == "Priya"

    def test_speaker_without_timestamp(self):
        turns = parse_transcript("Alex: hello there\nPriya: hi\n")

        assert [t.speaker for t in turns] == ["Alex", "Priya"]
        assert turns[0].timestamp is None
        assert turns[0].text == "hello there"

    def test_bare_leading_timestamp(self):
        turns = parse_transcript("00:01:02 Alex: hello there\n")

        assert turns[0].speaker == "Alex"
        assert turns[0].timestamp == "00:01:02"
        assert turns[0].text == "hello there"

    def test_mm_ss_timestamp(self):
        turns = parse_transcript("[12:03] Alex: hello\n")
        assert turns[0].timestamp == "12:03"

    def test_parenthesised_timestamp(self):
        turns = parse_transcript("(00:12:03) Alex: hello\n")
        assert turns[0].timestamp == "00:12:03"

    def test_continuation_lines_join_previous_turn(self):
        raw = "Alex: first part\ncontinued on the next line\nPriya: second\n"
        turns = parse_transcript(raw)

        assert len(turns) == 2
        assert turns[0].text == "first part continued on the next line"

    def test_preamble_before_first_speaker_is_kept(self):
        raw = "Weekly sync notes\nAlex: hello\n"
        turns = parse_transcript(raw)

        assert turns[0].speaker is None
        assert turns[0].text == "Weekly sync notes"
        assert turns[1].speaker == "Alex"

    def test_blank_lines_are_skipped(self):
        turns = parse_transcript("Alex: hello\n\n\nPriya: hi\n")
        assert len(turns) == 2

    def test_multiword_speaker_name(self):
        turns = parse_transcript("Dr. Priya Raman: the results are in\n")
        assert turns[0].speaker == "Dr. Priya Raman"

    def test_prose_lead_in_is_not_a_speaker(self):
        """A 'Note:' line should not be parsed as an utterance."""
        turns = parse_transcript("Note: revenue grew this quarter.")

        assert len(turns) == 1
        assert turns[0].speaker is None


class TestParseCueBlocks:
    def test_vtt_voice_spans(self):
        turns = parse_transcript(VTT, suffix=".vtt")

        assert len(turns) == 2
        assert turns[0].speaker == "Alex"
        assert turns[0].timestamp == "00:00:01"
        assert turns[0].text == "Morning everyone, let's start."
        assert turns[1].speaker == "Priya"
        assert turns[1].timestamp == "00:00:09"

    def test_srt_speaker_prefixed_cues(self):
        turns = parse_transcript(SRT, suffix=".srt")

        assert len(turns) == 2
        assert turns[0].speaker == "Alex"
        assert turns[0].timestamp == "00:00:01"
        assert turns[0].text == "Morning everyone, let's start."

    def test_milliseconds_are_trimmed(self):
        turns = parse_transcript(VTT, suffix=".vtt")
        assert "." not in (turns[0].timestamp or "")

    def test_multiline_cue_is_joined(self):
        raw = (
            "WEBVTT\n\n"
            "00:00:01.000 --> 00:00:04.000\n"
            "Morning everyone,\nlet's start.\n"
        )
        turns = parse_transcript(raw, suffix=".vtt")

        assert len(turns) == 1
        assert turns[0].text == "Morning everyone, let's start."

    def test_unattributed_cues(self):
        raw = "WEBVTT\n\n00:00:01.000 --> 00:00:04.000\nJust some narration.\n"
        turns = parse_transcript(raw, suffix=".vtt")

        assert turns[0].speaker is None
        assert turns[0].timestamp == "00:00:01"

    def test_inline_tags_are_stripped(self):
        raw = "WEBVTT\n\n00:00:01.000 --> 00:00:04.000\n<b>Bold</b> text.\n"
        turns = parse_transcript(raw, suffix=".vtt")

        assert turns[0].text == "Bold text."

    def test_note_blocks_are_skipped(self):
        raw = (
            "WEBVTT\n\nNOTE this is a comment\n\n"
            "00:00:01.000 --> 00:00:04.000\n<v Alex>Hello.\n"
        )
        turns = parse_transcript(raw, suffix=".vtt")

        assert len(turns) == 1
        assert turns[0].speaker == "Alex"

    def test_srt_sequence_numbers_are_not_text(self):
        turns = parse_transcript(SRT, suffix=".srt")
        assert all(not t.text.strip().isdigit() for t in turns)


class TestFallback:
    def test_unmarked_text_becomes_one_turn(self):
        turns = parse_transcript(PROSE)

        assert len(turns) == 1
        assert turns[0].speaker is None
        assert turns[0].timestamp is None
        assert turns[0].text == PROSE

    def test_empty_input_returns_no_turns(self):
        assert parse_transcript("") == []

    def test_whitespace_only_returns_no_turns(self):
        assert parse_transcript("   \n\n  \t ") == []

    def test_cue_file_with_no_cues_falls_back(self):
        turns = parse_transcript("WEBVTT\n\nJust a stray line.\n", suffix=".vtt")

        assert len(turns) == 1
        assert turns[0].speaker is None

    def test_suffix_is_case_insensitive(self):
        turns = parse_transcript(VTT, suffix=".VTT")
        assert turns[0].speaker == "Alex"
