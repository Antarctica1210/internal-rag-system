from unittest.mock import patch

import pytest

from app.import_process.agent.nodes.node_entry import node_entry
from app.import_process.agent.state import create_default_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_state(local_file_path: str = "", task_id: str = "test-task-001") -> dict:
    return create_default_state(task_id=task_id, local_file_path=local_file_path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNodeEntryPdfFile:
    def test_sets_is_pdf_read_enabled(self):
        state = make_state("docs/report.pdf")
        result = node_entry(state)
        assert result["is_pdf_read_enabled"] is True

    def test_sets_pdf_path(self):
        state = make_state("docs/report.pdf")
        result = node_entry(state)
        assert result["pdf_path"] == "docs/report.pdf"

    def test_does_not_set_md_flag(self):
        state = make_state("docs/report.pdf")
        result = node_entry(state)
        assert result["is_md_read_enabled"] is False

    def test_extracts_file_title(self):
        state = make_state("/some/path/my_report.pdf")
        result = node_entry(state)
        assert result["file_title"] == "my_report"


class TestNodeEntryMdFile:
    def test_sets_is_md_read_enabled(self):
        state = make_state("docs/notes.md")
        result = node_entry(state)
        assert result["is_md_read_enabled"] is True

    def test_sets_md_path(self):
        state = make_state("docs/notes.md")
        result = node_entry(state)
        assert result["md_path"] == "docs/notes.md"

    def test_does_not_set_pdf_flag(self):
        state = make_state("docs/notes.md")
        result = node_entry(state)
        assert result["is_pdf_read_enabled"] is False

    def test_extracts_file_title(self):
        state = make_state("/some/path/my_notes.md")
        result = node_entry(state)
        assert result["file_title"] == "my_notes"


class TestNodeEntryMissingPath:
    def test_returns_state_unchanged_when_path_empty(self):
        state = make_state("")
        result = node_entry(state)
        assert result["is_pdf_read_enabled"] is False
        assert result["is_md_read_enabled"] is False
        assert result["file_title"] == ""

    def test_returns_same_state_object(self):
        state = make_state("")
        result = node_entry(state)
        assert result is state


class TestNodeEntryUnsupportedFileType:
    def test_does_not_set_pdf_or_md_flag(self):
        state = make_state("docs/data.csv")
        result = node_entry(state)
        assert result["is_pdf_read_enabled"] is False
        assert result["is_md_read_enabled"] is False

    def test_still_extracts_file_title(self):
        state = make_state("docs/data.csv")
        result = node_entry(state)
        assert result["file_title"] == "data"


class TestNodeEntryTaskTracking:
    @patch("app.import_process.agent.nodes.node_entry.add_running_task")
    @patch("app.import_process.agent.nodes.node_entry.add_done_task")
    def test_marks_node_running_then_done(self, mock_done, mock_running):
        state = make_state("report.pdf", task_id="task-abc")
        node_entry(state)
        mock_running.assert_called_once_with("task-abc", "node_entry")
        mock_done.assert_called_once_with("task-abc", "node_entry")

    @patch("app.import_process.agent.nodes.node_entry.add_running_task")
    @patch("app.import_process.agent.nodes.node_entry.add_done_task")
    def test_done_not_called_when_path_missing(self, mock_done, mock_running):
        state = make_state("", task_id="task-abc")
        node_entry(state)
        mock_running.assert_called_once_with("task-abc", "node_entry")
        mock_done.assert_not_called()
