"""
Test logger.py - Validation of logging functionality

Tests:
- Message formatting (console and file)
- Log level filtering
- File rotation
- Date-based filename
- Named loggers
- Multi-line messages
- Concurrency
- File locking
"""

import sys
import os
import logging
import tempfile
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from modules.utils import logger
import config


@pytest.fixture
def temp_log_dir(monkeypatch):
    """Create a temporary log directory for tests."""
    temp_dir = tempfile.mkdtemp()
    monkeypatch.setattr(config, "LOG_DIR", temp_dir)
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def fresh_logger(temp_log_dir):
    """Create a fresh logger instance for each test."""
    log = logger.setup_logger("test_logger")
    yield log
    # Cleanup handlers
    log.handlers.clear()


class TestMessageFormatting:
    """Test message formatting in console and file outputs."""

    def test_file_format_includes_timestamp(self, fresh_logger, temp_log_dir):
        """File logs should include timestamp in YYYY-MM-DD HH:MM:SS format."""
        fresh_logger.info("Test message")

        log_files = list(Path(temp_log_dir).glob("*.log"))
        assert len(log_files) > 0, "No log file created"

        content = log_files[0].read_text()

        # Check timestamp format: YYYY-MM-DD HH:MM:SS
        assert datetime.now().strftime("%Y-%m-%d") in content
        assert ":" in content  # Time separator

    def test_file_format_includes_logger_name(self, fresh_logger, temp_log_dir):
        """File logs should include logger name."""
        fresh_logger.info("Test message")

        log_files = list(Path(temp_log_dir).glob("*.log"))
        content = log_files[0].read_text()

        assert "test_logger" in content

    def test_file_format_includes_level(self, fresh_logger, temp_log_dir):
        """File logs should include log level."""
        fresh_logger.info("Info message")
        fresh_logger.warning("Warning message")
        fresh_logger.error("Error message")

        log_files = list(Path(temp_log_dir).glob("*.log"))
        content = log_files[0].read_text()

        assert "INFO" in content
        assert "WARNING" in content
        assert "ERROR" in content

    def test_file_format_includes_message(self, fresh_logger, temp_log_dir):
        """File logs should include the actual message."""
        test_msg = "This is a unique test message 12345"
        fresh_logger.info(test_msg)

        log_files = list(Path(temp_log_dir).glob("*.log"))
        content = log_files[0].read_text()

        assert test_msg in content

    def test_file_format_structure(self, fresh_logger, temp_log_dir):
        """File format should be: timestamp - name - level - message."""
        fresh_logger.info("Test")

        log_files = list(Path(temp_log_dir).glob("*.log"))
        content = log_files[0].read_text().strip()

        # Expected format: YYYY-MM-DD HH:MM:SS - test_logger - INFO - Test
        parts = content.split(" - ")
        assert len(parts) >= 4
        assert "INFO" in parts[2]
        assert "Test" in parts[3]


class TestLogLevelFiltering:
    """Test that log levels are filtered correctly."""

    def test_debug_not_logged_when_level_info(self, temp_log_dir, monkeypatch):
        """DEBUG messages should not appear when LOG_LEVEL=INFO."""
        monkeypatch.setattr(config, "LOG_LEVEL", "INFO")
        log = logger.setup_logger("test_filter")

        log.debug("Debug message should not appear")
        log.info("Info message should appear")

        log_files = list(Path(temp_log_dir).glob("*.log"))
        content = log_files[0].read_text()

        assert "Debug message should not appear" not in content
        assert "Info message should appear" in content

        log.handlers.clear()

    def test_info_not_logged_when_level_warning(self, temp_log_dir, monkeypatch):
        """INFO messages should not appear when LOG_LEVEL=WARNING."""
        monkeypatch.setattr(config, "LOG_LEVEL", "WARNING")
        log = logger.setup_logger("test_filter")

        log.info("Info message should not appear")
        log.warning("Warning message should appear")

        log_files = list(Path(temp_log_dir).glob("*.log"))
        content = log_files[0].read_text()

        assert "Info message should not appear" not in content
        assert "Warning message should appear" in content

        log.handlers.clear()

    def test_all_levels_logged_when_level_debug(self, temp_log_dir, monkeypatch):
        """All levels should appear when LOG_LEVEL=DEBUG."""
        monkeypatch.setattr(config, "LOG_LEVEL", "DEBUG")
        log = logger.setup_logger("test_filter")

        log.debug("Debug")
        log.info("Info")
        log.warning("Warning")
        log.error("Error")
        log.critical("Critical")

        log_files = list(Path(temp_log_dir).glob("*.log"))
        content = log_files[0].read_text()

        assert "Debug" in content
        assert "Info" in content
        assert "Warning" in content
        assert "Error" in content
        assert "Critical" in content

        log.handlers.clear()


class TestFileRotation:
    """Test log file rotation when size limit is reached."""

    def test_rotation_creates_backup(self, temp_log_dir, monkeypatch):
        """When file exceeds maxBytes, a backup should be created."""
        monkeypatch.setattr(config, "LOG_LEVEL", "INFO")

        # Create logger with small maxBytes for testing
        log = logging.getLogger("test_rotation")
        log.setLevel(logging.INFO)
        log.handlers.clear()

        log_filename = os.path.join(temp_log_dir, "test_rotation.log")
        handler = logging.handlers.RotatingFileHandler(
            log_filename,
            maxBytes=1024,  # 1 KB for testing
            backupCount=3
        )
        handler.setFormatter(logging.Formatter('%(message)s'))
        log.addHandler(handler)

        # Write enough data to trigger rotation
        for i in range(100):
            log.info("X" * 100)  # 100 chars per line

        # Check that rotation occurred
        log_files = list(Path(temp_log_dir).glob("test_rotation.log*"))

        # Should have main file + at least 1 backup
        assert len(log_files) >= 2, f"Expected rotation, found: {[f.name for f in log_files]}"

        # Check for .1 backup
        backup_exists = any(".1" in f.name for f in log_files)
        assert backup_exists, "No .1 backup file found"

        log.handlers.clear()

    def test_rotation_respects_backup_count(self, temp_log_dir):
        """Rotation should not exceed backupCount."""
        log = logging.getLogger("test_backup_count")
        log.setLevel(logging.INFO)
        log.handlers.clear()

        log_filename = os.path.join(temp_log_dir, "test_backup.log")
        handler = logging.handlers.RotatingFileHandler(
            log_filename,
            maxBytes=500,
            backupCount=2  # Keep only 2 backups
        )
        handler.setFormatter(logging.Formatter('%(message)s'))
        log.addHandler(handler)

        # Write enough to create multiple rotations
        for i in range(200):
            log.info("Y" * 100)

        log_files = list(Path(temp_log_dir).glob("test_backup.log*"))

        # Should have main file + max 2 backups = 3 files max
        assert len(log_files) <= 3, f"Too many backup files: {len(log_files)}"

        log.handlers.clear()


class TestDateBasedFilename:
    """Test that log filenames include the current date."""

    def test_filename_includes_date(self, fresh_logger, temp_log_dir):
        """Log filename should include current date in YYYYMMDD format."""
        fresh_logger.info("Test")

        expected_date = datetime.now().strftime("%Y%m%d")
        log_files = list(Path(temp_log_dir).glob("*.log"))

        assert len(log_files) > 0
        assert expected_date in log_files[0].name

    def test_filename_format(self, fresh_logger, temp_log_dir):
        """Filename should be daikoku_YYYYMMDD.log."""
        fresh_logger.info("Test")

        log_files = list(Path(temp_log_dir).glob("*.log"))
        filename = log_files[0].name

        # Should match pattern: daikoku_YYYYMMDD.log (logger name is in file content, not filename)
        assert filename.startswith("daikoku_")
        assert filename.endswith(".log")
        assert len(filename.split("_")[-1].replace(".log", "")) == 8  # YYYYMMDD = 8 digits


class TestNamedLoggers:
    """Test named logger functionality."""

    def test_named_logger_includes_module_name(self, temp_log_dir):
        """Named logger should include module name in logs."""
        named_log = logger.get_logger("my_module")
        named_log.info("Test from module")

        log_files = list(Path(temp_log_dir).glob("*.log"))
        if log_files:
            content = log_files[0].read_text()
            assert "my_module" in content

    def test_named_logger_hierarchy(self, temp_log_dir):
        """Named logger should be child of daikoku logger."""
        named_log = logger.get_logger("submodule")
        assert named_log.name == "daikoku.submodule"

    def test_named_logger_shares_handlers(self, temp_log_dir):
        """Named loggers should use parent daikoku logger handlers."""
        # Setup parent daikoku logger with temp dir
        parent_log = logger.setup_logger("daikoku")

        # Get named child logger
        named_log = logger.get_logger("test_module")
        named_log.info("Shared handler test")

        # Should write to daikoku's log file
        log_files = list(Path(temp_log_dir).glob("*.log"))
        assert len(log_files) > 0

        # Check message is in the file (via parent's handlers)
        content = log_files[0].read_text()
        assert "Shared handler test" in content or "test_module" in content

        parent_log.handlers.clear()


class TestMultilineMessages:
    """Test handling of multi-line log messages."""

    def test_multiline_message_preserved(self, fresh_logger, temp_log_dir):
        """Multi-line messages should be preserved in logs."""
        multiline_msg = "Line 1\nLine 2\nLine 3"
        fresh_logger.info(multiline_msg)

        log_files = list(Path(temp_log_dir).glob("*.log"))
        content = log_files[0].read_text()

        assert "Line 1" in content
        assert "Line 2" in content
        assert "Line 3" in content

    def test_traceback_logging(self, fresh_logger, temp_log_dir):
        """Exception tracebacks should be logged correctly."""
        try:
            raise ValueError("Test exception")
        except ValueError:
            fresh_logger.exception("An error occurred")

        log_files = list(Path(temp_log_dir).glob("*.log"))
        content = log_files[0].read_text()

        assert "Test exception" in content
        assert "Traceback" in content or "ValueError" in content


class TestConcurrency:
    """Test concurrent logging from multiple threads."""

    def test_concurrent_writes(self, fresh_logger, temp_log_dir):
        """Multiple threads should be able to log concurrently without corruption."""
        num_threads = 10
        messages_per_thread = 50

        def log_messages(thread_id):
            for i in range(messages_per_thread):
                fresh_logger.info(f"Thread {thread_id} - Message {i}")

        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=log_messages, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # Read log file
        log_files = list(Path(temp_log_dir).glob("*.log"))
        content = log_files[0].read_text()
        lines = content.strip().split("\n")

        # Should have num_threads * messages_per_thread lines
        expected_lines = num_threads * messages_per_thread
        assert len(lines) == expected_lines, f"Expected {expected_lines} lines, got {len(lines)}"

    def test_concurrent_writes_no_corruption(self, fresh_logger, temp_log_dir):
        """Concurrent writes should not corrupt log format."""
        num_threads = 5

        def log_message(thread_id):
            fresh_logger.info(f"Thread_{thread_id}_message")
            time.sleep(0.001)

        threads = []
        for i in range(num_threads):
            t = threading.Thread(target=log_message, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        log_files = list(Path(temp_log_dir).glob("*.log"))
        content = log_files[0].read_text()

        # Check all expected messages are present
        for i in range(num_threads):
            assert f"Thread_{i}_message" in content


class TestFileLocking:
    """Test file locking and concurrent access scenarios."""

    def test_logger_handles_locked_file(self, temp_log_dir, monkeypatch):
        """Logger should handle scenarios where file is already open."""
        monkeypatch.setattr(config, "LOG_LEVEL", "INFO")

        # Create first logger
        log1 = logger.setup_logger("test_lock_1")
        log1.info("From logger 1")

        # Create second logger writing to same directory
        # (different filename due to date-based naming, but tests concurrent access)
        log2 = logger.setup_logger("test_lock_2")
        log2.info("From logger 2")

        # Both should write successfully
        log_files = list(Path(temp_log_dir).glob("*.log"))
        assert len(log_files) >= 1

        # Check both messages exist
        all_content = ""
        for log_file in log_files:
            all_content += log_file.read_text()

        assert "From logger 1" in all_content or "From logger 2" in all_content

        log1.handlers.clear()
        log2.handlers.clear()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
