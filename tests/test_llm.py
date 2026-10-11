import json
import unittest
from unittest.mock import Mock, patch

import requests

from src import llm


class OllamaErrorHandlingTests(unittest.TestCase):
    def assert_stream_error(self, exception, message):
        with patch("src.llm.requests.post", side_effect=exception):
            with self.assertRaisesRegex(RuntimeError, message):
                list(llm.stream_answer("test prompt"))

    def test_connection_failure_includes_endpoint_and_cause(self):
        self.assert_stream_error(
            requests.exceptions.ConnectionError("connection refused"),
            r"Could not connect to Ollama.*11434.*connection refused",
        )

    def test_timeout_is_reported_explicitly(self):
        self.assert_stream_error(
            requests.exceptions.Timeout("read timed out"),
            r"timed out after 180 seconds",
        )

    def test_http_model_error_includes_ollama_error_message(self):
        response = Mock()
        response.status_code = 404
        response.json.return_value = {
            "error": "model 'qwen3:8b' not found"
        }
        error = requests.exceptions.HTTPError(
            "404 Client Error",
            response=response,
        )
        self.assert_stream_error(
            error,
            r"HTTP 404: model 'qwen3:8b' not found",
        )

    def test_stream_model_error_is_not_returned_as_an_empty_answer(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.iter_lines.return_value = [
            json.dumps({"error": "model runner failed"}).encode("utf-8")
        ]
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        with patch("src.llm.requests.post", return_value=response):
            with self.assertRaisesRegex(
                RuntimeError,
                r"model 'qwen3:8b': model runner failed",
            ):
                list(llm.stream_answer("test prompt"))


if __name__ == "__main__":
    unittest.main()
