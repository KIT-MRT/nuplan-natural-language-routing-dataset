"""Reading a generated dataset: immutable opening, and surviving a transient I/O blip."""

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from route_description_generation import dataset_builder as db
from route_description_generation import dataset_index as di


def _small_dataset(tmp):
    paths = db.dataset_paths(Path(tmp), "val14", "lg_routing")
    with db._DatasetWriter(paths) as writer:
        writer.write({"token": "first", "payload": 1})
        writer.write({"token": "second", "payload": 2})
    return paths


class TestImmutableOpening(unittest.TestCase):
    def setUp(self):
        di._HANDLE_CACHE.clear()

    tearDown = setUp

    def test_the_uri_disables_locking(self):
        self.assertTrue(di._immutable_uri("/some/where/idx.sqlite").endswith("?immutable=1"))

    def test_a_path_with_spaces_still_opens(self):
        with tempfile.TemporaryDirectory() as tmp:
            room = Path(tmp) / "a dir with spaces"
            room.mkdir()
            paths = db.dataset_paths(room, "val14", "lg_routing")
            with db._DatasetWriter(paths) as writer:
                writer.write({"token": "first", "payload": 1})
            self.assertEqual(
                di.load_by_token(str(paths.index_file), str(paths.data_file), "first"),
                {"token": "first", "payload": 1},
            )

    def test_the_cached_connection_cannot_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = _small_dataset(tmp)
            di.load_by_token(str(paths.index_file), str(paths.data_file), "first")
            handles = di._HANDLE_CACHE[(str(paths.index_file), str(paths.data_file))]
            with self.assertRaises(sqlite3.OperationalError):
                handles["conn"].execute("INSERT INTO routes VALUES ('x', 0)")

    def test_a_missing_index_says_so_rather_than_no_such_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = _small_dataset(tmp)
            Path(paths.index_file).unlink()
            with self.assertRaisesRegex(FileNotFoundError, "routing index not found"):
                di.load_by_token(str(paths.index_file), str(paths.data_file), "first")


class TestTransientIoIsRetried(unittest.TestCase):
    def setUp(self):
        di._HANDLE_CACHE.clear()

    tearDown = setUp

    def test_one_blip_is_retried_and_the_row_still_comes_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = _small_dataset(tmp)
            real = di._read_by_token
            calls = []

            def flaky(*args):
                calls.append(args)
                if len(calls) == 1:
                    raise sqlite3.OperationalError("disk I/O error")
                return real(*args)

            with mock.patch.object(di, "_read_by_token", flaky), \
                    mock.patch.object(di, "_IO_RETRY_BACKOFF_S", 0.0):
                row = di.load_by_token(str(paths.index_file), str(paths.data_file), "second")

            self.assertEqual(row, {"token": "second", "payload": 2})
            self.assertEqual(len(calls), 2)

    def test_a_persistent_error_is_raised_after_the_last_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = _small_dataset(tmp)
            calls = []

            def broken(*args):
                calls.append(args)
                raise sqlite3.OperationalError("disk I/O error")

            with mock.patch.object(di, "_read_by_token", broken), \
                    mock.patch.object(di, "_IO_RETRY_BACKOFF_S", 0.0):
                with self.assertRaises(sqlite3.OperationalError):
                    di.load_by_token(str(paths.index_file), str(paths.data_file), "first")

            self.assertEqual(len(calls), di._IO_RETRY_ATTEMPTS)

    def test_a_missing_file_is_not_retried(self):
        calls = []

        def missing(*args):
            calls.append(args)
            raise FileNotFoundError("routing index not found: nowhere")

        with mock.patch.object(di, "_read_by_token", missing), \
                mock.patch.object(di, "_IO_RETRY_BACKOFF_S", 0.0):
            with self.assertRaises(FileNotFoundError):
                di.load_by_token("nowhere.sqlite", "nowhere.jsonl", "first")

        self.assertEqual(len(calls), 1)

    def test_the_retry_reconnects_instead_of_reusing_a_broken_handle(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = _small_dataset(tmp)
            key = (str(paths.index_file), str(paths.data_file))
            di.load_by_token(*key, "first")
            first_conn = di._HANDLE_CACHE[key]["conn"]

            state = {"failed": False}
            real = di._read_by_token

            def flaky(*args):
                if not state["failed"]:
                    state["failed"] = True
                    raise sqlite3.OperationalError("disk I/O error")
                return real(*args)

            with mock.patch.object(di, "_read_by_token", flaky), \
                    mock.patch.object(di, "_IO_RETRY_BACKOFF_S", 0.0):
                di.load_by_token(*key, "first")

            self.assertIsNot(di._HANDLE_CACHE[key]["conn"], first_conn)


if __name__ == "__main__":
    unittest.main()
