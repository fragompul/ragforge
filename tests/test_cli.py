import json
import tempfile
from pathlib import Path

import pytest

from ragforge.cli import main


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "ragforge" in captured.out or "ragforge" in captured.err


def test_cli_help(capsys):
    ret = main([])
    assert ret == 0


def test_cli_ingest_query_and_eval():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        doc_file = tmp_path / "sample.txt"
        doc_file.write_text(
            "The Hubble Space Telescope was launched into low Earth orbit in 1990.",
            encoding="utf-8",
        )

        index_file = tmp_path / "index.json"

        # 1. Test Ingest
        ingest_ret = main(["ingest", str(doc_file), "--index", str(index_file)])
        assert ingest_ret == 0
        assert index_file.exists()

        # 2. Test Ingest missing file
        assert main(["ingest", "non_existent_file.xyz"]) == 1

        # 3. Test Ingest empty directory
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        assert main(["ingest", str(empty_dir)]) == 0

        # 4. Test Query
        query_ret = main(
            ["query", "When was Hubble launched?", "--index", str(index_file), "-k", "1"]
        )
        assert query_ret == 0

        # 5. Test Query with MMR
        mmr_ret = main(
            [
                "query",
                "Hubble launch",
                "--index",
                str(index_file),
                "-k",
                "1",
                "--reranker",
                "mmr",
            ]
        )
        assert mmr_ret == 0

        # 6. Test Query with Noop
        noop_ret = main(
            [
                "query",
                "Hubble launch",
                "--index",
                str(index_file),
                "-k",
                "1",
                "--reranker",
                "none",
            ]
        )
        assert noop_ret == 0

        # 6b. Test Query with --trace prints a per-stage latency trace
        trace_ret = main(
            ["query", "Hubble launch", "--index", str(index_file), "-k", "1", "--trace"]
        )
        assert trace_ret == 0

        # 7. Test Query missing index
        assert main(["query", "test", "--index", "non_existent_idx.json"]) == 1

        # 8. Test Evaluate
        eval_cases_file = tmp_path / "cases.json"
        eval_cases_file.write_text(
            json.dumps([{"query": "Hubble launch date", "relevant_doc_ids": ["sample"]}]),
            encoding="utf-8",
        )
        report_file = tmp_path / "report.json"
        eval_ret = main(
            [
                "evaluate",
                str(eval_cases_file),
                "--index",
                str(index_file),
                "-o",
                str(report_file),
            ]
        )
        assert eval_ret == 0
        assert report_file.exists()

        # 9. Test Evaluate missing files
        assert main(["evaluate", "missing_cases.json", "--index", str(index_file)]) == 1
        assert main(["evaluate", str(eval_cases_file), "--index", "missing_index.json"]) == 1


def test_cli_benchmark():
    ret = main(["benchmark"])
    assert ret == 0


def test_cli_serve_missing_index_returns_error():
    assert main(["serve", "--index", "definitely_missing_idx.json"]) == 1


def test_cli_query_trace_prints_stage_spans(capsys):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        doc_file = tmp_path / "sample.txt"
        doc_file.write_text("Ragforge traces every retrieval stage.", encoding="utf-8")
        index_file = tmp_path / "index.json"

        assert main(["ingest", str(doc_file), "--index", str(index_file)]) == 0
        assert (
            main(["query", "retrieval stages", "--index", str(index_file), "-k", "1", "--trace"])
            == 0
        )

        out = capsys.readouterr().out
        assert "fusion_search" in out
        assert "rerank" in out
        assert "generate" in out
