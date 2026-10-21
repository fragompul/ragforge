"""The Claude Desktop extension bundle: manifest shape and a real launch of the packaged server."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from ragforge import RagPipeline
from ragforge._version import __version__

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "build_mcpb.py"


@pytest.fixture(scope="module")
def build_module():
    spec = importlib.util.spec_from_file_location("build_mcpb", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bundle(build_module, tmp_path_factory):
    return build_module.build(tmp_path_factory.mktemp("mcpb") / "ragforge.mcpb")


def test_manifest_has_the_required_fields_and_valid_substitutions(bundle):
    with zipfile.ZipFile(bundle) as zf:
        manifest = json.loads(zf.read("manifest.json"))
    for key in ("manifest_version", "name", "version", "description", "author", "server"):
        assert key in manifest
    assert manifest["version"] == __version__ and manifest["author"]["name"]
    server = manifest["server"]
    assert server["type"] == "python" and server["entry_point"] == "server/main.py"
    config = server["mcp_config"]
    assert config["args"] == ["${__dirname}/server/main.py"]
    declared = set(manifest["user_config"])
    used = {v[len("${user_config.") : -1] for v in config["env"].values()}
    assert used == declared  # every env var maps to a declared setting and vice versa
    assert manifest["user_config"]["index_path"]["required"] is True
    assert manifest["user_config"]["allow_ingest"]["default"] is False  # read-only by default
    assert all(
        f["type"] in {"string", "number", "boolean", "directory", "file"}
        for f in manifest["user_config"].values()
    )


def test_bundle_vendors_the_package_without_bytecode(bundle):
    with zipfile.ZipFile(bundle) as zf:
        names = zf.namelist()
    assert (
        "server/lib/ragforge/__init__.py" in names and "server/lib/ragforge/mcp_server.py" in names
    )
    assert not any("__pycache__" in n or n.endswith(".pyc") for n in names)


def _launch(bundle, tmp_path, env_extra, requests):
    root = tmp_path / "ext"
    with zipfile.ZipFile(bundle) as zf:
        zf.extractall(root)
    env = {"PATH": "/usr/bin:/bin", **env_extra}
    proc = subprocess.run(
        [sys.executable, "-I", str(root / "server" / "main.py")],
        input="\n".join(json.dumps(r) for r in requests) + "\n",
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    return proc


def test_packaged_server_answers_over_stdio_and_is_read_only(bundle, tmp_path):
    index = tmp_path / "kb.json"
    pipeline = RagPipeline()
    pipeline.ingest("eiffel", "The Eiffel Tower is in Paris. It was completed in 1889.")
    pipeline.save(index)
    requests = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18"},
        },
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "ask_knowledge_base",
                "arguments": {"question": "When was the Eiffel Tower completed?"},
            },
        },
    ]
    proc = _launch(
        bundle,
        tmp_path,
        {"RAGFORGE_INDEX": str(index), "RAGFORGE_GUARDRAILS": "sanitize"},
        requests,
    )

    assert proc.returncode == 0, proc.stderr
    replies = [json.loads(line) for line in proc.stdout.splitlines()]
    assert replies[0]["result"]["serverInfo"]["version"] == __version__
    assert "ingest_document" not in [t["name"] for t in replies[1]["result"]["tools"]]
    assert replies[2]["result"]["structuredContent"]["sources"][0]["doc_id"] == "eiffel"


def test_allow_ingest_flag_exposes_the_tool(bundle, tmp_path):
    index = tmp_path / "kb.json"
    RagPipeline().save(index)
    request = [{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}]
    proc = _launch(
        bundle,
        tmp_path,
        {
            "RAGFORGE_INDEX": str(index),
            "RAGFORGE_ALLOW_INGEST": "true",
            "RAGFORGE_GUARDRAILS": "off",
            "RAGFORGE_ROUTE": "true",
        },
        request,
    )
    names = [t["name"] for t in json.loads(proc.stdout)["result"]["tools"]]
    assert "ingest_document" in names


def test_missing_index_is_a_clear_error(bundle, tmp_path):
    proc = _launch(bundle, tmp_path, {}, [])
    assert proc.returncode == 2 and "choose an index file" in proc.stderr
