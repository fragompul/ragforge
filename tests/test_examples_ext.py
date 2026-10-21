"""The newer examples run as integration tests so they never silently rot."""

import importlib
import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
sys.path.insert(0, str(EXAMPLES_DIR))


def test_guardrails_demo():
    result = importlib.import_module("guardrails_demo").main()
    assert result["injection_risk"] >= 0.5
    assert "IGNORE ALL PREVIOUS" not in result["poisoned_text"]
    assert "jane.doe@example.com" not in result["record_text"]
    assert result["prompt_is_spotlighted"]


def test_citations_and_agent():
    result = importlib.import_module("citations_and_agent").main()
    assert 0 < result["groundedness"] < 1
    assert any("2031" in s for s in result["unsupported"])
    assert result["multi_hop_docs"] == ["error-codes", "rrf"]
    assert result["abstained"] is True


def test_mcp_client_demo():
    result = importlib.import_module("mcp_client_demo").main()
    assert result["server"] == "ragforge"
    assert "ask_knowledge_base" in result["tools"]
    assert result["cited_doc"] == "error-codes"


def test_cache_and_judge():
    result = importlib.import_module("cache_and_judge").main()
    assert (result["first"], result["second"]) == ("miss", "hit")
    assert result["entries_after_ingest"] == 0
    assert result["faithfulness"] == 1.0
