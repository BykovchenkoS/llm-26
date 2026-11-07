# Langfuse trace links (local)

Base: http://localhost:3000

## LangGraph

- F03 full tree (agent → generations → tool → retriever → context):
  http://localhost:3000/project/lab3/traces/d57678d8c24b161b1e88e486ca074904
- D01 compaction (8061 → 3444, 13 → 7):
  http://localhost:3000/project/lab3/traces/8525f19fc68e6c7dcc4f3acb2aba947c

## OpenCode

- New demo U01 native tree with tags and userId:
  http://localhost:3000/project/lab3/traces/20c0549cae41210ee7e27d87b8b17218
- New demo U01 proxy generation:
  http://localhost:3000/project/lab3/traces/18f467fa4d7526c4c595c61731a165d5
- Historical benchmark F03 native tree (tags empty in that old run):
  http://localhost:3000/project/lab3/traces/ce34eee529f15bd9d87ed9aeff0ea6c7
- Historical F03 proxy generation:
  http://localhost:3000/project/lab3/traces/61ea28a25f18b71b7d0a911715c46676

## MCP retrieval (separate trace)

- New demo U01 `apollo11_retrieval` with proxy session/tags:
  http://localhost:3000/project/lab3/traces/defaec7bc09f5cc5b1a9cc0548f0e9f2
- Historical example without session/tags:
  http://localhost:3000/project/lab3/traces/45e88cdc740f5a254ab14d80ce21b298

Note: native OpenCode session (`ses_...`) and proxy session
(`opencode-U01-...`) differ; they are mapped in
`outputs/demo_opencode_trace_u01.json`. MCP uses the proxy session/tags.
Parent-child trace context across processes is not supported by the installed
plugin/MCP transport, so these are correlated traces, not one tree.

## Context stress 4096 (separate from main comparison)

- D01 turn 3 compaction, 3648 → 1405, span `context_compaction`:
  http://localhost:3000/project/lab3/traces/4e37cc9422f02f0c3e2a0d7ba73eb4ee
- D02 turn 2 compaction, 5283 → 3280, span `context_compaction`:
  http://localhost:3000/project/lab3/traces/1ef4eeabab556c1e5af862dab0f29045
- D02 turn 4 compaction, 3624 → 1560, span `context_compaction`:
  http://localhost:3000/project/lab3/traces/ba37c4912fd727f29fc2535e197c8d46
- Results: `outputs/context_stress_d01_d02_4096.json`
- Screenshot: `docs/screenshots/context_stress_d02.png`

## Scores / run summaries

- LangGraph run summary: http://localhost:3000/project/lab3/traces/7afdbccf260f57b1ab24fa5b56048667
- OpenCode run summary: http://localhost:3000/project/lab3/traces/fd7409f4c0b626b80b01f71aa1cb9278

## Screenshots

- `docs/screenshots/langgraph_f03.png`
- `docs/screenshots/langgraph_d01_compaction.png`
- `docs/screenshots/langgraph_d01_compaction_span.png`
- `docs/screenshots/opencode_f03.png`
- `docs/screenshots/opencode_proxy_generation.png`
- `docs/screenshots/mcp_retrieval.png`
- `docs/screenshots/langfuse_scores_langgraph.png`
- `docs/screenshots/langfuse_scores_opencode.png`

Note: proxy generation, native `opencode.turn`, and MCP retrieval are not
parented into one tree. Correlation is by timestamp/session/prompt.
