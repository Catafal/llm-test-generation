# Decontamination Report — data/train vs heldout

Generated 2026-09-07. Floor date 2026-06-01. Candidates 586, kept 550, families 88.

## Layers and thresholds

- n-gram: word-level, n=10; any shared n-gram with a reference function flags.
- AST: identifiers→slots, docstrings dropped, constants→type; exact hash flags; k=5 node shingles Jaccard ≥ 0.85 flags.
- embedding: jinaai/jina-embeddings-v2-base-code; cosine ≥ 0.9 flags, 0.8–0.9 listed for human review.
- family: same repository owner, plus union-find over every flagged candidate pair; one function kept per family cluster of near-duplicates.

## Removed against heldout

11 candidates removed.

- `3169657175/gpt-webcodex:resources/coding-tools-mcp/coding_tools_mcp/protocol.py::response_id`: ngram10 x1 shepherd-agents/shepherd:commons-vcs/src/commons_vcs/_types.py::_normalize_json_value
- `ARahim3/mlx-dspark:src/mlx_dspark/anthropic_api.py::_blocks_to_text`: ngram10 x3 icesixgod/codex-trajectory:plugins/codex-trajectory/scripts/codex_trajectory/privacy.py::content_text
- `Birfy/agentdescent:agentdescent/treestrategy.py::_first_json_object`: ngram10 x4 microsoft/Mage:mage_flow/models/modules/mage_text.py::_extract_json_object
- `EvolvingLMMs-Lab/SkillOpt-Lite:copilot_example/alfworld/vendor/alfworld_projection.py::alfworld_projection`: ngram10 x45 jinyangwu/SEED:agent_system/environments/env_package/alfworld/projection.py::alfworld_projection
- `Glitch-Cat-Club/graph-memory-starter:digest/session_end.py::text_blocks`: ngram10 x4 icesixgod/codex-trajectory:plugins/codex-trajectory/scripts/codex_trajectory/privacy.py::content_text
- `PKU-YuanGroup/OpenAI4S:openai4s/agent/progress_circuit.py::canonical_arguments`: ngram10 x8 shepherd-agents/shepherd:commons-vcs/src/commons_vcs/_types.py::_normalize_json_value
- `addsumtech/slides_maker:scripts/scan_secrets.py::entropy`: ngram10 x1 2akouwu/reverify:reverify/binary.py::shannon_entropy
- `mikehasa/agentacct:src/agentacct/receipt.py::evidence_coverage_ledger`: ngram10 x1 huggingface/tau:src/tau_ai/http_errors.py::provider_error_detail_from_mapping
- `mikehasa/agentacct:src/agentacct/receipt.py::plan_share_headline`: ngram10 x1 huggingface/tau:src/tau_ai/http_errors.py::provider_error_detail_from_mapping
- `mikehasa/agentacct:src/agentacct/subagent_roles.py::_first_text`: ngram10 x3 lbx154/Argus:argus_skill/adapters/stream_progress.py::_extract_text
- `zhangxunvvv-ux/chatgpt2apiEditV141:services/content_filter.py::_resolve_fail_open`: ngram10 x1 shepherd-agents/shepherd:shepherd/packages/core/src/shepherd_core/_shared/coerce.py::_coerce_to_bool

## Review band (kept, human check requested)

- `ARahim3/mlx-dspark:src/mlx_dspark/anthropic_api.py::convert_tools`: REVIEW cosine 0.85 robocurve/inspect-robots:plugins/inspect-robots-agent/src/inspect_robots_agent/_anthropic.py::_translate_tools
- `Birfy/agentdescent:agentdescent/treestrategy.py::_first_json_object`: REVIEW cosine 0.81 adam-s/car-diagnosis:src/cardiag/pipeline/llm.py::parse_json; REVIEW cosine 0.83 microsoft/Mage:mage_flow/models/modules/mage_text.py::_extract_json_object
- `EvolvingLMMs-Lab/SkillOpt-Lite:copilot_example/alfworld/vendor/alfworld_projection.py::alfworld_projection`: REVIEW cosine 0.89 jinyangwu/SEED:agent_system/environments/env_package/alfworld/projection.py::alfworld_projection
- `PromptPartner/agentsmith:scripts/test-agent-conformance.py::parse_frontmatter`: REVIEW cosine 0.82 Encod3d-Sec/TORCH:scripts/archive/retag_iatt.py::parse_frontmatter
- `addsumtech/slides_maker:scripts/scan_secrets.py::entropy`: REVIEW cosine 0.84 2akouwu/reverify:reverify/binary.py::shannon_entropy
- `ai4s-research/ai4s-skills:skills/mindmap-render/scripts/generate_mindmap.py::sanitize_filename`: REVIEW cosine 0.89 GangTailorUpgrade/undress-service:coomertool/utils.py::sanitize_filename
- `ai4s-research/ai4s-skills:tools/validate_skills.py::parse_frontmatter`: REVIEW cosine 0.81 Encod3d-Sec/TORCH:scripts/archive/retag_iatt.py::parse_frontmatter; REVIEW cosine 0.86 HKUDS/OpenOPC:.opc/skills/skill-creator/scripts/quick_validate.py::_parse_simple_frontmatter
- `dl1683/irys-stateful-swarms:src/loop/llm.py::parse_json`: REVIEW cosine 0.80 adam-s/car-diagnosis:src/cardiag/pipeline/llm.py::parse_json; REVIEW cosine 0.83 microsoft/Mage:mage_flow/models/modules/mage_text.py::_extract_json_object

## Near-duplicate merges within the pool

- ngram10: Skyvern-AI/rustwright:python/rustwright/_agent/attach.py::page_target_id  ~  Skyvern-AI/rustwright:python/rustwright/_agent/refs.py::_page_target_id
- ast-exact: shuxiachai/academic-commercialization-agent:openalex_role_directed_live.py::_compatible_url_owner  ~  shuxiachai/academic-commercialization-agent:openalex_role_gap_evaluation.py::_compatible_url_owner
- ngram10: ARahim3/mlx-dspark:src/mlx_dspark/anthropic_api.py::convert_tools  ~  MindLab-Research/Mixture-of-LoRA-Harness:mol_harness/responses.py::responses_tools_to_oai_tools
- ngram10: Floe-Labs/floe-guard:src/floe_guard/integrations/anthropic.py::_usage_from  ~  Floe-Labs/floe-guard:src/floe_guard/integrations/gemini.py::_usage_from
- ngram10: Floe-Labs/floe-guard:src/floe_guard/integrations/anthropic.py::_usage_from  ~  Floe-Labs/floe-guard:src/floe_guard/integrations/langgraph.py::_usage_from_update
- ngram10: Floe-Labs/floe-guard:src/floe_guard/integrations/gemini.py::_usage_from  ~  Floe-Labs/floe-guard:src/floe_guard/integrations/langgraph.py::_usage_from_update
- ngram10: Floe-Labs/floe-guard:src/floe_guard/integrations/langchain.py::_usage_from_result  ~  Rodiun/frugon:src/frugon/measure.py::_extract_usage
- ast-exact: Floe-Labs/floe-guard:src/floe_guard/integrations/litellm.py::_cached_tokens  ~  Floe-Labs/floe-guard:src/floe_guard/integrations/openai.py::_cached_tokens
- ngram10: omagents/omagents:skills/deep-research/scripts/audit.py::check_missing_sources  ~  omagents/omagents:skills/deep-research/scripts/audit.py::check_conflicting_data
- ngram10: omagents/omagents:skills/deep-research/scripts/audit.py::check_missing_sources  ~  omagents/omagents:skills/deep-research/scripts/audit.py::check_coverage_gaps
- ngram10: omagents/omagents:skills/deep-research/scripts/audit.py::check_missing_sources  ~  omagents/omagents:skills/deep-research/scripts/audit.py::check_source_duplicates
- ngram10: omagents/omagents:skills/deep-research/scripts/audit.py::check_conflicting_data  ~  omagents/omagents:skills/deep-research/scripts/audit.py::check_coverage_gaps
- ngram10: omagents/omagents:skills/deep-research/scripts/audit.py::check_conflicting_data  ~  omagents/omagents:skills/deep-research/scripts/audit.py::check_source_duplicates
- ngram10: omagents/omagents:skills/deep-research/scripts/audit.py::check_coverage_gaps  ~  omagents/omagents:skills/deep-research/scripts/audit.py::check_source_duplicates
- ngram10: oncologylab/fp-tools:scripts/audit_docs.py::check_expanded_aggregate_layout  ~  oncologylab/fp-tools:scripts/audit_docs.py::check_grouped_aggregate_legend
- ast-exact: oncologylab/fp-tools:src/fp_tools/tools/atacorrect.py::_sample_worker_plan  ~  oncologylab/fp-tools:src/fp_tools/tools/score_bigwig.py::_sample_worker_plan
- ngram10: landonbtw/NostalgiaBox:nostalgiabox/overlay.py::_filled_rect  ~  landonbtw/NostalgiaBox:nostalgiabox/overlay.py::_dot
- ngram10: AuuCoder/gptGrok2api:app/dataplane/reverse/protocol/xai_chat.py::classify_line  ~  AuuCoder/gptGrok2api:app/dataplane/reverse/protocol/xai_console_chat.py::classify_console_line
- ngram10: AuuCoder/gptGrok2api:app/dataplane/reverse/protocol/xai_image_edit.py::extract_streaming_response  ~  AuuCoder/gptGrok2api:app/dataplane/reverse/protocol/xai_image_edit.py::extract_model_response_urls
- ngram10: AuuCoder/gptGrok2api:app/dataplane/reverse/protocol/xai_image_edit.py::extract_streaming_response  ~  AuuCoder/gptGrok2api:app/dataplane/reverse/protocol/xai_image_edit.py::extract_model_response_file_attachments
- ast-exact: AuuCoder/gptGrok2api:app/dataplane/reverse/protocol/xai_image_edit.py::extract_model_response_urls  ~  AuuCoder/gptGrok2api:app/dataplane/reverse/protocol/xai_image_edit.py::extract_model_response_file_attachments
- ngram10: ai4s-research/ai4s-skills:skills/integrity-auditor/forensics_tools/magnitude_consistency.py::auto_detect_entity_col  ~  ai4s-research/ai4s-skills:skills/integrity-auditor/forensics_tools/xlsx_aggregate_consistency.py::auto_detect_entity_col
- ngram10: mstan/MetroidPrimeHuntersRecomp:tools/pgo_ab_compare.py::phase_emu_ms  ~  mstan/MetroidPrimeHuntersRecomp:tools/pgo_ab_compare.py::state_fingerprint
- ast-exact: nando0x/ProspectOS:backend/processar.py::_e_falha_de_dns  ~  M1n-n9/academic-ppt-master:scripts/image_backends/backend_common.py::is_rate_limit_error
- ngram10: Birfy/agentdescent:agentdescent/selection.py::pareto_win_frequency  ~  Birfy/agentdescent:agentdescent/selection.py::sigmoid_novelty_weights
- ngram10: asimons81/hermes-gpt:operator_contract.py::_prompt_meta  ~  asimons81/hermes-gpt:operator_mission.py::_prompt_meta
- ngram10: wp-a/nature-academic-search:src/nature_academic_search/conversion/converters.py::crossref_to_bib  ~  wp-a/nature-academic-search:src/nature_academic_search/conversion/converters.py::arxiv_to_bib
- ast-exact: msrbuilds/voice-studio:backend/core/engines/vibevoice_engine.py::_normalize_speaker_tags  ~  msrbuilds/voice-studio:backend/services/synthesize.py::_normalize_speaker_tags
- ast-exact: M1n-n9/academic-ppt-master:scripts/image_backends/backend_ideogram.py::_resolve_url  ~  M1n-n9/academic-ppt-master:scripts/image_backends/backend_qwen.py::_resolve_url
- ngram10: pagecat/vowifi_gateway:control/app/config.py::normalize_idr_mode  ~  pagecat/vowifi_gateway:control/app/config.py::normalize_cp_mode

## Limitations

- Layer 1 and 3 are exact/structural; a semantic rewrite passes both.
- The embedding model has blind spots; the review band exists for that reason.
- The cutoff for Qwen3.5 is year-only; the floor is a margin, not a proof.
- Renames are not followed in the introducing-commit check (see harvest.py).
