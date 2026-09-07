# Decontamination Report — held-out pool vs MBPP

Generated 2026-09-07. Floor date 2026-06-01. Candidates 538, kept 503, families 86.

## Layers and thresholds

- n-gram: word-level, n=10; any shared n-gram with an MBPP function flags.
- AST: identifiers→slots, docstrings dropped, constants→type; exact hash flags; k=5 node shingles Jaccard ≥ 0.85 flags.
- embedding: jinaai/jina-embeddings-v2-base-code; cosine ≥ 0.9 flags, 0.8–0.9 listed for human review.
- family: same source repo, plus union-find over every flagged candidate pair; one function kept per family cluster of near-duplicates.

## Removed against MBPP

0 candidates removed.


## Review band (kept, human check requested)

none

## Near-duplicate merges within the pool

- ast-exact: guillaumemeyer/watermarks-remover:service/scripts/bench_synthid_text.py::_numbers_preserved  ~  guillaumemeyer/watermarks-remover:service/scripts/bench_synthid_text.py::_urls_preserved
- ngram10: andrewyng/openworker:coworker/attachments.py::content_to_text  ~  andrewyng/openworker:coworker/compaction.py::_text_of
- ngram10: andrewyng/openworker:coworker/compaction.py::_text_of  ~  sums001/Deepseek-API:server/openai_format.py::_text_of
- ngram10: andrewyng/openworker:coworker/pdf_support.py::_encode_png  ~  ZeKaiNie/universal-examprep-skill:coach/figures.py::png_bytes
- ngram10: huggingface/tau:src/tau_coding/models_dev.py::models_dev_catalog_overlay  ~  vinhhien112/img2obj:scripts/sculpt_contract.py::component_type
- ngram10: BigDawnGhost/wenyi:trans_novel/assemble/docx_writer.py::_style_slices  ~  zenstory-ai/oh-story-dsh:packages/knowledge/drama/skills/short-drama-write/scripts/duration_estimate.py::_block_text
- ngram10: shepherd-agents/shepherd:commons-vcs/src/commons_vcs/_types.py::_normalize_json_value  ~  shepherd-agents/shepherd:commons-vcs/src/commons_vcs/canonical.py::_validate_json_primitive
- ngram10: shepherd-agents/shepherd:commons-vcs/src/commons_vcs/_types.py::_normalize_json_value  ~  shepherd-agents/shepherd:shepherd/packages/core/src/shepherd_core/effects/commons_vcs.py::normalize_commons_value
- ast-exact: shepherd-agents/shepherd:commons-vcs/src/commons_vcs/backends/git.py::_digest_to_segment  ~  shepherd-agents/shepherd:commons-vcs/src/commons_vcs/backends/git.py::_segment_to_digest
- ngram10: shepherd-agents/shepherd:commons-vcs/src/commons_vcs/canonical.py::_validate_json_primitive  ~  shepherd-agents/shepherd:shepherd/packages/core/src/shepherd_core/effects/commons_vcs.py::normalize_commons_value
- ngram10: Hao0321/video-autopilot-kit:src/context_router.py::estimate_tokens  ~  no8d/ComfyUI-NO8D-controls:prompt_plus.py::_approx_prompt_tokens
- ngram10: MatrAIx-ai/MatrAIx-Persona-8B:application/playground/backend/service/job_aggregation.py::_distribution_directive_dedupe_marker  ~  MatrAIx-ai/MatrAIx-Persona-8B:application/playground/backend/service/job_aggregation.py::_distribution_directive_dimensions
- ngram10: zenstory-ai/drama-skills:skills/short-drama/scripts/project_tool.py::project_video_model_profile  ~  zenstory-ai/drama-skills:skills/short-drama/scripts/project_tool.py::_artifact_state_from
- ngram10: zenstory-ai/drama-skills:skills/short-drama/scripts/project_tool.py::project_video_model_profile  ~  zenstory-ai/drama-skills:skills/short-drama/scripts/project_tool.py::_json_kind
- ngram10: zenstory-ai/drama-skills:skills/short-drama/scripts/project_tool.py::_artifact_state_from  ~  zenstory-ai/drama-skills:skills/short-drama/scripts/project_tool.py::_json_kind
- ngram10: littledivy/mimic:mimic/extract.py::base_url  ~  littledivy/mimic:mimic/sources/mitm.py::hosts
- ngram10: AMAP-ML/LongHorizon-Harness:eval/OSWorldv2-harness/OSWorld-V2/desktop_env/evaluators/metrics/chrome.py::check_enabled_experiments  ~  AMAP-ML/LongHorizon-Harness:eval/OSWorldv2-harness/OSWorld-V2/desktop_env/evaluators/metrics/chrome.py::check_font_size
- ast-exact: pyang5166/gbro-collage-broll:scripts/generate_video.py::parse_and_validate_duration  ~  yokel1121/muyang-flat-animation:muyang-flat-animation/scripts/generate_video.py::parse_and_validate_duration
- ngram10: sums001/Windows-Copilot-API:server/prompt.py::content_text  ~  sums001/Deepseek-API:server/openai_format.py::_text_of
- ngram10: Jwuthri/Tracely-ai:backend/tracely/api/routers/monitors.py::_sample_at  ~  kelvinfkr/company_skill:skills/company-talent-economics/scripts/i18n.py::t
- ngram10: Jwuthri/Tracely-ai:backend/tracely/api/routers/sessions.py::_shape_declared_agent  ~  Jwuthri/Tracely-ai:backend/tracely/api/routers/sessions.py::_declared_tools
- cosine 0.98: GangTailorUpgrade/undress-service:coomertool/utils.py::format_size  ~  luli395/android_everything:ui/styles.py::format_size
- ngram10: 2akouwu/reverify:reverify/agent.py::parse_claims  ~  dongguatanglinux/grok-build-auth:xconsole_client/sso.py::parse_sso_jwt_url
- ngram10: 2akouwu/reverify:reverify/agent.py::parse_claims  ~  dongguatanglinux/grok-build-auth:xconsole_client/sso.py::parse_jwt_payload
- ngram10: 2akouwu/reverify:reverify/agent.py::parse_claims  ~  dongguatanglinux/grok-build-auth:xconsole_client/sso.py::parse_sso_from_set_cookies
- ngram10: 2akouwu/reverify:reverify/agent.py::parse_claims  ~  dongguatanglinux/grok-build-auth:xconsole_client/sso.py::_extract_jwt_from_url
- ngram10: robocurve/inspect-robots:plugins/inspect-robots-agent/src/inspect_robots_agent/_anthropic.py::_history_tool_use_ids  ~  robocurve/inspect-robots:plugins/inspect-robots-agent/src/inspect_robots_agent/_responses.py::_history_call_ids
- ngram10: robocurve/inspect-robots:plugins/inspect-robots-agent/src/inspect_robots_agent/_anthropic.py::_translate_tools  ~  robocurve/inspect-robots:plugins/inspect-robots-agent/src/inspect_robots_agent/_responses.py::_translate_tools
- ngram10: livetennisapi/polymarket-tennis:src/polymarket_tennis/join.py::derive_break_point  ~  livetennisapi/polymarket-tennis:src/polymarket_tennis/join.py::score_line
- ngram10: zenstory-ai/oh-story-dsh:packages/knowledge/video-recap/skills/video-assemble/scripts/audio_automation.py::coalesce_duck_windows  ~  zenstory-ai/oh-story-dsh:packages/knowledge/video-recap/skills/video-assemble/scripts/audio_automation.py::coalesce_release_duck_windows
- ngram10: lbx154/Argus:argus_skill/adapters/stream_progress.py::_action_summary  ~  lbx154/Argus:argus_skill/adapters/stream_progress.py::_shell_tool_bucket
- ngram10: adongwanai/learn-workbuddy:s13_output_externalization/code.py::estimate_messages_tokens  ~  adongwanai/learn-workbuddy:s14_context_compact/code.py::estimate_tokens
- ngram10: adongwanai/learn-workbuddy:s19_visualizer/code.py::svg_node  ~  adongwanai/learn-workbuddy:s19_visualizer/code.py::svg_title
- ngram10: dongguatanglinux/grok-build-auth:xconsole_client/sso.py::parse_sso_jwt_url  ~  dongguatanglinux/grok-build-auth:xconsole_client/sso.py::parse_jwt_payload
- ngram10: dongguatanglinux/grok-build-auth:xconsole_client/sso.py::parse_sso_jwt_url  ~  dongguatanglinux/grok-build-auth:xconsole_client/sso.py::parse_sso_from_set_cookies
- ast-exact: dongguatanglinux/grok-build-auth:xconsole_client/sso.py::parse_sso_jwt_url  ~  dongguatanglinux/grok-build-auth:xconsole_client/sso.py::_extract_jwt_from_url
- ngram10: dongguatanglinux/grok-build-auth:xconsole_client/sso.py::parse_jwt_payload  ~  dongguatanglinux/grok-build-auth:xconsole_client/sso.py::parse_sso_from_set_cookies
- ngram10: dongguatanglinux/grok-build-auth:xconsole_client/sso.py::parse_jwt_payload  ~  dongguatanglinux/grok-build-auth:xconsole_client/sso.py::_extract_jwt_from_url
- ngram10: dongguatanglinux/grok-build-auth:xconsole_client/sso.py::parse_sso_from_set_cookies  ~  dongguatanglinux/grok-build-auth:xconsole_client/sso.py::_extract_jwt_from_url
- ngram10: jinyangwu/SEED:agent_system/environments/env_package/alfworld/projection.py::alfworld_projection  ~  jinyangwu/SEED:agent_system/environments/env_package/appworld/projection.py::appworld_projection
- ast-exact: jinyangwu/SEED:agent_system/environments/env_package/alfworld/projection.py::alfworld_projection  ~  jinyangwu/SEED:agent_system/environments/env_package/sciworld/projection.py::sciworld_projection
- ngram10: jinyangwu/SEED:agent_system/environments/env_package/alfworld/projection.py::alfworld_projection  ~  jinyangwu/SEED:agent_system/environments/env_package/webshop/projection.py::webshop_projection
- ngram10: jinyangwu/SEED:agent_system/environments/env_package/appworld/projection.py::appworld_projection  ~  jinyangwu/SEED:agent_system/environments/env_package/sciworld/projection.py::sciworld_projection
- ngram10: jinyangwu/SEED:agent_system/environments/env_package/appworld/projection.py::appworld_projection  ~  jinyangwu/SEED:agent_system/environments/env_package/webshop/projection.py::webshop_projection
- ngram10: jinyangwu/SEED:agent_system/environments/env_package/sciworld/projection.py::sciworld_projection  ~  jinyangwu/SEED:agent_system/environments/env_package/webshop/projection.py::webshop_projection
- ngram10: icesixgod/codex-trajectory:plugins/codex-trajectory/scripts/codex_trajectory/privacy.py::content_text  ~  icesixgod/codex-trajectory:plugins/codex-trajectory/scripts/codex_trajectory/privacy.py::reasoning_summary

## Limitations

- Layer 1 and 3 are exact/structural; a semantic rewrite passes both.
- The embedding model has blind spots; the review band exists for that reason.
- The cutoff for Qwen3.5 is year-only; the floor is a margin, not a proof.
- Renames are not followed in the introducing-commit check (see harvest.py).
