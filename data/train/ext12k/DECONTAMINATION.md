# Decontamination of the external training set

Candidates: 21085. Reference: both held-out splits of data/heldout/pool.jsonl (486 functions). Layers: 1 (n-gram) and 3 (AST hash, AST Jaccard); layer 2 (code embeddings) skipped: the source is 2025 synthetic data, the held-out pool is post-2026-06 GitHub, so only generic near-duplicates are at risk and layers 1/3 catch those.

Removed as matching a held-out function: 14

- kodcode:Algorithm/Algorithm_23634_I::range_product: ngram10 x1 BigDawnGhost/wenyi:trans_novel/assemble/docx_writer.py::_style_slices
- kodcode:Algorithm/Algorithm_4127_I::is_valid_path: ngram10 x1 dongguatanglinux/grok-build-auth:xconsole_client/fingerprint.py::_next_cookie_boundary
- kodcode:Algorithm/Algorithm_47239_C::count_set_bits_in_range: ngram10 x1 BigDawnGhost/wenyi:trans_novel/assemble/docx_writer.py::_style_slices
- kodcode:Docs/Docs: Python310_24837_C::analyze_log_file: ngram10 x1 Rimagination/bili-note:scripts/score_bili_note.py::count_evidence_refs
- kodcode:Docs/Docs: Python310_26405_I::transform_python2_to_python3: ast-jaccard 0.89 AMAP-ML/LongHorizon-Harness:eval/OSWorldv2-harness/OSWorld-V2/desktop_env/evaluators/metrics/general.py::remove_comments_and_strings
- kodcode:Docs/Docs: Python310_40397_C::validate_and_extract: ngram10 x1 Rimagination/bili-note:scripts/score_bili_note.py::count_evidence_refs
- kodcode:Docs/Docs: Python310_40397_I::validate_and_extract: ngram10 x1 Rimagination/bili-note:scripts/score_bili_note.py::count_evidence_refs
- kodcode:Filter/Filter_10881_I::greet: ast-exact HKUDS/OpenOPC:opc/core/windows_ssl.py::format_windows_sslkeylog_warning; ast-jaccard 1.00 HKUDS/OpenOPC:opc/core/windows_ssl.py::format_windows_sslkeylog_warning
- kodcode:Filter/Filter_26161_I::transformed_multiples: ngram10 x1 BigDawnGhost/wenyi:trans_novel/assemble/docx_writer.py::_style_slices
- kodcode:Filter/Filter_40089_I::build_update_date_of_birth_mutation: ast-exact HKUDS/OpenOPC:opc/core/windows_ssl.py::format_windows_sslkeylog_warning; ast-jaccard 1.00 HKUDS/OpenOPC:opc/core/windows_ssl.py::format_windows_sslkeylog_warning
- kodcode:Filter/Filter_41797_I::sum_of_cubes: ngram10 x1 BigDawnGhost/wenyi:trans_novel/assemble/docx_writer.py::_style_slices
- kodcode:Filter/Filter_64878_I::generate_greeting: ast-exact HKUDS/OpenOPC:opc/core/windows_ssl.py::format_windows_sslkeylog_warning; ast-jaccard 1.00 HKUDS/OpenOPC:opc/core/windows_ssl.py::format_windows_sslkeylog_warning
- kodcode:Filter/Filter_78822_I::create_greeting_template: ast-exact HKUDS/OpenOPC:opc/core/windows_ssl.py::format_windows_sslkeylog_warning; ast-jaccard 1.00 HKUDS/OpenOPC:opc/core/windows_ssl.py::format_windows_sslkeylog_warning
- kodcode:Filter/Filter_87762_I::extract_datetime: ngram10 x1 Rimagination/bili-note:scripts/score_bili_note.py::count_evidence_refs
