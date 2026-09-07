# Source report 01 — Qwen3.5 training-data cutoff

Research agent report (Claude Sonnet), 2026-09-07. Unedited except for this header.
Claims marked unverified by the agent remain unverified.

---

## Bottom line up front
No Qwen3.5 model card or the Qwen3.5-Omni technical report states an explicit knowledge-cutoff date or pretraining-corpus date range. The only concrete signal is **"2026"** (year-only) surfaced from Alibaba's official DashScope API system prompt, per a third-party tracker. No month-level precision is available anywhere.

## 1. Qwen3.5-9B / Qwen3.5-9B-Base model cards (Hugging Face)
- huggingface.co/Qwen/Qwen3.5-9B and .../Qwen3.5-9B-Base: no stated knowledge cutoff, no pretraining date range.
- Release: February 2026 (per the model-card citation block: "Qwen3.5: Towards Native Multimodal Agents", Qwen Team, Feb 2026). Per Alibaba Cloud community blog coverage, Qwen3.5-397B-A17B launched Feb 16, 2026; Qwen3.5-122B-A10B/35B-A3B/27B on Feb 24, 2026; Qwen3.5-9B/4B/2B/0.8B on Mar 2, 2026.
- Both cards cite a blog post (qwen.ai/blog?id=qwen3.5) rather than a technical report. No arXiv ID for a text-only Qwen3.5 report was found.

## 2. "Qwen3.5 technical report"
- The only Qwen3.5-branded arXiv paper found is **Qwen3.5-Omni Technical Report** (arXiv 2604.15804, submitted 2026-04-17, revised 2026-04-21) — the multimodal/omni variant. The fetched excerpt contains **no** pretraining collection dates, no GitHub-crawl/Stack-version details, and no post-training data dates. Only the abstract page was fetched.
- Original Qwen3 technical report is arXiv 2505.09388 — different generation.

## 3. Official Qwen blog
- qwen.ai/blog?id=qwen3.5; the Alibaba Cloud Community mirror (alibabacloud.com/blog/602894) returned empty content on fetch (JS-rendered). Third-party coverage confirms the Feb 16, 2026 initial release and the Feb 24 / Mar 2 rollout of smaller dense models; none quote a cutoff date.

## 4. Third-party cutoff determinations
- **metehan.ai "LLM Knowledge Cutoff Dates" (updated June 2026)** — classifies Qwen3.5 at Tier 1 confidence, sourced from Alibaba's **DashScope API system prompt**, stating Qwen 3.5 and above "claim 2026 knowledge." Year-only.
- The same article warns (citing github.com/QwenLM/Qwen3/issues/1442) that Qwen models **hallucinate their own cutoff when asked directly**.
- GitHub QwenLM/Qwen3 Discussion #1093 does NOT cover Qwen3.5; it covers Qwen2.5 (~Oct 2023 claimed, disputed) and Qwen2.5-Coder (maintainer: "June is appropriate for us to use", i.e. June 2024).
- aiknowledgecutoff.com/qwen exists but was not fetched in detail.

## 5. Baseline comparisons
- **Qwen2.5-Coder-7B-Instruct**: community consensus cutoff **June 30, 2024** (Qwen maintainer comment in Discussion #1093; OpenRouter metadata agrees). One conflicting claim of March 2024 (llm-stats.com).
- **Qwen3.5-4B**: same as the 9B — no cutoff on the card.

## (a) Best-supported cutoff for Qwen3.5-9B
**Year 2026, no month specified — Medium confidence.** Source: metehan.ai's tracker citing Alibaba's DashScope system prompt. Not corroborated by any primary source (HF card, blog, arXiv).

## (b) Recommended safe post-cutoff date
- **Use data dated on or after June 1, 2026** as the safe floor: ~3 months past the latest (Mar 2) release, clearing the "2026" claim without month precision.
- For a more conservative margin, **September 1, 2026**.

## (c) Could not verify
- No month-level cutoff for any Qwen3.5 variant.
- Whether the Qwen3.5-Omni report is the same report underlying the text-only 9B/4B.
- Full body of arXiv 2604.15804.
- The Alibaba Cloud blog content (empty fetch).
- aiknowledgecutoff.com/qwen page content.

## Sources
- [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) · [Qwen3.5-9B-Base](https://huggingface.co/Qwen/Qwen3.5-9B-Base) · [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) · [Qwen3.5-4B-Base](https://huggingface.co/Qwen/Qwen3.5-4B-Base)
- [Qwen3.5-Omni Technical Report (arXiv 2604.15804)](https://arxiv.org/abs/2604.15804) · [Qwen3 Technical Report (arXiv 2505.09388)](https://arxiv.org/abs/2505.09388) · [Qwen2.5-Coder Technical Report (arXiv 2409.12186)](https://arxiv.org/pdf/2409.12186)
- [Alibaba Cloud Community blog 602894](https://www.alibabacloud.com/blog/602894)
- [LLM Knowledge Cutoff Dates (metehan.ai)](https://metehan.ai/articles/llm-knowledge-cutoff-dates/) · [QwenLM/Qwen3 Discussion #1093](https://github.com/QwenLM/Qwen3/discussions/1093) · [QwenLM/Qwen3 issue #1442](https://github.com/QwenLM/Qwen3/issues/1442)
- [Qwen2.5 Coder 7B Instruct - OpenRouter](https://openrouter.ai/qwen/qwen2.5-coder-7b-instruct) · [aiknowledgecutoff.com/qwen](https://aiknowledgecutoff.com/qwen)
