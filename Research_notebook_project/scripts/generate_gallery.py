# scripts/generate_gallery.py
"""
Generates the static SYNTHETIC demo gallery. The 10 reports below are hand-written
examples, NOT outputs of the CiteGuard pipeline. Token/cost/latency are not measured.
Produces 10 example reports complete with:
- Executive summaries
- Claim-level verification badges (kept, dropped, rewritten)
- Verbatim quote citations and source links
- Cross-source contradiction notes
- Audit entries (hand-written, not produced by the verifier)
- An interactive single-page HTML/JS application (gallery/index.html)
"""
import json
from pathlib import Path

PLACEHOLDER = "synthetic placeholder - not measured"
GALLERY_DIR = Path(__file__).resolve().parents[1] / "gallery"
REPORTS_DIR = GALLERY_DIR / "reports"

GALLERY_REPORTS = [
    {
        "id": "report-01",
        "title": "Perovskite-Silicon Tandem Solar Cells: Efficiency & Degradation",
        "question": "What are the efficiency and degradation benchmarks of perovskite-silicon tandem solar cells as of 2024?",
        "summary": "Perovskite-silicon tandem solar cells surpassed the theoretical 29.4% Shockley-Queisser limit for single-junction silicon, achieving certified efficiencies up to 33.9% in late 2023. Encapsulation techniques have improved operational stability, with cells retaining 95% initial performance after 1,000 hours of accelerated ISOS thermal cycling.",
        "mode": "multi",
        "claims": [
            {
                "id": 1,
                "text": "Certified power conversion efficiency of perovskite-silicon tandem cells reached 33.9% in late 2023.",
                "quote": "Perovskite-silicon tandem solar cells reached a certified power conversion efficiency of 33.9% in late 2023.",
                "domain": "nature.com",
                "url": "https://nature.com/articles/s41560-023-01402-y",
                "status": "kept",
                "confidence": "high",
                "entailment": "supports",
                "reason": "Direct verbatim entailment confirmed against stored snapshot."
            },
            {
                "id": 2,
                "text": "Encapsulated tandem cells maintained 95% of initial efficiency after 1,000 hours under ISOS-L-2 testing at 65°C.",
                "quote": "Encapsulated devices retained 95% of initial efficiency after 1,000 hours of ISOS-L-2 continuous light soaking at 65°C.",
                "domain": "sciencedirect.com",
                "url": "https://sciencedirect.com/science/article/pii/S254243512300456X",
                "status": "kept",
                "confidence": "high",
                "entailment": "supports",
                "reason": "Quantitatively supported by independent laboratory measurements."
            },
            {
                "id": 3,
                "text": "Perovskite-silicon tandem solar cells achieved a certified power conversion efficiency of 43.9% in 2023.",
                "quote": "Perovskite-silicon tandem solar cells reached a certified power conversion efficiency of 33.9% in late 2023.",
                "domain": "nature.com",
                "url": "https://nature.com/articles/s41560-023-01402-y",
                "status": "dropped",
                "confidence": "low",
                "entailment": "unsupported",
                "reason": "Number hallucination: draft altered 33.9% to 43.9%."
            }
        ],
        "conflicts": []
    },
    {
        "id": "report-02",
        "title": "FlashAttention-2: Memory IO Optimization & Parallelization",
        "question": "How does FlashAttention-2 reduce memory IO complexity and improve training throughput in Transformer architectures?",
        "summary": "FlashAttention-2 achieves an approximate 2x speedup over the original FlashAttention kernel (and up to 9x over standard PyTorch attention) by minimizing non-matrix-multiplication operations, partitioning compute across the sequence length dimension, and optimizing GPU warp layout.",
        "mode": "multi",
        "claims": [
            {
                "id": 1,
                "text": "FlashAttention-2 achieves roughly a 2x speedup over FlashAttention by cutting non-matmul FLOPs and parallelizing over sequence length.",
                "quote": "FlashAttention-2 yields a 2x speedup over FlashAttention by reducing non-matmul FLOPs and parallelizing over the sequence length dimension.",
                "domain": "arxiv.org",
                "url": "https://arxiv.org/abs/2307.08691",
                "status": "kept",
                "confidence": "high",
                "entailment": "supports",
                "reason": "Exact mathematical and architectural claim confirmed from paper snapshot."
            },
            {
                "id": 2,
                "text": "FlashAttention-2 partitions work across sequence length rather than solely batch and head dimensions.",
                "quote": "parallelizing over the sequence length dimension even within a single thread block.",
                "domain": "arxiv.org",
                "url": "https://arxiv.org/abs/2307.08691",
                "status": "kept",
                "confidence": "high",
                "entailment": "supports",
                "reason": "Accurately represents the work scheduling scheme."
            },
            {
                "id": 3,
                "text": "FlashAttention-2 achieves a 10x speedup by completely eliminating SRAM memory traffic on Nvidia A100 GPUs.",
                "quote": "FlashAttention-2 yields a 2x speedup over FlashAttention",
                "domain": "arxiv.org",
                "url": "https://arxiv.org/abs/2307.08691",
                "status": "dropped",
                "confidence": "low",
                "entailment": "unsupported",
                "reason": "Extrapolated exaggeration: 10x speedup unsupported by empirical benchmarks."
            }
        ],
        "conflicts": []
    },
    {
        "id": "report-03",
        "title": "CRISPR Base Editing: Clinical Efficacy & Safety in Phase 1/2",
        "question": "What are the verified clinical efficacy and off-target mutation rates of CRISPR base editing therapies in Phase 1/2 trials?",
        "summary": "In vivo adenine base editing (specifically VERVE-101 targeting PCSK9) demonstrated dose-dependent reductions in blood PCSK9 (up to 84%) and LDL-C (up to 55%) in human Phase 1b trials. Base editors utilize catalytically dead Cas9 fused with deaminase domains, avoiding double-strand DNA breaks.",
        "mode": "multi",
        "claims": [
            {
                "id": 1,
                "text": "In Phase 1b clinical evaluation of VERVE-101, highest dose participants showed up to 84% reduction in PCSK9 and 55% in LDL cholesterol.",
                "quote": "In the Phase 1b trial of adenine base editor VERVE-101, blood PCSK9 levels were reduced by up to 84% and LDL-C by 55% at highest dose cohorts.",
                "domain": "nejm.org",
                "url": "https://nejm.org/doi/full/10.1056/NEJMoa2300700",
                "status": "kept",
                "confidence": "high",
                "entailment": "supports",
                "reason": "Clinical trial endpoints match peer-reviewed NEJM findings."
            },
            {
                "id": 2,
                "text": "VERVE-101 completely cured 100% of familial hypercholesterolemia patients with zero transient liver enzyme elevations.",
                "quote": "transient ALT elevations were observed in two patients in cohort 3.",
                "domain": "nejm.org",
                "url": "https://nejm.org/doi/full/10.1056/NEJMoa2300700",
                "status": "dropped",
                "confidence": "low",
                "entailment": "unsupported",
                "reason": "Absolute cure and absence of adverse events refuted by paper data."
            }
        ],
        "conflicts": []
    },
    {
        "id": "report-04",
        "title": "Google Sycamore vs IBM Eagle: Superconducting Quantum Processors",
        "question": "What are the confirmed quantum volume and two-qubit gate fidelity metrics of Google Sycamore vs IBM Eagle processors?",
        "summary": "Google's 53-qubit Sycamore processor demonstrated random circuit sampling in 200 seconds in 2019. IBM Eagle expanded physical count to 127 qubits using a heavy-hexagonal coupling topology with average two-qubit gate error rates near 1.0e-2.",
        "mode": "multi",
        "claims": [
            {
                "id": 1,
                "text": "Google Sycamore sampled random quantum circuit instances in 200 seconds using 53 operational transmon qubits.",
                "quote": "The 53-qubit Sycamore processor sampled random quantum circuit instances in 200 seconds, estimating classical supercomputers would take 10,000 years.",
                "domain": "nature.com",
                "url": "https://nature.com/articles/s41586-019-1666-5",
                "status": "kept",
                "confidence": "high",
                "entailment": "supports",
                "reason": "Accurately represents Nature quantum supremacy experiment."
            },
            {
                "id": 2,
                "text": "IBM Eagle operates 127 qubits arranged in a heavy-hex layout with average two-qubit error rates around 1.0e-2.",
                "quote": "IBM Eagle features 127 superconducting qubits with heavy-hexagonal layout and average two-qubit gate error rates around 1.0e-2.",
                "domain": "ibm.com",
                "url": "https://research.ibm.com/blog/127-qubit-quantum-processor-eagle",
                "status": "kept",
                "confidence": "high",
                "entailment": "supports",
                "reason": "Hardware specifications verified from official research release."
            },
            {
                "id": 3,
                "text": "Google Sycamore ran Shor's algorithm to factor a 2048-bit RSA key in 200 seconds.",
                "quote": "sampled random quantum circuit instances in 200 seconds",
                "domain": "nature.com",
                "url": "https://nature.com/articles/s41586-019-1666-5",
                "status": "dropped",
                "confidence": "low",
                "entailment": "unsupported",
                "reason": "Catastrophic entity hallucination: benchmark was random circuit sampling, not Shor factoring."
            }
        ],
        "conflicts": []
    },
    {
        "id": "report-05",
        "title": "PostgreSQL pgvector: HNSW vs IVFFlat Indexing Performance",
        "question": "Compare HNSW and IVFFlat index recall, build time, and query QPS in PostgreSQL pgvector against dedicated vector engines.",
        "summary": "pgvector's HNSW index implementation provides up to 20x higher queries per second at 99% recall relative to IVFFlat, though it requires up to 10x longer build durations and greater memory allocation. Both indexes reside within standard PostgreSQL ACID tables.",
        "mode": "multi",
        "claims": [
            {
                "id": 1,
                "text": "pgvector HNSW indexes deliver significantly higher query QPS at 99% recall compared to IVFFlat, at the expense of substantially longer build times.",
                "quote": "On 1M 1536-dimensional embeddings, pgvector HNSW achieved 20x higher QPS at 99% recall than IVFFlat, though index build time was 10x longer.",
                "domain": "jkatz05.com",
                "url": "https://jkatz05.com/post/postgres/pgvector-hnsw-performance/",
                "status": "kept",
                "confidence": "high",
                "entailment": "supports",
                "reason": "Empirical pgvector benchmark accurately represented."
            },
            {
                "id": 2,
                "text": "pgvector 0.5.0 introduced iterative index builds and HNSW vector search directly within PostgreSQL.",
                "quote": "introduced HNSW indexing in pgvector 0.5.0",
                "domain": "jkatz05.com",
                "url": "https://jkatz05.com/post/postgres/pgvector-hnsw-performance/",
                "status": "kept",
                "confidence": "medium",
                "entailment": "supports",
                "reason": "Release history verified."
            }
        ],
        "conflicts": []
    },
    {
        "id": "report-06",
        "title": "Llama 3 70B vs Mistral Large: Enterprise RAG Architecture",
        "question": "Compare Llama 3 70B and Mistral Large regarding inference latency, context window limits, and commercial licensing for enterprise RAG.",
        "summary": "Llama 3 70B offers open model weights under Meta's Community License (requiring commercial approval above 700M monthly active users). Mistral Large provides native 32K context with high multilingual performance and strict JSON schema guarantees.",
        "mode": "multi",
        "claims": [
            {
                "id": 1,
                "text": "Initial Meta Llama 3 70B models supported an 8k context window under the Llama 3 Community License.",
                "quote": "Llama 3 70B was released with an 8k context window under the Meta Llama 3 Community License, later expanded to 128k with Llama 3.1.",
                "domain": "ai.meta.com",
                "url": "https://ai.meta.com/blog/meta-llama-3/",
                "status": "kept",
                "confidence": "high",
                "entailment": "supports",
                "reason": "Licensing parameters and context specifications confirmed."
            },
            {
                "id": 2,
                "text": "Mistral Large natively provides a 32k context window and structured JSON formatting across English and European languages.",
                "quote": "Mistral Large natively supports a 32k context window and precise JSON output formatting with native support for English, French, German, Spanish, and Italian.",
                "domain": "mistral.ai",
                "url": "https://mistral.ai/news/mistral-large/",
                "status": "kept",
                "confidence": "high",
                "entailment": "supports",
                "reason": "Context window and multilingual support confirmed."
            },
            {
                "id": 3,
                "text": "Meta Llama 3 is licensed under pure unencumbered Apache 2.0 with no commercial MAU threshold conditions.",
                "quote": "under the Meta Llama 3 Community License",
                "domain": "ai.meta.com",
                "url": "https://ai.meta.com/blog/meta-llama-3/",
                "status": "dropped",
                "confidence": "low",
                "entailment": "unsupported",
                "reason": "Licensing error: Meta uses a bespoke Community License, not Apache 2.0."
            }
        ],
        "conflicts": []
    },
    {
        "id": "report-07",
        "title": "Rust vs Go for High-Concurrency Network Proxies",
        "question": "Compare Rust vs Go for high-throughput network proxies regarding memory footprint, tail latency (p99), and GC pause times.",
        "summary": "Rust proxies (such as Cloudflare Pingora) demonstrate predictable sub-millisecond tail latency and up to 67% lower memory footprint compared to garbage-collected alternatives. Go provides rapid development and built-in goroutine scheduling, though GC cycles can impact extreme p99 tail requirements.",
        "mode": "multi",
        "claims": [
            {
                "id": 1,
                "text": "Cloudflare reported that switching their proxy infrastructure to Rust (Pingora) yielded 70% CPU reduction and 67% memory reduction.",
                "quote": "Cloudflare's Rust-based Pingora proxy consumed 70% less CPU and 67% less memory compared to their previous NGINX/Lua service under identical traffic.",
                "domain": "blog.cloudflare.com",
                "url": "https://blog.cloudflare.com/how-we-built-pingora-in-rust/",
                "status": "kept",
                "confidence": "high",
                "entailment": "supports",
                "reason": "Production engineering benchmarks verified from Cloudflare technical report."
            },
            {
                "id": 2,
                "text": "Go proxies completely avoid stop-the-world garbage collection pauses under 500,000 concurrent TLS connections.",
                "quote": "tracing garbage collector periodically pauses goroutines for mark-sweep termination.",
                "domain": "golang.org",
                "url": "https://golang.org/doc/gc",
                "status": "dropped",
                "confidence": "low",
                "entailment": "unsupported",
                "reason": "Unsubstantiated claim: Go runtime GC is non-zero."
            }
        ],
        "conflicts": []
    },
    {
        "id": "report-08",
        "title": "Apache Iceberg vs Delta Lake: Data Lakehouse Table Architecture",
        "question": "Compare Apache Iceberg and Delta Lake table formats regarding ACID guarantees, hidden partitioning, and engine-agnostic reader support.",
        "summary": "Both Iceberg and Delta Lake provide ACID transactions over Parquet files using optimistic concurrency control. Iceberg distinguishes itself through engine-agnostic hidden partitioning and fine-grained snapshot metadata, while Delta Lake features deep integration with the Databricks and Spark ecosystems.",
        "mode": "multi",
        "claims": [
            {
                "id": 1,
                "text": "Apache Iceberg uses hidden partitioning to automatically derive partition values from timestamp columns without query rewrites.",
                "quote": "Apache Iceberg's hidden partitioning prevents silent data corruption by generating partition values automatically without user-specified column derivations.",
                "domain": "tabular.io",
                "url": "https://tabular.io/blog/iceberg-hidden-partitioning/",
                "status": "kept",
                "confidence": "high",
                "entailment": "supports",
                "reason": "Specification feature confirmed."
            },
            {
                "id": 2,
                "text": "Both Iceberg and Delta Lake rely on Apache Parquet as the primary underlying columnar file storage layer.",
                "quote": "Both table formats store columnar row groups in standard immutable Apache Parquet format.",
                "domain": "tabular.io",
                "url": "https://tabular.io/blog/iceberg-hidden-partitioning/",
                "status": "kept",
                "confidence": "high",
                "entailment": "supports",
                "reason": "Underlying storage layer verified across both ecosystems."
            }
        ],
        "conflicts": []
    },
    {
        "id": "report-09",
        "title": "Artificial Sweeteners: Cardiovascular & Metabolic Evidence (Contested)",
        "question": "Does aspartame and sucralose consumption increase cardiovascular disease risk or metabolic dysfunction according to RCTs vs observational cohorts?",
        "summary": "Evidence on artificial sweeteners exhibits significant divergence between study methodologies. Observational cohorts (such as NutriNet-Santé) detect modest correlations with increased cardiovascular event risk, whereas short-term randomized controlled trials report no acute glycemic or insulin elevation.",
        "mode": "multi",
        "claims": [
            {
                "id": 1,
                "text": "The NutriNet-Santé observational cohort observed a statistically significant 9% relative increase in cardiovascular events among high artificial sweetener consumers.",
                "quote": "In the NutriNet-Santé cohort of 102,865 adults, artificial sweetener intake was associated with a 9% increased risk of cardiovascular events (HR 1.09, 95% CI 1.01 to 1.18).",
                "domain": "bmj.com",
                "url": "https://bmj.com/content/378/bmj-2022-071204",
                "status": "kept",
                "confidence": "medium",
                "entailment": "supports",
                "reason": "Observational cohort hazard ratio accurately cited."
            },
            {
                "id": 2,
                "text": "Short-term randomized controlled trials demonstrate that non-sugar sweeteners do not significantly raise blood glucose compared to placebos.",
                "quote": "Short-term randomized controlled trials generally show non-sugar sweeteners do not adversely elevate blood glucose or HbA1c compared to placebo.",
                "domain": "who.int",
                "url": "https://who.int/publications/i/item/9789240046429",
                "status": "kept",
                "confidence": "medium",
                "entailment": "supports",
                "reason": "WHO systematic review findings verified."
            },
            {
                "id": 3,
                "text": "The NutriNet-Santé study proved that artificial sweeteners cause 50% of heart attacks through direct vascular poisoning.",
                "quote": "intake was associated with a 9% increased risk of cardiovascular events",
                "domain": "bmj.com",
                "url": "https://bmj.com/content/378/bmj-2022-071204",
                "status": "dropped",
                "confidence": "low",
                "entailment": "unsupported",
                "reason": "Causal exaggeration: observational correlation falsely characterized as direct vascular poisoning."
            }
        ],
        "conflicts": [
            {
                "source_a": "bmj.com (NutriNet-Santé)",
                "claim_a": "Observational cohort finds 9% increased cardiovascular risk with sweetener consumption.",
                "source_b": "who.int (Systematic Review)",
                "claim_b": "RCTs demonstrate no acute elevation in glycemic or metabolic dysfunction.",
                "note": "Methodological disagreement: observational cohort residual confounding vs short-term RCT physiological neutrality."
            }
        ]
    },
    {
        "id": "report-10",
        "title": "Remote Work Impact on Software Engineering Productivity (Contested)",
        "question": "What is the empirical effect of full-time remote work on software engineer productivity and PR cycle times across large tech organizations?",
        "summary": "Large-scale organizational studies reveal contrasting outcomes: Microsoft network analysis documented increased communication silos and reduced cross-group ties during company-wide remote periods, whereas randomized trials by NBER indicated hybrid schedules reduced attrition by 35% with neutral performance impacts.",
        "mode": "multi",
        "claims": [
            {
                "id": 1,
                "text": "A study of over 61,000 Microsoft employees showed company-wide remote work caused communication networks to become more siloed with reduced cross-group ties.",
                "quote": "Analysis of 61,182 Microsoft employees found firm-wide remote work caused collaboration networks to become more static and siloed, with fewer cross-group bridges.",
                "domain": "microsoft.com",
                "url": "https://microsoft.com/en-us/research/publication/the-effects-of-remote-work-on-collaboration/",
                "status": "kept",
                "confidence": "medium",
                "entailment": "supports",
                "reason": "Organizational network study numbers verified."
            },
            {
                "id": 2,
                "text": "An NBER randomized trial across 1,612 engineers found hybrid work decreased employee turnover by 35% with neutral impact on performance ratings.",
                "quote": "In a randomized trial of 1,612 engineers, hybrid work reduced attrition by 35% with no measurable decrease in performance ratings or promotion rates.",
                "domain": "nber.org",
                "url": "https://nber.org/papers/w30664",
                "status": "kept",
                "confidence": "high",
                "entailment": "supports",
                "reason": "Empirical RCT metrics confirmed."
            },
            {
                "id": 3,
                "text": "Full-time remote work universally reduced individual pull request merge counts by 85% across all software teams.",
                "quote": "Analysis of 61,182 Microsoft employees found firm-wide remote work",
                "domain": "microsoft.com",
                "url": "https://microsoft.com/en-us/research/publication/the-effects-of-remote-work-on-collaboration/",
                "status": "dropped",
                "confidence": "low",
                "entailment": "unsupported",
                "reason": "Overgeneralization: 85% PR drop was completely fabricated by LLM draft."
            }
        ],
        "conflicts": [
            {
                "source_a": "microsoft.com (Network Analysis)",
                "claim_a": "Remote work degrades cross-team bridging and isolates collaboration clusters.",
                "source_b": "nber.org (Randomized Trial)",
                "claim_b": "Hybrid work yields 35% lower attrition without detectable performance degradation.",
                "note": "Organizational network siloing versus measured output retention trade-off."
            }
        ]
    }
]

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CiteGuard — Synthetic Demo Gallery</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0b0f19;
      --card-bg: #111827;
      --border: #1f2937;
      --accent: #3b82f6;
      --accent-hover: #2563eb;
      --text: #f3f4f6;
      --text-muted: #9ca3af;
      --kept: #10b981;
      --dropped: #ef4444;
      --conflict: #f59e0b;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      background: var(--bg);
      color: var(--text);
      display: flex;
      height: 100vh;
      overflow: hidden;
    }
    #sidebar {
      width: 340px;
      background: #0d121f;
      border-right: 1px solid var(--border);
      display: flex;
      flex-direction: column;
      flex-shrink: 0;
    }
    .brand {
      padding: 20px;
      border-bottom: 1px solid var(--border);
    }
    .brand h1 {
      font-size: 1.25rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .badge {
      display: inline-block;
      font-size: 0.7rem;
      font-weight: 600;
      padding: 2px 8px;
      border-radius: 9999px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .badge-live { background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid #3b82f6; }
    .badge-kept { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #10b981; }
    .badge-dropped { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; }
    .badge-conflict { background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #f59e0b; }

    #report-list {
      flex: 1;
      overflow-y: auto;
      padding: 12px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .report-item {
      padding: 12px 14px;
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.15s ease;
      border: 1px solid transparent;
    }
    .report-item:hover {
      background: #172033;
      border-color: #26354f;
    }
    .report-item.active {
      background: #1e293b;
      border-color: var(--accent);
    }
    .report-item h4 {
      font-size: 0.88rem;
      font-weight: 600;
      margin-bottom: 4px;
      color: #e2e8f0;
    }
    .report-item p {
      font-size: 0.75rem;
      color: var(--text-muted);
    }

    #main-content {
      flex: 1;
      overflow-y: auto;
      padding: 36px 48px;
    }
    .report-header {
      margin-bottom: 28px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 24px;
    }
    .report-header h2 {
      font-size: 1.85rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      margin-bottom: 12px;
    }
    .meta-bar {
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
      font-size: 0.85rem;
      color: var(--text-muted);
      font-family: 'JetBrains Mono', monospace;
    }
    .section-title {
      font-size: 1.15rem;
      font-weight: 600;
      margin: 28px 0 16px 0;
      display: flex;
      align-items: center;
      gap: 8px;
      border-bottom: 1px solid #1e293b;
      padding-bottom: 8px;
    }
    .claim-card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 20px;
      margin-bottom: 16px;
      transition: border-color 0.2s;
    }
    .claim-card:hover { border-color: #374151; }
    .claim-card.status-dropped { border-left: 4px solid var(--dropped); }
    .claim-card.status-kept { border-left: 4px solid var(--kept); }
    .claim-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 10px;
    }
    .claim-text {
      font-size: 1rem;
      font-weight: 500;
      line-height: 1.5;
      margin-bottom: 14px;
    }
    .quote-box {
      background: #0b0f19;
      border: 1px solid #1f2937;
      border-radius: 6px;
      padding: 12px 16px;
      font-size: 0.88rem;
      font-style: italic;
      color: #94a3b8;
      margin-bottom: 12px;
    }
    .quote-box strong { color: #f8fafc; font-style: normal; }
    .audit-reason {
      font-size: 0.82rem;
      font-family: 'JetBrains Mono', monospace;
      color: #cbd5e1;
      padding: 8px 12px;
      background: rgba(30, 41, 59, 0.5);
      border-radius: 6px;
    }
    .conflict-card {
      background: rgba(245, 158, 11, 0.05);
      border: 1px solid rgba(245, 158, 11, 0.3);
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 16px;
    }
  </style>
</head>
<body>
  <div id="sidebar">
    <div class="brand">
      <h1>CiteGuard <span class="badge badge-live">Synthetic</span></h1>
      <p style="font-size: 0.78rem; color: var(--text-muted); margin-top: 4px;">Hand-written example reports. Not pipeline output; verdicts and metrics are synthetic placeholders.</p>
    </div>
    <div id="report-list"></div>
  </div>

  <div id="main-content">
    <div style="background:#451a03;border:1px solid #f59e0b;color:#fde68a;padding:10px 14px;border-radius:8px;margin-bottom:20px;font-size:0.85rem;">SYNTHETIC DEMO DATA: these reports and their kept/dropped verdicts are hand-written examples, not outputs of the pipeline. Metrics: synthetic placeholder - not measured.</div>
    <div id="report-view"></div>
  </div>

  <script>
    const reports = %REPORTS_JSON%;
    let activeIndex = 0;

    function renderSidebar() {
      const container = document.getElementById("report-list");
      container.innerHTML = "";
      reports.forEach((r, idx) => {
        const item = document.createElement("div");
        item.className = "report-item" + (idx === activeIndex ? " active" : "");
        item.onclick = () => selectReport(idx);
        item.innerHTML = `
          <h4>${r.title}</h4>
          <p>${r.claims.length} claims • synthetic</p>
        `;
        container.appendChild(item);
      });
    }

    function selectReport(idx) {
      activeIndex = idx;
      renderSidebar();
      renderActiveReport();
    }

    function renderActiveReport() {
      const r = reports[activeIndex];
      const container = document.getElementById("report-view");

      let claimsHtml = "";
      r.claims.forEach(c => {
        const isKept = c.status === "kept";
        claimsHtml += `
          <div class="claim-card status-${c.status}">
            <div class="claim-header">
              <span class="badge badge-${isKept ? 'kept' : 'dropped'}">${c.status.toUpperCase()}</span>
              <span style="font-size: 0.8rem; color: var(--text-muted)">Source: <a href="${c.url}" target="_blank" style="color: #60a5fa; text-decoration: none;">${c.domain}</a> • Confidence: ${c.confidence}</span>
            </div>
            <div class="claim-text">${c.text}</div>
            <div class="quote-box">
              <strong>Verbatim Quote:</strong> "${c.quote}"
            </div>
            <div class="audit-reason">
              <strong>Audit:</strong> [${c.entailment}] ${c.reason}
            </div>
          </div>
        `;
      });

      let conflictsHtml = "";
      if (r.conflicts && r.conflicts.length > 0) {
        r.conflicts.forEach(conf => {
          conflictsHtml += `
            <div class="conflict-card">
              <div style="font-weight: 600; color: #fbbf24; margin-bottom: 6px;">⚠️ Cross-Source Disagreement Detected</div>
              <div style="font-size: 0.88rem; margin-bottom: 4px;">• <strong>${conf.source_a}:</strong> ${conf.claim_a}</div>
              <div style="font-size: 0.88rem; margin-bottom: 8px;">• <strong>${conf.source_b}:</strong> ${conf.claim_b}</div>
              <div style="font-size: 0.82rem; color: #cbd5e1; font-style: italic;">Analysis: ${conf.note}</div>
            </div>
          `;
        });
      } else {
        conflictsHtml = `<p style="font-size: 0.88rem; color: var(--text-muted); font-style: italic;">No cross-source factual contradictions detected in verified claims.</p>`;
      }

      container.innerHTML = `
        <div class="report-header">
          <h2>${r.title}</h2>
          <div class="meta-bar">
            <span><strong>Mode:</strong> ${r.mode}</span>
            <span><strong>Tokens / cost / latency:</strong> ${r.tokens_used}</span>
          </div>
        </div>

        <div class="section-title">1. Executive Summary</div>
        <p style="font-size: 1rem; line-height: 1.6; color: #cbd5e1; margin-bottom: 24px;">${r.summary}</p>

        <div class="section-title">2. Verified Findings & Claim Audit Trail</div>
        ${claimsHtml}

        <div class="section-title">3. Where Sources Disagree</div>
        ${conflictsHtml}
      `;
    }

    renderSidebar();
    renderActiveReport();
  </script>
</body>
</html>
"""


def main():
    for r in GALLERY_REPORTS:
        r["tokens_used"] = PLACEHOLDER
        r["synthetic"] = True
    GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save reports JSON
    data_path = GALLERY_DIR / "synthetic_data.json"
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(GALLERY_REPORTS, f, indent=2)

    # Generate individual markdown reports
    for r in GALLERY_REPORTS:
        md_lines = [
            "> SYNTHETIC EXAMPLE: hand-written, not pipeline output.\n",
            f"# {r['title']}\n",
            f"**Research Question:** {r['question']}\n",
            "## 1. Executive Summary\n",
            f"{r['summary']}\n",
            "## 2. Findings per Sub-Question\n"
        ]
        for c in r["claims"]:
            status_tag = f"[{c['status'].upper()}]"
            md_lines.append(f"- {c['text']} *({status_tag} - Source: {c['domain']})*")
            md_lines.append(f"  > \"{c['quote']}\"\n")

        md_lines.append("## 3. Where Sources Disagree\n")
        if r.get("conflicts"):
            for conf in r["conflicts"]:
                md_lines.append(f"- **Disagreement:** {conf['source_a']} vs {conf['source_b']}")
                md_lines.append(f"  - Note: {conf['note']}\n")
        else:
            md_lines.append("*No cross-source factual contradictions detected.*\n")

        md_lines.append("## 4. Run Metadata\n")
        md_lines.append(f"- **Tokens / cost / latency:** {r['tokens_used']}")
        md_lines.append(f"- **Mode:** {r['mode']}\n")

        report_file = REPORTS_DIR / f"{r['id']}.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))

    # Generate index.html
    html_content = HTML_TEMPLATE.replace("%REPORTS_JSON%", json.dumps(GALLERY_REPORTS))
    index_file = GALLERY_DIR / "index.html"
    with open(index_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Generated 10 SYNTHETIC gallery reports and interactive index at {index_file}")


if __name__ == "__main__":
    main()

