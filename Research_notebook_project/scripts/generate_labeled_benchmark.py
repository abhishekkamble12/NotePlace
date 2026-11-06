# scripts/generate_labeled_benchmark.py
"""
Generates a SYNTHETIC claim table. Nothing here is human-labeled or produced by the verifier:
labels and verifier verdicts are hand-typed in RAW_BENCHMARK_DATA, rows are repeated, and
tokens/cost/latency are not measured. Do not report any statistic computed from it.
Shaped like 14 benchmark questions across 4 categories (factual synthesis, comparisons,
contested topics, thin-evidence topics), across Config A (single-agent, verifier off)
and Config B+C (multi-agent, shadow mode / verifier on).

Output: evals/synthetic_labeled_claims.csv (schema example for the real labeling work in Phase 2).
"""
import csv
from pathlib import Path

PLACEHOLDER = "synthetic placeholder - not measured"
BENCHMARK_QUESTIONS_PATH = Path(__file__).resolve().parents[1] / "evals" / "benchmark_questions.json"
OUTPUT_CSV_PATH = Path(__file__).resolve().parents[1] / "evals" / "synthetic_labeled_claims.csv"

# Comprehensive claim definitions mapped to questions
RAW_BENCHMARK_DATA = [
    # Q1: Perovskite Solar Cells
    {
        "q_id": "q1",
        "subq": "What is the certified power conversion efficiency of tandem cells?",
        "domain": "nature.com",
        "url": "https://nature.com/articles/s41560-023-01402-y",
        "quote": "Perovskite-silicon tandem solar cells reached a certified power conversion efficiency of 33.9% in late 2023.",
        "claims": [
            {
                "claim": "Certified power conversion efficiency of perovskite-silicon tandem cells reached 33.9% in late 2023.",
                "type": "faithful", "a1": "supported", "a2": "supported", "found": True, "entail": "supports", "action": "kept"
            },
            {
                "claim": "Perovskite-silicon tandem solar cells achieved a certified power conversion efficiency of 43.9% in 2023.",
                "type": "number_distortion", "a1": "unsupported", "a2": "unsupported", "found": False, "entail": "unsupported", "action": "dropped"
            },
            {
                "claim": "All commercial silicon panels have now been replaced with 33.9% perovskite tandems globally.",
                "type": "overgeneralization", "a1": "unsupported", "a2": "unsupported", "found": True, "entail": "unsupported", "action": "dropped"
            }
        ]
    },
    {
        "q_id": "q1",
        "subq": "What are the degradation mechanisms under ISOS thermal cycling?",
        "domain": "sciencedirect.com",
        "url": "https://sciencedirect.com/science/article/pii/S254243512300456X",
        "quote": "Encapsulated devices retained 95% of initial efficiency after 1,000 hours of ISOS-L-2 continuous light soaking at 65°C.",
        "claims": [
            {
                "claim": "Encapsulated tandem cells maintained 95% of initial efficiency after 1,000 hours under ISOS-L-2 testing at 65°C.",
                "type": "faithful", "a1": "supported", "a2": "supported", "found": True, "entail": "supports", "action": "kept"
            },
            {
                "claim": "Unencapsulated perovskite layers exhibit zero halide segregation or phase breakdown at temperatures above 100°C.",
                "type": "unsupported_extrapolation", "a1": "unsupported", "a2": "unsupported", "found": False, "entail": "unsupported", "action": "dropped"
            }
        ]
    },

    # Q2: FlashAttention-2
    {
        "q_id": "q2",
        "subq": "How does FlashAttention-2 optimize memory IO and parallelization?",
        "domain": "arxiv.org",
        "url": "https://arxiv.org/abs/2307.08691",
        "quote": "FlashAttention-2 yields a 2x speedup over FlashAttention by reducing non-matmul FLOPs and parallelizing over the sequence length dimension.",
        "claims": [
            {
                "claim": "FlashAttention-2 achieves roughly a 2x speedup over FlashAttention by cutting non-matmul FLOPs and parallelizing over sequence length.",
                "type": "faithful", "a1": "supported", "a2": "supported", "found": True, "entail": "supports", "action": "kept"
            },
            {
                "claim": "FlashAttention-2 achieves a 10x speedup by completely eliminating SRAM memory traffic on Nvidia A100 GPUs.",
                "type": "number_distortion", "a1": "unsupported", "a2": "unsupported", "found": False, "entail": "unsupported", "action": "dropped"
            },
            {
                "claim": "FlashAttention-2 partitions work across sequence length rather than solely batch and head dimensions.",
                "type": "faithful", "a1": "supported", "a2": "supported", "found": True, "entail": "supports", "action": "kept"
            }
        ]
    },

    # Q3: CRISPR Base Editing
    {
        "q_id": "q3",
        "subq": "What are clinical efficacy and adverse rates of base editing in Phase 1/2 trials?",
        "domain": "nejm.org",
        "url": "https://nejm.org/doi/full/10.1056/NEJMoa2300700",
        "quote": "In the Phase 1b trial of adenine base editor VERVE-101, blood PCSK9 levels were reduced by up to 84% and LDL-C by 55% at highest dose cohorts.",
        "claims": [
            {
                "claim": "In Phase 1b clinical evaluation of VERVE-101, highest dose participants showed up to 84% reduction in PCSK9 and 55% in LDL cholesterol.",
                "type": "faithful", "a1": "supported", "a2": "supported", "found": True, "entail": "supports", "action": "kept"
            },
            {
                "claim": "VERVE-101 completely cured 100% of familial hypercholesterolemia patients with zero transient liver enzyme elevations.",
                "type": "overgeneralization", "a1": "unsupported", "a2": "unsupported", "found": True, "entail": "unsupported", "action": "dropped"
            },
            {
                "claim": "Base editors use catalytically impaired Cas nucleases fused to deaminases to introduce point mutations without double-strand breaks.",
                "type": "faithful", "a1": "supported", "a2": "supported", "found": True, "entail": "supports", "action": "kept"
            }
        ]
    },

    # Q4: Quantum Processors (Sycamore vs Eagle)
    {
        "q_id": "q4",
        "subq": "What are the two-qubit gate fidelities and qubit counts of Sycamore and Eagle?",
        "domain": "ibm.com",
        "url": "https://research.ibm.com/blog/127-qubit-quantum-processor-eagle",
        "quote": "IBM Eagle features 127 superconducting qubits with heavy-hexagonal layout and average two-qubit gate error rates around 1.0e-2.",
        "claims": [
            {
                "claim": "IBM Eagle operates 127 qubits arranged in a heavy-hex layout with average two-qubit error rates around 1.0e-2.",
                "type": "faithful", "a1": "supported", "a2": "supported", "found": True, "entail": "supports", "action": "kept"
            },
            {
                "claim": "IBM Eagle achieved fault-tolerant quantum error correction with a physical qubit fidelity exceeding 99.999% in 2021.",
                "type": "unsupported_extrapolation", "a1": "unsupported", "a2": "unsupported", "found": False, "entail": "unsupported", "action": "dropped"
            }
        ]
    },
    {
        "q_id": "q4",
        "subq": "What were the benchmarks for Google's Sycamore quantum processor?",
        "domain": "nature.com",
        "url": "https://nature.com/articles/s41586-019-1666-5",
        "quote": "The 53-qubit Sycamore processor sampled random quantum circuit instances in 200 seconds, estimating classical supercomputers would take 10,000 years.",
        "claims": [
            {
                "claim": "Google Sycamore sampled random quantum circuit instances in 200 seconds using 53 operational transmon qubits.",
                "type": "faithful", "a1": "supported", "a2": "supported", "found": True, "entail": "supports", "action": "kept"
            },
            {
                "claim": "Google Sycamore ran Shor's algorithm to factor a 2048-bit RSA key in 200 seconds.",
                "type": "entity_swap", "a1": "unsupported", "a2": "unsupported", "found": False, "entail": "unsupported", "action": "dropped"
            }
        ]
    },

    # Q5: pgvector HNSW vs IVFFlat
    {
        "q_id": "q5",
        "subq": "How do HNSW and IVFFlat compare in pgvector for build time and query QPS?",
        "domain": "jkatz05.com",
        "url": "https://jkatz05.com/post/postgres/pgvector-hnsw-performance/",
        "quote": "On 1M 1536-dimensional embeddings, pgvector HNSW achieved 20x higher QPS at 99% recall than IVFFlat, though index build time was 10x longer.",
        "claims": [
            {
                "claim": "pgvector HNSW indexes deliver significantly higher query QPS at 99% recall compared to IVFFlat, at the expense of substantially longer build times.",
                "type": "faithful", "a1": "supported", "a2": "supported", "found": True, "entail": "supports", "action": "kept"
            },
            {
                "claim": "IVFFlat indexes require zero training lists and build instantly without memory overhead.",
                "type": "unsupported_extrapolation", "a1": "unsupported", "a2": "unsupported", "found": False, "entail": "unsupported", "action": "dropped"
            },
            {
                "claim": "pgvector 0.5.0 introduced iterative index builds and HNSW vector search directly within PostgreSQL.",
                "type": "faithful", "a1": "supported", "a2": "supported", "found": True, "entail": "supports", "action": "kept"
            }
        ]
    },

    # Q6: Llama 3 70B vs Mistral Large
    {
        "q_id": "q6",
        "subq": "Compare context length, licensing, and RAG benchmarks between Llama 3 and Mistral Large.",
        "domain": "ai.meta.com",
        "url": "https://ai.meta.com/blog/meta-llama-3/",
        "quote": "Llama 3 70B was released with an 8k context window under the Meta Llama 3 Community License, later expanded to 128k with Llama 3.1.",
        "claims": [
            {
                "claim": "Initial Meta Llama 3 70B models supported an 8k context window under the Llama 3 Community License.",
                "type": "faithful", "a1": "supported", "a2": "supported", "found": True, "entail": "supports", "action": "kept"
            },
            {
                "claim": "Meta Llama 3 is licensed under pure unencumbered Apache 2.0 with no commercial MAU threshold conditions.",
                "type": "entity_swap", "a1": "unsupported", "a2": "unsupported", "found": False, "entail": "unsupported", "action": "dropped"
            }
        ]
    },
    {
        "q_id": "q6",
        "subq": "How does Mistral Large handle context and multilingual RAG tasks?",
        "domain": "mistral.ai",
        "url": "https://mistral.ai/news/mistral-large/",
        "quote": "Mistral Large natively supports a 32k context window and precise JSON output formatting with native support for English, French, German, Spanish, and Italian.",
        "claims": [
            {
                "claim": "Mistral Large natively provides a 32k context window and structured JSON formatting across English and European languages.",
                "type": "faithful", "a1": "supported", "a2": "supported", "found": True, "entail": "supports", "action": "kept"
            },
            {
                "claim": "Mistral Large is a 7B parameter open-weights model available for local offline execution on iPhone 12.",
                "type": "entity_swap", "a1": "unsupported", "a2": "unsupported", "found": False, "entail": "unsupported", "action": "dropped"
            }
        ]
    },

    # Q7: Rust vs Go for Network Proxies
    {
        "q_id": "q7",
        "subq": "Compare memory footprint, tail latency, and GC pauses between Rust and Go.",
        "domain": "blog.cloudflare.com",
        "url": "https://blog.cloudflare.com/how-we-built-pingora-in-rust/",
        "quote": "Cloudflare's Rust-based Pingora proxy consumed 70% less CPU and 67% less memory compared to their previous NGINX/Lua service under identical traffic.",
        "claims": [
            {
                "claim": "Cloudflare reported that switching their proxy infrastructure to Rust (Pingora) yielded 70% CPU reduction and 67% memory reduction.",
                "type": "faithful", "a1": "supported", "a2": "supported", "found": True, "entail": "supports", "action": "kept"
            },
            {
                "claim": "Go proxies completely avoid stop-the-world garbage collection pauses under 500,000 concurrent TLS connections.",
                "type": "overgeneralization", "a1": "unsupported", "a2": "unsupported", "found": False, "entail": "unsupported", "action": "dropped"
            },
            {
                "claim": "Rust memory management relies on compile-time borrow checking rather than runtime tracing garbage collection.",
                "type": "faithful", "a1": "supported", "a2": "supported", "found": True, "entail": "supports", "action": "kept"
            }
        ]
    },

    # Q8: Apache Iceberg vs Delta Lake
    {
        "q_id": "q8",
        "subq": "Compare ACID guarantees, hidden partitioning, and engine support.",
        "domain": "tabular.io",
        "url": "https://tabular.io/blog/iceberg-hidden-partitioning/",
        "quote": "Apache Iceberg's hidden partitioning prevents silent data corruption by generating partition values automatically without user-specified column derivations.",
        "claims": [
            {
                "claim": "Apache Iceberg uses hidden partitioning to automatically derive partition values from timestamp columns without query rewrites.",
                "type": "faithful", "a1": "supported", "a2": "supported", "found": True, "entail": "supports", "action": "kept"
            },
            {
                "claim": "Delta Lake format requires proprietary Databricks runtime engines and cannot be read by Apache Spark or Trino.",
                "type": "unsupported_extrapolation", "a1": "unsupported", "a2": "unsupported", "found": False, "entail": "unsupported", "action": "dropped"
            },
            {
                "claim": "Both Iceberg and Delta Lake rely on Apache Parquet as the primary underlying columnar file storage layer.",
                "type": "faithful", "a1": "supported", "a2": "supported", "found": True, "entail": "supports", "action": "kept"
            }
        ]
    },

    # Q9: Contested Topic - Artificial Sweeteners
    {
        "q_id": "q9",
        "subq": "What do observational cohort studies report on aspartame and cardiovascular risk?",
        "domain": "bmj.com",
        "url": "https://bmj.com/content/378/bmj-2022-071204",
        "quote": "In the NutriNet-Santé cohort of 102,865 adults, artificial sweetener intake was associated with a 9% increased risk of cardiovascular events (HR 1.09, 95% CI 1.01 to 1.18).",
        "claims": [
            {
                "claim": "The NutriNet-Santé observational cohort observed a statistically significant 9% relative increase in cardiovascular events among high artificial sweetener consumers.",
                "type": "faithful", "a1": "supported", "a2": "supported", "found": True, "entail": "supports", "action": "kept"
            },
            {
                "claim": "The NutriNet-Santé study proved that artificial sweeteners cause 50% of heart attacks through direct vascular poisoning.",
                "type": "causal_inversion", "a1": "unsupported", "a2": "unsupported", "found": False, "entail": "unsupported", "action": "dropped"
            }
        ]
    },
    {
        "q_id": "q9",
        "subq": "What do randomized controlled trials conclude regarding non-sugar sweeteners and glycemic response?",
        "domain": "who.int",
        "url": "https://who.int/publications/i/item/9789240046429",
        "quote": "Short-term randomized controlled trials generally show non-sugar sweeteners do not adversely elevate blood glucose or HbA1c compared to placebo.",
        "claims": [
            {
                "claim": "Short-term randomized controlled trials demonstrate that non-sugar sweeteners do not significantly raise blood glucose compared to placebos.",
                "type": "faithful", "a1": "supported", "a2": "supported", "found": True, "entail": "supports", "action": "kept"
            },
            {
                "claim": "The World Health Organization recommends consuming 500mg of sucralose daily as a certified diabetes medication.",
                "type": "entity_swap", "a1": "unsupported", "a2": "unsupported", "found": False, "entail": "unsupported", "action": "dropped"
            }
        ]
    },

    # Q10: Contested Topic - Remote Work Productivity
    {
        "q_id": "q10",
        "subq": "What empirical metrics do tech companies report on developer output during remote work?",
        "domain": "microsoft.com",
        "url": "https://microsoft.com/en-us/research/publication/the-effects-of-remote-work-on-collaboration/",
        "quote": "Analysis of 61,182 Microsoft employees found firm-wide remote work caused collaboration networks to become more static and siloed, with fewer cross-group bridges.",
        "claims": [
            {
                "claim": "A study of over 61,000 Microsoft employees showed company-wide remote work caused communication networks to become more siloed with reduced cross-group ties.",
                "type": "faithful", "a1": "supported", "a2": "supported", "found": True, "entail": "supports", "action": "kept"
            },
            {
                "claim": "Full-time remote work universally reduced individual pull request merge counts by 85% across all software teams.",
                "type": "overgeneralization", "a1": "unsupported", "a2": "unsupported", "found": False, "entail": "unsupported", "action": "dropped"
            }
        ]
    },
    {
        "q_id": "q10",
        "subq": "What do randomized controlled trials on hybrid and remote work find regarding productivity?",
        "domain": "nber.org",
        "url": "https://nber.org/papers/w30664",
        "quote": "In a randomized trial of 1,612 engineers, hybrid work reduced attrition by 35% with no measurable decrease in performance ratings or promotion rates.",
        "claims": [
            {
                "claim": "An NBER randomized trial across 1,612 engineers found hybrid work decreased employee turnover by 35% with neutral impact on performance ratings.",
                "type": "faithful", "a1": "supported", "a2": "supported", "found": True, "entail": "supports", "action": "kept"
            },
            {
                "claim": "The NBER study proved that working from office 5 days a week produces 4x higher patent filings than hybrid schedules.",
                "type": "unsupported_extrapolation", "a1": "unsupported", "a2": "unsupported", "found": False, "entail": "unsupported", "action": "dropped"
            }
        ]
    },

    # Q11: Contested Topic - LK-99 Superconductivity
    {
        "q_id": "q11",
        "subq": "What did the original authors claim regarding LK-99?",
        "domain": "arxiv.org",
        "url": "https://arxiv.org/abs/2307.12008",
        "quote": "Authors claimed modified lead-apatite crystals (LK-99) exhibited room-temperature superconductivity at ambient pressure with critical temperatures up to 400 K.",
        "claims": [
            {
                "claim": "Preprint authors claimed modified lead-apatite LK-99 exhibited room-temperature superconductivity with critical temperature above 400 K.",
                "type": "faithful", "a1": "supported", "a2": "supported", "found": True, "entail": "supports", "action": "kept"
            },
            {
                "claim": "LK-99 was certified by NIST as a validated high-temperature superconductor powering commercial magnetic levitation trains.",
                "type": "unsupported_extrapolation", "a1": "unsupported", "a2": "unsupported", "found": False, "entail": "unsupported", "action": "dropped"
            }
        ]
    },
    {
        "q_id": "q11",
        "subq": "What did international replication efforts determine about LK-99 resistivity drops?",
        "domain": "nature.com",
        "url": "https://nature.com/articles/d41586-023-02585-7",
        "quote": "Independent replications by the Max Planck Institute synthesized pure single crystals and proved copper sulfide (Cu2S) impurities caused the sharp resistivity drop, confirming LK-99 is an insulator.",
        "claims": [
            {
                "claim": "Max Planck Institute researchers proved the sharp resistance drop in LK-99 was caused by copper sulfide impurities rather than superconductivity.",
                "type": "faithful", "a1": "supported", "a2": "supported", "found": True, "entail": "supports", "action": "kept"
            },
            {
                "claim": "Max Planck Institute confirmed zero-resistance state in pure LK-99 single crystals at 300 Kelvin.",
                "type": "entity_swap", "a1": "unsupported", "a2": "unsupported", "found": False, "entail": "unsupported", "action": "dropped"
            }
        ]
    },

    # Q12: Thin-Evidence Topic - Memristor Commercial Deployment
    {
        "q_id": "q12",
        "subq": "What verified commercial customer tape-outs exist for analog memristor accelerators in 2024?",
        "domain": "ieee.org",
        "url": "https://ieeexplore.ieee.org/document/9774652",
        "quote": "Experimental 64-tile analog memristor prototype chips achieved 64 TOPS/W in research testbeds, but high device-to-device variability remains an obstacle for commercial mass fabrication.",
        "claims": [
            {
                "claim": "Experimental memristor prototypes demonstrated energy efficiency up to 64 TOPS/W in testbeds, while device variability challenges large-scale commercial yields.",
                "type": "faithful", "a1": "supported", "a2": "supported", "found": True, "entail": "supports", "action": "kept"
            },
            {
                "claim": "Over 50 million smartphones sold in 2024 integrate 3nm commercial analog memristor NPUs as primary accelerators.",
                "type": "thin_evidence_hallucination", "a1": "unsupported", "a2": "unsupported", "found": False, "entail": "unsupported", "action": "dropped"
            }
        ]
    },

    # Q13: Thin-Evidence Topic - Bubble Fusion
    {
        "q_id": "q13",
        "subq": "What are the reproducible experimental metrics for acoustic cavitation bubble fusion?",
        "domain": "aps.org",
        "url": "https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.96.034301",
        "quote": "Follow-up experiments by independent laboratories failed to observe statistically significant 2.45 MeV neutron emissions during acoustic cavitation of deuterated acetone.",
        "claims": [
            {
                "claim": "Independent replication experiments failed to detect statistically significant 2.45 MeV neutron emissions during cavitation of deuterated acetone.",
                "type": "faithful", "a1": "supported", "a2": "supported", "found": True, "entail": "supports", "action": "kept"
            },
            {
                "claim": "Commercial acoustic bubble fusion reactors currently supply 15% of France's domestic electrical grid.",
                "type": "thin_evidence_hallucination", "a1": "unsupported", "a2": "unsupported", "found": False, "entail": "unsupported", "action": "dropped"
            }
        ]
    },

    # Q14: Thin-Evidence Topic - Apple AI Server Silicon
    {
        "q_id": "q14",
        "subq": "What verified public specifications exist for Apple internal AI server silicon?",
        "domain": "reuters.com",
        "url": "https://reuters.com/technology/apple-plans-make-chips-run-ai-software-data-centers-2024-05-06/",
        "quote": "Reports indicate Apple is developing server chips under codename ACDC using TSMC 3nm nodes to power Cloud Intelligence, though architectural details have not been officially disclosed.",
        "claims": [
            {
                "claim": "Industry reporting indicates Apple's internal server-side chip initiative (Project ACDC) utilizes TSMC foundry nodes to run data center AI workloads.",
                "type": "faithful", "a1": "supported", "a2": "supported", "found": True, "entail": "supports", "action": "kept"
            },
            {
                "claim": "Apple officially published a 500-page architectural whitepaper detailing 512-core M5 Server CPUs with 10 TB/s HBM4 memory in February 2024.",
                "type": "thin_evidence_hallucination", "a1": "unsupported", "a2": "unsupported", "found": False, "entail": "unsupported", "action": "dropped"
            }
        ]
    }
]


def expand_benchmark_dataset():
    """Expands RAW_BENCHMARK_DATA into Config A, B, and C labeled claims."""
    rows = []
    claim_counter = 1
    job_counter = 100

    # We generate entries across Config A (single-agent, verifier off) and Config B/C (multi-agent, shadow mode / verifier on)
    for q_idx, item in enumerate(RAW_BENCHMARK_DATA):
        q_id = item["q_id"]
        subq = item["subq"]
        domain = item["domain"]
        url = item["url"]
        quote = item["quote"]
        claims = item["claims"]

        # Run multi-mode (Config B & C)
        job_b_c = job_counter + q_idx
        for c in claims:
            # Annotator 2 labels 40 specific claims for inter-annotator agreement
            a2_val = ""  # no second annotator exists

            # In Config B (draft, verifier off): all claims appear
            # In Config C (verifier on): only claims with action 'kept' survive
            # We record both in the unified shadow-mode dataset
            row = {
                "claim_id": claim_counter,
                "job_id": job_b_c,
                "question_id": q_id,
                "subquestion": subq,
                "mode": "multi",
                "config": "B_and_C",
                "claim_text": c["claim"],
                "quote": quote,
                "source_url": url,
                "source_domain": domain,
                "perturbation_type": c["type"],
                "annotator_1_label": c["a1"],
                "annotator_2_label": a2_val,
                "verifier_quote_found": "True" if c["found"] else "False",
                "verifier_entailment": c["entail"],
                "verifier_action": c["action"],  # 'kept' or 'dropped'
                "survives_in_config_c": "True" if c["action"] == "kept" else "False",
                "tokens_used": PLACEHOLDER,
                "cost_usd_modeled": PLACEHOLDER,
                "latency_seconds": PLACEHOLDER,
            }
            rows.append(row)
            claim_counter += 1

        # Run single-mode (Config A: verifier off) for first 8 questions per Spec §9.1
        if q_idx < 8:
            job_a = 200 + q_idx
            for c in claims:
                row_a = {
                    "claim_id": claim_counter,
                    "job_id": job_a,
                    "question_id": q_id,
                    "subquestion": subq,
                    "mode": "single",
                    "config": "A",
                    "claim_text": c["claim"],
                    "quote": quote,
                    "source_url": url,
                    "source_domain": domain,
                    "perturbation_type": c["type"],
                    "annotator_1_label": c["a1"],
                    "annotator_2_label": "",
                    "verifier_quote_found": "N/A",
                    "verifier_entailment": "N/A",
                    "verifier_action": "kept",  # Verifier off in Config A
                    "survives_in_config_c": "False",
                    "tokens_used": PLACEHOLDER,
                    "cost_usd_modeled": PLACEHOLDER,
                    "latency_seconds": PLACEHOLDER,
                }
                rows.append(row_a)
                claim_counter += 1

    # Replicate variations to reach ~195-200 labeled claims
    multiplier = 3
    expanded_rows = []
    final_id = 1
    for m in range(multiplier):
        for r in rows:
            new_r = dict(r)
            new_r["claim_id"] = final_id
            new_r["annotator_2_label"] = ""
            new_r["provenance"] = PLACEHOLDER
            expanded_rows.append(new_r)
            final_id += 1
            if len(expanded_rows) >= 195:
                break
        if len(expanded_rows) >= 195:
            break

    return expanded_rows


def main():
    rows = expand_benchmark_dataset()
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with open(OUTPUT_CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {len(rows)} SYNTHETIC claim rows saved to {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
