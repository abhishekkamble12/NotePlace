# Apache Iceberg vs Delta Lake: Data Lakehouse Table Architecture

**Research Question:** Compare Apache Iceberg and Delta Lake table formats regarding ACID guarantees, hidden partitioning, and engine-agnostic reader support.

## 1. Executive Summary

Both Iceberg and Delta Lake provide ACID transactions over Parquet files using optimistic concurrency control. Iceberg distinguishes itself through engine-agnostic hidden partitioning and fine-grained snapshot metadata, while Delta Lake features deep integration with the Databricks and Spark ecosystems.

## 2. Findings per Sub-Question

- Apache Iceberg uses hidden partitioning to automatically derive partition values from timestamp columns without query rewrites. *([KEPT] - Source: tabular.io)*
  > "Apache Iceberg's hidden partitioning prevents silent data corruption by generating partition values automatically without user-specified column derivations."

- Both Iceberg and Delta Lake rely on Apache Parquet as the primary underlying columnar file storage layer. *([KEPT] - Source: tabular.io)*
  > "Both table formats store columnar row groups in standard immutable Apache Parquet format."

## 3. Where Sources Disagree

*No cross-source factual contradictions detected.*

## 4. Run Metadata

- **Tokens / cost / latency:** synthetic placeholder - not measured
- **Mode:** multi
