"""initial_schema

Revision ID: 001_initial_schema
Revises:
Create Date: 2026-11-06 00:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from alembic import op

# revision identifiers, used by Alembic.
revision: str = '001_initial_schema'
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 1. Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    # 2. Table: jobs
    op.create_table(
        'jobs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('question', sa.Text(), nullable=False),
        sa.Column('max_tokens', sa.Integer(), nullable=False, server_default='50000'),
        sa.Column('max_minutes', sa.Integer(), nullable=False, server_default='8'),
        sa.Column('mode', sa.String(length=10), nullable=False, server_default='multi'),
        sa.Column('status', sa.String(length=20), nullable=False, server_default='queued'),
        sa.Column('stage', sa.String(length=20), nullable=False, server_default='planning'),
        sa.Column('deadline_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('report_md', sa.Text(), nullable=False, server_default=''),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )

    # 3. Table: subquestions
    op.create_table(
        'subquestions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('job_id', sa.Integer(), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('status', sa.String(length=20), nullable=False, server_default='pending'),
        sa.ForeignKeyConstraint(['job_id'], ['jobs.id']),
        sa.PrimaryKeyConstraint('id')
    )

    # 4. Table: sources
    op.create_table(
        'sources',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('job_id', sa.Integer(), nullable=False),
        sa.Column('url', sa.String(length=1000), nullable=False),
        sa.Column('domain', sa.String(length=255), nullable=False, server_default=''),
        sa.Column('content_hash', sa.String(length=64), nullable=False),
        sa.Column('snapshot', sa.Text(), nullable=False),
        sa.Column('fetched_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['job_id'], ['jobs.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('job_id', 'content_hash')
    )

    # 5. Table: chunks
    op.create_table(
        'chunks',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('source_id', sa.Integer(), nullable=False),
        sa.Column('subq_id', sa.Integer(), nullable=True),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('embedding', Vector(384), nullable=True),
        sa.ForeignKeyConstraint(['source_id'], ['sources.id']),
        sa.ForeignKeyConstraint(['subq_id'], ['subquestions.id']),
        sa.PrimaryKeyConstraint('id')
    )

    # 6. Table: claims
    op.create_table(
        'claims',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('job_id', sa.Integer(), nullable=False),
        sa.Column('subq_id', sa.Integer(), nullable=True),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('source_id', sa.Integer(), nullable=True),
        sa.Column('quote', sa.Text(), nullable=False),
        sa.Column('quote_found', sa.Boolean(), nullable=True),
        sa.Column('entailment', sa.String(length=15), nullable=False, server_default=''),
        sa.Column('entailment_reason', sa.Text(), nullable=False, server_default=''),
        sa.Column('final_status', sa.String(length=15), nullable=False, server_default='pending'),
        sa.Column('rewrite_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('confidence', sa.String(length=10), nullable=False, server_default='low'),
        sa.Column('shadow_verdict', sa.String(length=15), nullable=False, server_default=''),
        sa.Column('human_label', sa.String(length=15), nullable=False, server_default=''),
        sa.ForeignKeyConstraint(['job_id'], ['jobs.id']),
        sa.ForeignKeyConstraint(['source_id'], ['sources.id']),
        sa.ForeignKeyConstraint(['subq_id'], ['subquestions.id']),
        sa.PrimaryKeyConstraint('id')
    )

    # 7. Table: conflicts
    op.create_table(
        'conflicts',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('job_id', sa.Integer(), nullable=False),
        sa.Column('claim_a_id', sa.Integer(), nullable=False),
        sa.Column('claim_b_id', sa.Integer(), nullable=False),
        sa.Column('note', sa.Text(), nullable=False, server_default=''),
        sa.ForeignKeyConstraint(['claim_a_id'], ['claims.id']),
        sa.ForeignKeyConstraint(['claim_b_id'], ['claims.id']),
        sa.ForeignKeyConstraint(['job_id'], ['jobs.id']),
        sa.PrimaryKeyConstraint('id')
    )

    # 8. Table: ledger
    op.create_table(
        'ledger',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('scope_type', sa.String(length=20), nullable=False, server_default='job'),
        sa.Column('scope_id', sa.Integer(), nullable=False),
        sa.Column('phase', sa.String(length=20), nullable=False),
        sa.Column('kind', sa.String(length=20), nullable=False, server_default='llm'),
        sa.Column('model', sa.String(length=50), nullable=False, server_default=''),
        sa.Column('tokens_in', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('tokens_out', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('cost_usd_modeled', sa.Numeric(precision=8, scale=5), nullable=False, server_default='0.0'),
        sa.Column('latency_ms', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('cache_hit', sa.Boolean(), nullable=False, server_default=sa.text('false')),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_ledger_scope', 'ledger', ['scope_type', 'scope_id'], unique=False)

    # 9. Table: verifier_testset
    op.create_table(
        'verifier_testset',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('quote', sa.Text(), nullable=False),
        sa.Column('claim', sa.Text(), nullable=False),
        sa.Column('context', sa.Text(), nullable=False, server_default=''),
        sa.Column('perturbation_type', sa.String(length=50), nullable=False),
        sa.Column('label', sa.String(length=15), nullable=False),
        sa.Column('source', sa.String(length=500), nullable=False, server_default=''),
        sa.PrimaryKeyConstraint('id')
    )


def downgrade() -> None:
    op.drop_table('verifier_testset')
    op.drop_index('ix_ledger_scope', table_name='ledger')
    op.drop_table('ledger')
    op.drop_table('conflicts')
    op.drop_table('claims')
    op.drop_table('chunks')
    op.drop_table('sources')
    op.drop_table('subquestions')
    op.drop_table('jobs')
