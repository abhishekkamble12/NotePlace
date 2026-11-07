"""notebooks + PDF sources + structure-aware chunk metadata + full-text index

Additive only: existing job sources/chunks keep working (job_id stays set, new columns are NULL).
Sources in a notebook have job_id NULL, so every job-scoped query ignores them.

Revision ID: 004_notebooks_hybrid_retrieval
Revises: 003_ledger_usage_accounting
"""
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = '004_notebooks_hybrid_retrieval'
down_revision: str | None = '003_ledger_usage_accounting'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table('users', sa.Column('id', sa.Integer(), primary_key=True),
                    sa.Column('external_sub', sa.String(255), nullable=False, unique=True))
    op.create_table('notebooks', sa.Column('id', sa.Integer(), primary_key=True),
                    sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id'), nullable=False, index=True),
                    sa.Column('title', sa.String(255), nullable=False, server_default=''))

    op.alter_column('sources', 'job_id', existing_type=sa.Integer(), nullable=True)
    op.add_column('sources', sa.Column('notebook_id', sa.Integer(), sa.ForeignKey('notebooks.id'), nullable=True))
    op.add_column('sources', sa.Column('kind', sa.String(10), nullable=False, server_default='url'))
    op.add_column('sources', sa.Column('title', sa.String(500), nullable=False, server_default=''))
    op.add_column('sources', sa.Column('page_offsets', postgresql.ARRAY(sa.Integer()), nullable=True))
    op.create_index('ix_sources_notebook', 'sources', ['notebook_id'])
    # NULLs are distinct in Postgres, so job-scoped URL sources are unaffected
    op.create_unique_constraint('uq_sources_notebook_hash', 'sources', ['notebook_id', 'content_hash'])

    op.add_column('chunks', sa.Column('notebook_id', sa.Integer(), sa.ForeignKey('notebooks.id'), nullable=True))
    op.add_column('chunks', sa.Column('section', sa.Text(), nullable=True))
    op.add_column('chunks', sa.Column('page_start', sa.Integer(), nullable=True))
    op.add_column('chunks', sa.Column('page_end', sa.Integer(), nullable=True))
    op.add_column('chunks', sa.Column('chunker', sa.String(20), nullable=True))
    op.add_column('chunks', sa.Column(
        'tsv', postgresql.TSVECTOR(), sa.Computed("to_tsvector('english', text)", persisted=True), nullable=True))
    op.create_index('ix_chunks_notebook', 'chunks', ['notebook_id'])
    op.create_index('ix_chunks_tsv', 'chunks', ['tsv'], postgresql_using='gin')


def downgrade() -> None:
    op.drop_index('ix_chunks_tsv', 'chunks')
    op.drop_index('ix_chunks_notebook', 'chunks')
    for c in ('tsv', 'chunker', 'page_end', 'page_start', 'section', 'notebook_id'):
        op.drop_column('chunks', c)
    op.drop_constraint('uq_sources_notebook_hash', 'sources')
    op.drop_index('ix_sources_notebook', 'sources')
    for c in ('page_offsets', 'title', 'kind', 'notebook_id'):
        op.drop_column('sources', c)
    op.alter_column('sources', 'job_id', existing_type=sa.Integer(), nullable=False)
    op.drop_table('notebooks')
    op.drop_table('users')
