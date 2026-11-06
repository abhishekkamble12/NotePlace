"""citation_by_reference: snapshot offsets on chunks and claims

Revision ID: 002_citation_by_reference
Revises: 001_initial_schema
"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = '002_citation_by_reference'
down_revision: str | None = '001_initial_schema'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column('chunks', sa.Column('start_char', sa.Integer(), nullable=True))
    op.add_column('chunks', sa.Column('end_char', sa.Integer(), nullable=True))
    op.add_column('claims', sa.Column('chunk_id', sa.Integer(), sa.ForeignKey('chunks.id'), nullable=True))
    op.add_column('claims', sa.Column('start_char', sa.Integer(), nullable=True))
    op.add_column('claims', sa.Column('end_char', sa.Integer(), nullable=True))


def downgrade() -> None:
    for c in ('end_char', 'start_char', 'chunk_id'):
        op.drop_column('claims', c)
    op.drop_column('chunks', 'end_char')
    op.drop_column('chunks', 'start_char')
