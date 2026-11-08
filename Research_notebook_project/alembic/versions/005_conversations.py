"""conversations + messages for grounded notebook chat

Additive only. Chat LLM calls are ledgered under scope ("conversation", id).

Revision ID: 005_conversations
Revises: 004_notebooks_hybrid_retrieval
"""
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = '005_conversations'
down_revision: str | None = '004_notebooks_hybrid_retrieval'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        'conversations',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('notebook_id', sa.Integer(), sa.ForeignKey('notebooks.id'), nullable=False, index=True),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id'), nullable=False, index=True),
        sa.Column('max_tokens', sa.Integer(), nullable=False, server_default='60000'),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_table(
        'messages',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('conversation_id', sa.Integer(), sa.ForeignKey('conversations.id'), nullable=False, index=True),
        sa.Column('role', sa.String(10), nullable=False),
        sa.Column('content', sa.Text(), nullable=False, server_default=''),
        sa.Column('standalone_question', sa.Text(), nullable=True),
        sa.Column('citations', postgresql.JSONB(), nullable=True),
        sa.Column('dropped', postgresql.JSONB(), nullable=True),
        sa.Column('abstained', sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column('abstain_reason', sa.String(40), nullable=False, server_default=''),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table('messages')
    op.drop_table('conversations')
