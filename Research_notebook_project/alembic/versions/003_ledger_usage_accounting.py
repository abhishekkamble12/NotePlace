"""ledger_usage_accounting: separate reserved / estimated / provider-reported usage, per-attempt rows

Revision ID: 003_ledger_usage_accounting
Revises: 002_citation_by_reference
"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = '003_ledger_usage_accounting'
down_revision: str | None = '002_citation_by_reference'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Rows that exist before this migration are labelled usage_provenance='legacy': their tokens_in/out may be provider
# usage OR the old chars/4 estimate, and the two cannot be told apart. They keep usage_known=true so budget
# calculations still count what was recorded (conservative), but reports never call them provider-reported.


def upgrade() -> None:
    op.add_column('ledger', sa.Column('usage_known', sa.Boolean(), nullable=False, server_default=sa.true()))
    op.add_column('ledger', sa.Column('reserved_tokens', sa.Integer(), nullable=False, server_default='0'))
    op.add_column('ledger', sa.Column('est_tokens_in', sa.Integer(), nullable=False, server_default='0'))
    op.add_column('ledger', sa.Column('status', sa.String(10), nullable=False, server_default='ok'))
    op.add_column('ledger', sa.Column('provider', sa.String(30), nullable=False, server_default=''))
    op.add_column('ledger', sa.Column('call_id', sa.String(36), nullable=False, server_default=''))
    # existing rows get 'legacy'; the default is then dropped so every new row must state its provenance
    op.add_column('ledger', sa.Column('usage_provenance', sa.String(10), nullable=False, server_default='legacy'))
    op.alter_column('ledger', 'usage_provenance', server_default=None)
    op.create_check_constraint('ck_ledger_provenance_known', 'ledger', "(usage_provenance = 'unknown') = (NOT usage_known)")
    op.create_check_constraint('ck_ledger_provenance_values', 'ledger', "usage_provenance IN ('reported', 'unknown', 'legacy')")


def downgrade() -> None:
    # Reversible: drops only the added columns/constraints; original tokens_in/out values are untouched.
    op.drop_constraint('ck_ledger_provenance_values', 'ledger', type_='check')
    op.drop_constraint('ck_ledger_provenance_known', 'ledger', type_='check')
    for c in ('usage_provenance', 'call_id', 'provider', 'status', 'est_tokens_in', 'reserved_tokens', 'usage_known'):
        op.drop_column('ledger', c)
