"""The supplied telecom data (data/corpus/db.toml) and policy: all six bill statuses, and the
fields our guessed schema did not have. The customer's postal address is deliberately not stored:
no journey needs it, so the system does not hold it.

Expand only: new columns are nullable or have defaults. A release from before this migration
rejects a bill in one of the new statuses (it fails closed and escalates).

Revision ID: 0007
Revises: 0006
"""

from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE bills DROP CONSTRAINT IF EXISTS bills_status_check;
        ALTER TABLE bills ADD CONSTRAINT bills_status_check CHECK (status IN
            ('DRAFT', 'ISSUED', 'OVERDUE', 'AWAITING_PAYMENT', 'PAID', 'DISPUTED'));
        ALTER TABLE bills
            ADD COLUMN IF NOT EXISTS period_start DATE,
            ADD COLUMN IF NOT EXISTS period_end DATE,
            ADD COLUMN IF NOT EXISTS issue_date DATE;

        ALTER TABLE customers
            ADD COLUMN IF NOT EXISTS account_status TEXT NOT NULL DEFAULT 'ACTIVE',
            ADD COLUMN IF NOT EXISTS last_extension_date DATE,
            ADD COLUMN IF NOT EXISTS goodwill_credit_used_this_year NUMERIC(12,2) NOT NULL DEFAULT 0;

        ALTER TABLE plans
            ADD COLUMN IF NOT EXISTS refuel_price_per_gb NUMERIC(12,2);

        ALTER TABLE lines
            ADD COLUMN IF NOT EXISTS data_refueling_mb INTEGER NOT NULL DEFAULT 0,
            ADD COLUMN IF NOT EXISTS roaming_enabled BOOLEAN NOT NULL DEFAULT false,
            ADD COLUMN IF NOT EXISTS last_plan_change_date DATE,
            ADD COLUMN IF NOT EXISTS last_sim_replacement_date DATE,
            ADD COLUMN IF NOT EXISTS suspension_start_date DATE;

        ALTER TABLE payment_methods
            ADD COLUMN IF NOT EXISTS expiration TEXT;

        ALTER TABLE devices
            ADD COLUMN IF NOT EXISTS device_type TEXT,
            ADD COLUMN IF NOT EXISTS esim_capable BOOLEAN,
            ADD COLUMN IF NOT EXISTS activated BOOLEAN,
            ADD COLUMN IF NOT EXISTS activation_date TIMESTAMP,
            ADD COLUMN IF NOT EXISTS last_esim_transfer_date TIMESTAMP;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        ALTER TABLE devices DROP COLUMN IF EXISTS device_type, DROP COLUMN IF EXISTS esim_capable,
            DROP COLUMN IF EXISTS activated, DROP COLUMN IF EXISTS activation_date,
            DROP COLUMN IF EXISTS last_esim_transfer_date;
        ALTER TABLE payment_methods DROP COLUMN IF EXISTS expiration;
        ALTER TABLE lines DROP COLUMN IF EXISTS data_refueling_mb,
            DROP COLUMN IF EXISTS roaming_enabled, DROP COLUMN IF EXISTS last_plan_change_date,
            DROP COLUMN IF EXISTS last_sim_replacement_date,
            DROP COLUMN IF EXISTS suspension_start_date;
        ALTER TABLE plans DROP COLUMN IF EXISTS refuel_price_per_gb;
        ALTER TABLE customers DROP COLUMN IF EXISTS account_status,
            DROP COLUMN IF EXISTS last_extension_date,
            DROP COLUMN IF EXISTS goodwill_credit_used_this_year;
        ALTER TABLE bills DROP COLUMN IF EXISTS period_start, DROP COLUMN IF EXISTS period_end,
            DROP COLUMN IF EXISTS issue_date;
        ALTER TABLE bills DROP CONSTRAINT IF EXISTS bills_status_check;
        ALTER TABLE bills ADD CONSTRAINT bills_status_check
            CHECK (status IN ('ISSUED', 'AWAITING_PAYMENT', 'PAID'));
        """
    )
