def upgrade():
    op.execute("ALTER TABLE bills ALTER COLUMN amount_due TYPE TEXT")


def downgrade():
    pass
