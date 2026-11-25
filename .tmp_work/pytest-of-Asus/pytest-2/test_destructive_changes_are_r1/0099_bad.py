def upgrade():
    op.execute("ALTER TABLE customers DROP COLUMN email")


def downgrade():
    pass
