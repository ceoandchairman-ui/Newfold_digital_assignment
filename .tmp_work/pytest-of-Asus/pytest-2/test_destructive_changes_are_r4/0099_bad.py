def upgrade():
    op.execute("ALTER TABLE customers ALTER COLUMN email SET NOT NULL")


def downgrade():
    pass
