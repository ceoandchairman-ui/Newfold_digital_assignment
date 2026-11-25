def upgrade():
    op.execute("DROP TABLE bills")


def downgrade():
    pass
