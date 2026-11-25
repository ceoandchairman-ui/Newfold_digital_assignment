from lib import save, sub

sub(
    "infra/terraform/network.tf",
    '    blob       = "privatelink.blob.core.windows.net"\n',
    '    blob       = "privatelink.blob.core.windows.net"\n    cosmos     = "privatelink.documents.azure.com"\n',
)

f = "scripts/helm_values_from_terraform.py"
sub(f, '    "servicebus_namespace",\n)', '    "servicebus_namespace",\n    "cosmos_endpoint",\n)')
sub(
    f,
    '            "UPLOADS_CONTAINER": _value(outputs, "uploads_container"),\n',
    '            "UPLOADS_CONTAINER": _value(outputs, "uploads_container"),\n            "COSMOS_ENDPOINT": _value(outputs, "cosmos_endpoint"),\n',
)

sub(
    "deploy/helm/support/values.yaml",
    '  USE_STUB_GATEWAY: "true"      # the real payment gateway is still an open decision\n',
    '  USE_STUB_GATEWAY: "true"      # the real payment gateway is still an open decision\n  # The metadata registry (Cosmos DB). COSMOS_ENDPOINT comes from `endpoints`; with this set, a\n  # service without it refuses to start instead of silently skipping the release checks.\n  REGISTRY_REQUIRED: "true"\n',
)

# Every workload must compute the same search release, so each gets the same embedding dimension.
sub(
    "deploy/helm/support/templates/services.yaml",
    "            # Semantic-convention resource attributes",
    "            # Part of the search release (releases.py): must match the publisher's value.\n            - { name: EMBEDDING_DIM, value: {{ $.Values.embeddingDim | quote }} }\n            # Semantic-convention resource attributes",
)
sub(
    "deploy/helm/support/templates/indexer.yaml",
    '                - { name: DATABASE_ENTRA_USER, value: "id-{{ .Values.identityPrefix }}-indexer" }\n',
    '                - { name: DATABASE_ENTRA_USER, value: "id-{{ .Values.identityPrefix }}-indexer" }\n                - { name: EMBEDDING_DIM, value: {{ .Values.embeddingDim | quote }} }\n',
)

f = "deploy/helm/support/templates/db-bootstrap.yaml"
sub(
    f,
    "python -m support_assistant.dbroles{{ if",
    "python -m support_assistant.dbroles && python -m support_assistant.releases publish{{ if",
)
sub(
    f,
    "# Pre-deploy hook: migrations, then a Postgres role per workload identity with least-privilege\n",
    "# The hook also publishes the shared-settings releases to the metadata registry (releases.py),\n# so they exist before any new pod checks for them.\n#\n# Pre-deploy hook: migrations, then a Postgres role per workload identity with least-privilege\n",
)
sub(
    f,
    "            {{- range $k, $v := .Values.commonEnv }}\n            - { name: {{ $k }}, value: {{ $v | quote }} }\n            {{- end }}\n",
    "            {{- range $k, $v := .Values.commonEnv }}\n            - { name: {{ $k }}, value: {{ $v | quote }} }\n            {{- end }}\n            {{- range $k, $v := .Values.endpoints }}\n            - { name: {{ $k }}, value: {{ $v | quote }} }\n            {{- end }}\n",
)

sub(
    ".env.example",
    "# Cosine similarity a vector match needs",
    """# The metadata registry (Azure Cosmos DB, Entra auth only: there is no key to set). Unset = no
# registry. Production sets REGISTRY_REQUIRED=true.
COSMOS_ENDPOINT=
COSMOS_DATABASE=registry
REGISTRY_REQUIRED=false
# Raise to embed every document again with the same model name.
EMBEDDING_VERSION=1
# Cosine similarity a vector match needs""",
)

f = "tests/test_helm_values_script.py"
sub(
    f,
    '        "servicebus_namespace": {"value": "sb-support-dev-abc12"},\n',
    '        "servicebus_namespace": {"value": "sb-support-dev-abc12"},\n        "cosmos_endpoint": {"value": "https://cosmos-x.documents.azure.com:443/"},\n',
)
sub(
    f,
    '        "UPLOADS_CONTAINER": "customer-uploads",\n    }\n',
    '        "UPLOADS_CONTAINER": "customer-uploads",\n        "COSMOS_ENDPOINT": "https://cosmos-x.documents.azure.com:443/",\n    }\n',
)
sub(f, '        "servicebus_namespace",\n    ],\n)', '        "servicebus_namespace",\n        "cosmos_endpoint",\n    ],\n)')
save()
