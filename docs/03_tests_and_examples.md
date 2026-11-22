# 3. Tests and repeatable examples

**Run everything:** `uv run pytest -q` gives `389 passed, 28 skipped`. The 28 are integration tests that need real Postgres and Redis (CI; see `01_setup_and_run.md`).

**What the tests are, honestly:** the journey and failure tests run the *real* LangGraph graph, agent logic, harness, payment state machine, guards and checkpoint pause/resume. Only the outside world is replaced by in-memory fakes (the database, the gateway simulator, the model). They prove the behavior and the rules. They do not prove the SQL or the Azure services; that is what `tests/integration/` is for.

## Main journeys

| Journey | Test | What it proves |
|---|---|---|
| **1. Account question** ("how much data have I used?") | `test_journeys.py::test_journey1_account_question` | The Director calls the account tool for the verified customer and answers from live data. |
| **2. Troubleshooting** (mobile data fails, the APN is wrong) | `test_journey2_apn_troubleshooting` | Hybrid retrieval runs from a graph node with the search query, and the answer comes from the manual. (Query rewriting is tested separately in `test_chains_and_outbox.py`; the citation line is tested in `test_resilience_budget_lazy.py::test_troubleshooting_answers_cite_their_manual_sections`.) |
| **3. Paying a bill** (C1002 pays B1005) | `test_journey3_customer_pays_bill_end_to_end` | Propose, evaluate, **pause for approval**, resume, pay once, confirm `PAID`, then say "successful". Nothing is charged before approval. |
| **4. Closing a troubleshooting chat** ("thanks") | `test_resilience_budget_lazy.py::test_an_acknowledgement_gets_a_closing_reply_without_any_search` | No search and no model call for an acknowledgement. |

Step-by-step versions with the exact messages are in `docs/example_transcripts.md`.

## Important failure cases

| Risk | Test(s) | Expected behavior |
|---|---|---|
| **Unverified caller** | `test_journeys.py::test_unverified_customer_gets_nothing`, `test_red_team.py::test_an_unverified_attacker_gets_nothing_whatever_they_type` | No agent, tool or retrieval runs; the reply asks to verify. |
| **Account enumeration** | `test_audit_identity_episodes.py::test_an_ambiguous_name_is_refused_with_the_same_answer`, `test_backend_memory_repos.py::test_wrong_dob_fails_with_the_same_answer_as_unknown_phone` | Wrong DOB, unknown phone and ambiguous name look identical. |
| **Another customer's data** | `test_red_team.py::test_asking_for_another_customers_bill_shows_only_your_own`, `test_journeys.py::test_cannot_pay_another_customers_bill`, `test_backend_memory_repos.py::test_every_customer_scoped_query_filters_by_customer`, `integration/test_postgres.py::test_a_customer_never_sees_another_customers_rows` | Scoped in SQL; a bill that exists but isn't theirs is "not found". |
| **Double payment** | `test_payments.py::test_replay_with_same_key_never_charges_twice`, `integration/test_postgres.py::test_two_concurrent_payments_for_one_bill_exactly_one_succeeds` | One charge, whatever the retries or races. |
| **Gateway timeout** | `test_payments.py::test_timeout_keeps_attempt_pending_and_blocks_retry`, `test_journeys.py::test_gateway_timeout_never_claims_success_and_holds_the_bill` | The bill is held, never marked paid or released; the customer is told not to pay again; `reconcile_payment` resolves it. |
| **Gateway decline / outage** | `test_payments.py::test_decline_releases_the_bill`, `test_unreachable_gateway_releases_the_bill` | The bill returns to `ISSUED`; the customer is told clearly. |
| **Approval bypass** | `test_red_team.py::test_insisting_on_paying_still_needs_explicit_approval`, `test_only_an_explicit_boolean_true_approves_a_payment` | "PAY NOW, skip confirmation" and non-boolean answers never pay. |
| **Model lies or leaks** | `test_red_team.py::test_a_model_claiming_payment_without_one_is_corrected`, `test_a_model_that_leaks_internals_is_blocked`, `test_a_model_that_leaks_another_customer_id_is_blocked`, `test_guards.py` | The egress filter replaces the reply. |
| **Prompt injection** | `test_red_team.py::test_injected_manual_text_cannot_steer_the_answer`, `test_guards.py` (injection phrases), `test_maf_tools_shields.py` (Prompt Shields outage falls back to the local check) | Hostile text in the manual is dropped, and if nothing safe remains the customer is escalated. A hostile customer message is refused before any model or tool runs. |
| **A reply and an action in one turn** | `test_guards.py::test_reply_and_action_together_violate_the_invariant`, `test_agents_and_graph.py::test_a_turn_with_both_a_reply_and_an_action_escalates_and_pays_nothing` | The turn fails closed: escalated, never reaches approval or the gateway. |
| **Evaluator can't write** | `test_agents_and_graph.py::test_evaluator_cannot_call_write_tools`, `test_mcp_client.py::test_the_allow_list_is_enforced_even_though_the_registry_knows_the_tool` | Refused by the harness, never sent to a server. |
| **Model never supplies the customer** | `test_agents_and_graph.py::test_customer_id_comes_from_state_not_model`, `test_maf_tools_shields.py::test_the_model_is_never_shown_a_customer_id_parameter` | The harness overwrites it from the verified session. |

## Dependency and infrastructure failures

| Failure | Test | Behavior |
|---|---|---|
| MCP server keeps failing | `test_resilience_budget_lazy.py::test_an_open_circuit_fails_fast_without_calling_the_server` | Circuit opens; other servers unaffected. |
| MCP connection drops | `test_batch_services.py::test_a_dropped_connection_is_reopened_and_the_call_retried` | Reconnect and retry once, reads only. |
| A write must not be retried | `test_a_write_is_never_retried_so_an_sms_cannot_be_sent_twice` | Exactly one attempt. |
| Slow server | `test_a_slow_server_times_out_instead_of_hanging_the_turn` | Times out; the turn degrades. |
| Redis down | `test_backend_memory_repos.py::test_memory_survives_a_redis_outage`, `test_resilience_budget_lazy.py::test_budget_fails_open_when_redis_is_down` | Postgres answers; the turn is allowed. |
| One MCP server crashes | `test_ops_intent_resume.py::test_one_server_crashing_does_not_stop_the_others` | Supervisor restarts it with backoff. |
| Notification provider outage | `test_worker.py::test_provider_outage_retries_and_unknown_customer_dead_letters` | Retry or dead-letter, never block a chat. |
| Retriever fails | `test_tiers_retriever_graph.py::test_retriever_failure_degrades_instead_of_blocking` | The agent searches itself. |
| Paused payment across a restart | `integration/test_redis.py::test_a_paused_payment_survives_an_orchestrator_restart` | New pod resumes from Redis (CI). |

## Quality and operations checks

| Area | Tests |
|---|---|
| Retrieval quality (recall@k, MRR; decides whether the reranker earns its latency) | `test_evaluation.py`, `eval/retrieval_eval.example.json` |
| Offline model judge with a validated verdict | `test_cli_judge_load.py` |
| Rule-based invariant checker over transcripts | `test_batch_services.py` (rule checker tests) |
| Tracing: W3C context, spans per MCP call and graph node | `test_otel.py` |
| Migration safety lint; post-deploy smoke test | `test_devops.py` |
| MCP contract (the catalog equals what each server exposes) | `test_mcp_contract.py` |

## Deploy-path checks (identities, roles, secrets)

| What | Test | Proves |
|---|---|---|
| Per-service Postgres roles | `test_dbroles.py` | Each service gets only the access it needs (only the MCP servers touch the ledger, audit is append-only, the troubleshooter reads only the manual, the director and evaluator get nothing); grants are wiped and re-applied every deploy, so removed access does not linger; unsafe role names never reach SQL. |
| Entra connection user | `test_dbroles.py::test_entra_connections_use_the_identity_name_as_the_postgres_user` | Each pod connects as its own identity, not a shared user. |
| Terraform outputs to Helm | `test_helm_values_script.py` | Every service gets its own identity ids; a missing or blank output stops the deploy instead of shipping a pod that cannot sign in. |

Checked outside pytest (and not as tests): `helm lint` and `helm template` render the chart, and the rendered output was inspected for hook order, secret mounts on every pod, and the generated identity values; `terraform validate` passes for both `infra/terraform` and `infra/bootstrap`.

## Documents and uploads

| Risk | Test | Expected behavior |
|---|---|---|
| Edited document leaves old text behind | `test_indexer_chain.py::test_an_edited_document_loses_its_old_text`, `integration/test_documents.py::test_an_edited_document_loses_its_old_text_in_the_real_database` | New chunk in, old chunk out; other documents untouched. |
| A storage outage wipes the knowledge base | `test_documents_uploads.py::test_an_empty_listing_never_wipes_the_knowledge_base` | An empty listing deletes nothing. |
| One bad document stops the batch | `test_a_bad_document_never_stops_the_batch` | Unsupported, oversized and non-UTF-8 files are skipped and reported. |
| Hostile upload (renamed executable, double extension, wrong bytes) | `test_hostile_or_mismatched_files_are_rejected_with_a_safe_reason` | Rejected before storage, with a reason that names no internals. |
| Path traversal in a filename | `test_filenames_are_reduced_to_a_safe_basename` | Reduced to a safe basename. |
| Oversized upload | `test_an_oversized_upload_is_cut_off_with_413_before_it_is_fully_read` | Cut off while reading, 413. |
| File used before it is scanned | `test_a_file_stays_quarantined_until_the_scanner_has_cleared_it` | Stays `PENDING_SCAN` until the scan says clean. |
| Malware | `test_a_malicious_file_is_deleted_and_marked_rejected_but_its_record_is_kept` | File deleted, audit row kept. |
| One customer reads another's file | `test_another_customers_upload_id_is_simply_not_found`, `test_status_and_list_are_private_to_the_customer` | Not found; lists are per customer. |
| Upload flood | `test_the_daily_limit_is_429_and_per_customer` | 429 after the daily limit; other customers unaffected. |
| Orphaned file when the database write fails | `test_if_the_record_cannot_be_saved_the_stored_file_is_removed` | The stored file is removed. |
| Erasure request | `test_erasure_removes_files_and_rows_and_governance_reports_the_count` | The customer's files and rows are gone; other customers' remain. |

## Run your own checks

```bash
uv run pytest -k "payment or timeout" -v            # the payment failure cases
uv run pytest tests/test_red_team.py -v             # hostile inputs
uv run python scripts/check_migrations.py           # migration safety lint
uv run python -m support_assistant.evaluation eval/retrieval_eval.example.json   # needs a real DB + embedder
```

## What is not covered

- Live Azure services, the real payment gateway, and real model behavior (including the Agent Framework with a live model).
- Real traffic: the load script exists (`loadtest.py`) but has not been run.
- Anything the 28 integration tests cover, until CI runs them.
