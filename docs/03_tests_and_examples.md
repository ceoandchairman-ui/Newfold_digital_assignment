# 3. Tests and repeatable examples

**Run everything:** `uv run pytest -q` gives `307 passed, 18 skipped`. The 18 are integration tests that need real Postgres and Redis (CI; see `01_setup_and_run.md`).

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
- Anything the 18 integration tests cover, until CI runs them.
