# 3. Tests and repeatable examples

**Run everything:** `uv run pytest -q` gives `535 passed, 28 skipped`. The 28 are integration tests that need real Postgres and Redis (CI; see `01_setup_and_run.md`).

**What the tests are, honestly:** the journey and failure tests run the *real* LangGraph graph (the three agent subgraphs and the code around them), agent logic, harness, payment state machine, guards and checkpoint pause/resume. Only the outside world is replaced: in-memory fakes for the database, the gateway simulator and the model, and mock HTTP transports for Twilio and Stripe. They prove the behavior and the rules. They do not prove the SQL, the Azure services, or that a real provider accepts our requests; that is what `tests/integration/` and the first deployment are for.

## Main journeys

| Journey | Test | What it proves |
|---|---|---|
| **1. Account question** ("how much data have I used?") | `test_journeys.py::test_journey1_account_question`, `test_support_agent.py::test_keyword_router_separates_faults_from_account_questions` | The router sends it to `billing_gen`, which reads the account for the verified customer and answers from live data. "Data" alone is not treated as a fault. |
| **2. Troubleshooting** (mobile data fails, the APN is wrong) | `test_journey2_apn_troubleshooting`, `test_troubleshooter_tools.py::test_it_checks_the_device_first_then_answers_with_what_it_found` | Hybrid retrieval runs from a graph node, the troubleshooter checks the device, and the answer comes from the manual with a source line. (Query rewriting is tested in `test_chains_and_outbox.py`.) |
| **3. Paying a bill** (C1002 pays B1005) | `test_journey3_customer_pays_bill_end_to_end` | Propose, evaluate, **pause for approval**, resume, pay once, confirm `PAID`, then say "successful". Nothing is charged before approval. |
| **4. Closing a troubleshooting chat** ("thanks") | `test_resilience_budget_lazy.py::test_an_acknowledgement_gets_a_closing_reply_without_any_search` | No search and no model call for an acknowledgement. |
| **5. A reasoning question** ("why was I charged?") | `test_support_agent.py::test_instruct_reasons_over_account_facts_and_wrapped_policy_and_cites_it`, `test_instruct_also_reasons_over_the_contract_past_cases_and_recent_chats` | `instruct_gen` reasons over the account, contract, past cases, recent chats and the retrieved policy, and cites the source. |

Step-by-step versions with the exact messages are in `docs/example_transcripts.md`.

## The graph and the three agents

| What | Test | Expected behavior |
|---|---|---|
| The main graph is exactly three agents plus the gates | `test_graph_topology.py::test_the_main_graph_has_exactly_the_three_agents_and_the_code_around_them` | Eight nodes and a fixed set of edges; any other wiring fails. |
| The support agent is one agent of five nodes | `test_the_support_agent_is_one_agent_made_of_a_router_and_three_generators`, `test_the_troubleshoot_agent_retrieves_then_diagnoses` | Router plus three generators, and `retrieve` then `diagnose`. No generator calls another. |
| No MCP hub node | `test_there_is_no_mcp_hub_node_the_one_write_is_the_pay_step` | The one graph-side write is `pay`. |
| Each intent reaches only its own branch | `test_normal_goes_to_the_simple_generator_only`, `test_instruct_retrieves_policy_then_reasons`, `test_billing_goes_to_the_task_generator_and_never_retrieves`, `test_troubleshoot_retrieves_then_the_eval_agent_checks_the_reply` | The recorded agent calls match the branch exactly. |
| A router that is down or unsure | `test_when_the_router_is_down_the_backend_hint_decides_and_unsure_means_normal`, `test_support_agent.py::test_bad_or_unsure_model_output_falls_back_to_keywords` | Keyword hint, then the generic answer. The turn is never lost. |
| A reply that fails the eval agent | `test_a_reply_that_fails_the_eval_agent_is_escalated_not_shown` | Escalated, never shown. |
| The unverified path | `test_the_unverified_path_reaches_no_agent_tool_or_retriever` | `deny` leads only to `egress`. |

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
| **Prompt injection** | `test_red_team.py::test_injected_manual_text_cannot_steer_the_answer`, `test_support_agent.py::test_instruct_drops_policy_text_that_tries_to_give_the_model_orders`, `test_guards.py`, `test_prompt_shields.py` (outage falls back to the local check) | Hostile text in the manual is dropped, and if nothing safe remains the customer is escalated. A hostile customer message is refused before any model or tool runs. |
| **A reply and an action in one turn** | `test_guards.py::test_reply_and_action_together_violate_the_invariant`, `test_agents_and_graph.py::test_a_turn_with_both_a_reply_and_an_action_escalates_and_pays_nothing` | The turn fails closed: escalated, never reaches approval or the gateway. |
| **Evaluator can't write** | `test_agents_and_graph.py::test_evaluator_cannot_call_write_tools`, `test_mcp_client.py::test_the_allow_list_is_enforced_even_though_the_registry_knows_the_tool` | Refused by the harness, never sent to a server. |
| **Model never supplies the customer** | `test_agents_and_graph.py::test_customer_id_comes_from_state_not_model`, `test_mcp_contract.py::test_every_tool_takes_a_required_customer_id` | The harness overwrites it from the verified session. |
| **No agent can call a write tool** | `test_mcp_account_crm.py::test_every_catalog_tool_is_allowed_to_an_agent_or_is_a_write_no_agent_may_call` | Every tool is either allowed to an agent (a read) or is exactly one of the writes no agent may call. |

## The LLM judge and the eval agent

| Risk | Test | Expected behavior |
|---|---|---|
| The judge blocks a turn by failing | `test_judge.py::test_a_broken_judge_never_blocks_a_reply`, `test_an_outage_or_a_timeout_means_no_objection` | An outage, a timeout or garbage means no objection. |
| An unsure judge rejects a good reply | `test_unclear_or_unsure_answers_never_object` | Only a confident "unsupported" objects. |
| An unsupported reply is shown | `test_an_unsupported_reply_is_rejected_by_the_eval_agent` | Escalated, not shown. |
| The judge costs a call when rules already failed | `test_the_rules_run_first_so_a_reply_without_a_citation_needs_no_judge` | No model call. |
| The judge checks the wrong things | `test_the_judge_is_skipped_when_there_is_nothing_to_check`, `test_a_payment_verdict_never_involves_the_judge` | Skipped for non-troubleshooting replies, for no sources and for the fixed closing text. A payment verdict never uses it. |
| Hostile text in a source steers the judge | `test_the_sources_reach_the_judge_as_data_and_the_citation_line_is_removed` | Sources are wrapped as data. |

## Troubleshooter device checks

| Risk | Test | Expected behavior |
|---|---|---|
| A runaway loop | `test_troubleshooter_tools.py::test_the_loop_is_bounded_and_never_repeats_the_same_action` | At most three actions, never the same one twice. |
| A locked SIM is retried | `test_a_locked_sim_escalates_at_once_and_runs_nothing_more` | Escalated at once; no reboot, no retry, no wording call. |
| The model touches someone else's device | `test_a_device_that_is_not_the_customers_is_never_touched` | Refused before any call. |
| The model picks a non-device tool | `test_only_device_tools_can_be_chosen_never_a_payment_or_a_message` | Payments, messages and reads are never chosen. |
| A toggle with no on or off | `test_a_fix_with_a_toggle_needs_an_explicit_on_or_off` | Refused. |
| Bad model output or a failing tool | `test_garbage_from_the_model_means_no_tool_and_the_answer_still_comes`, `test_a_failing_tool_ends_the_loop_but_not_the_turn` | The answer is written from the manual alone. |
| The loop is turned off | `test_with_the_loop_turned_off_no_device_tool_is_called_at_all` | No device tool is called. |

## Data contracts (MCP)

| Area | Test | Expected behavior |
|---|---|---|
| The catalog equals what each of the nine servers exposes | `test_mcp_contract.py::test_every_server_exposes_exactly_the_tools_the_catalog_lists` | A tool added to a server or the catalog alone fails. |
| Profile, devices, contract | `test_mcp_account_crm.py::test_the_profile_masks_the_email_and_never_returns_a_date_of_birth`, `test_devices_expose_only_the_last_four_of_the_imei`, `test_the_contract_flags_an_expired_line` | Masked, minimal, correct. |
| Case history and recent chats | `test_case_history_is_capped_and_returns_no_embeddings`, `test_recent_interactions_default_to_five_days_and_clamp_the_window` | Capped; an agent cannot pull the whole history. |

## Providers (Twilio, Stripe, Gmail, Drive) and the chat channel

| Risk | Test | Expected behavior |
|---|---|---|
| A model chooses the SMS text or number | `test_providers.py::test_sms_text_comes_from_a_template_and_the_number_from_the_customer_record`, `test_an_unknown_template_sends_nothing` | Template only; the number comes from the customer record; the tool has no way to take either. |
| Someone reads another customer's message status | `test_message_status_is_only_visible_for_the_customers_own_number` | Looks like "no such message". |
| A model chooses an email recipient or body | `test_email_goes_to_the_address_on_file_with_template_text`, `test_a_customer_with_no_email_gets_a_clean_error_and_nothing_is_sent` | Template only, to the address on file. |
| Another customer's payment status | `test_payment_status_is_hidden_from_everyone_but_its_owner` | Not found. |
| Drive text steers the agent | `test_drive_documents_are_marked_untrusted_and_cut_to_size` | Returned as untrusted text, cut to a fixed size. |
| Stripe double charge or a guessed outcome | `test_a_succeeded_intent_is_approved_and_sent_with_the_idempotency_key_and_cents`, `test_an_unsettled_answer_leaves_the_payment_pending_not_failed`, `test_a_timeout_is_unknown_and_a_failure_to_connect_is_a_clean_failure` | Idempotency key sent; processing, 5xx, 429 and timeouts stay `PENDING`; only a failure to connect releases the bill. |
| A real charge with no payment-method mapping | `test_the_customer_to_payment_method_mapping_is_a_placeholder_that_refuses` | Refuses rather than guessing. |
| A real provider used by accident | `test_twilio_is_a_stub_until_it_is_switched_on_and_fully_configured`, `test_the_gateway_is_the_stub_unless_stripe_is_switched_on_and_has_a_key`, `test_the_google_providers_are_placeholders_that_refuse_until_they_are_written` | The stub runs until the switch is off and every credential exists. |
| A forged or replayed Twilio request | `test_a_request_without_a_valid_signature_never_reaches_the_pipeline`, `test_anything_else_is_refused`, `test_a_repeated_delivery_of_the_same_message_is_answered_only_once` | Refused (403), or answered once. |
| The channel open by mistake | `test_the_channel_is_off_by_default`, `test_enabled_without_a_token_is_refused_not_open` | 404 off; 503 if enabled but unconfigured. |
| A phone number taken as proof of identity | `test_an_unverified_sender_is_asked_for_a_date_of_birth_and_nothing_else_runs`, `test_a_wrong_date_of_birth_is_refused_and_the_right_one_verifies` | Verification is still required. |
| Markup injected through a reply | `test_a_verified_sender_gets_the_answer_escaped_as_xml` | Escaped. |
| A payment confirmed by text | `test_yes_and_no_answer_a_pending_payment_confirmation` | YES and NO answer the pending question. |

## History and models

| What | Test | Expected behavior |
|---|---|---|
| The last five days of chats | `test_history.py::test_the_first_turn_carries_a_note_about_the_last_days_conversations`, `test_the_current_session_is_not_repeated_back_to_itself`, `test_later_turns_do_not_reload_the_history`, `test_history_can_be_turned_off` | First turn only, other sessions only, capped, can be switched off. |
| History must not cost the turn | `test_a_database_error_never_costs_the_turn` | The turn proceeds without the note. |
| One model per job | `test_models.py::test_every_role_has_its_own_deployment_by_default`, `test_a_model_swap_is_configuration_only`, `test_the_support_agent_has_two_models_chat_and_planner` | Five distinct deployments; a swap is configuration. |

## Dependency and infrastructure failures

| Failure | Test | Behavior |
|---|---|---|
| MCP server keeps failing | `test_resilience_budget_lazy.py::test_an_open_circuit_fails_fast_without_calling_the_server` | Circuit opens; other servers unaffected. |
| MCP connection drops | `test_batch_services.py::test_a_dropped_connection_is_reopened_and_the_call_retried` | Reconnect and retry once, reads only. |
| A write must not be retried | `test_a_write_is_never_retried_so_an_sms_cannot_be_sent_twice`, `test_payments_and_every_outbound_message_are_write_tools` | Exactly one attempt for every payment and outbound message tool. |
| Slow server | `test_a_slow_server_times_out_instead_of_hanging_the_turn` | Times out; the turn degrades. |
| Redis down | `test_backend_memory_repos.py::test_memory_survives_a_redis_outage`, `test_resilience_budget_lazy.py::test_budget_fails_open_when_redis_is_down` | Postgres answers; the turn is allowed. |
| One MCP server crashes | `test_ops_intent_resume.py::test_one_server_crashing_does_not_stop_the_others` | Supervisor restarts it with backoff. |
| Notification provider outage | `test_worker.py::test_provider_outage_retries_and_unknown_customer_dead_letters`, `test_providers.py::test_the_worker_sends_sms_through_twilio_only_when_asked` | Retry or dead-letter, never block a chat. SMS goes through Azure Communication Services unless Twilio is chosen. |
| Retriever fails | `test_retriever_graph.py::test_retriever_failure_degrades_instead_of_blocking` | The agent searches itself. |
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
| Per-service Postgres roles | `test_dbroles.py`, `test_dbroles_documents.py` | Each service gets only the access it needs (only the MCP servers touch the ledger, audit is append-only, the troubleshooter reads only the manual, the director and evaluator get nothing, the `index-worker` has exactly the indexer's rights); grants are wiped and re-applied every deploy; unsafe role names never reach SQL. |
| Entra connection user | `test_dbroles.py::test_entra_connections_use_the_identity_name_as_the_postgres_user` | Each pod connects as its own identity, not a shared user. |
| Terraform outputs to Helm | `test_helm_values_script.py` | Every service gets its own identity ids; a missing or blank output stops the deploy. |

Checked outside pytest (and not as tests): `helm lint` and `helm template` render the chart, and `terraform validate` passes for `infra/terraform` and `infra/bootstrap`. The three Mermaid diagrams in `docs/diagrams/` were parsed with the Mermaid parser.

## Documents and uploads

| Risk | Test | Expected behavior |
|---|---|---|
| A new or edited document is not noticed | `test_index_worker.py::test_an_upload_is_indexed_after_it_settles_and_not_before`, `test_an_edit_is_noticed_because_the_etag_changes_even_if_the_name_does_not`, `test_a_deleted_document_triggers_a_run_so_its_chunks_are_removed` | Indexed about a minute after the listing settles. |
| A multi-file upload indexes many times | `test_many_files_uploaded_over_several_looks_cause_one_run` | One run, after the upload finishes. |
| The worker and the CronJob collide | `test_only_one_run_holds_the_indexing_slot_and_it_is_released`, `test_when_the_cronjob_holds_the_slot_the_worker_waits_and_runs_later` | One at a time; the lock is released even on failure. |
| A failed run is forgotten | `test_a_failed_run_is_retried_on_the_next_look_and_nothing_is_marked_done`, `test_a_storage_listing_error_changes_nothing` | Retried; nothing marked done. |
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
uv run pytest tests/test_providers.py -v            # Twilio, Stripe, Gmail, Drive, the chat channel
uv run python scripts/check_migrations.py           # migration safety lint
uv run python -m support_assistant.evaluation eval/retrieval_eval.example.json   # needs a real DB + embedder
```

## What is not covered

- Live Azure services, a real model (including the Agent Framework, reasoning effort and the API versions), and the model quota.
- A real Twilio, Stripe or Google account: the clients are tested against mock HTTP only, and the Gmail and Drive providers are not written.
- Real traffic: the load script exists (`loadtest.py`) but has not been run.
- Anything the 28 integration tests cover, until CI runs them.
