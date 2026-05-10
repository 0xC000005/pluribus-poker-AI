# Poker Autoresearch Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the missing executable pieces that make the poker autoresearch workflow safe to run unattended.

**Architecture:** Add a small testable Python module under `poker_ai/research/` and a thin CLI wrapper under `scripts/`. The module owns state initialization, readiness checks, gate execution, cycle creation/closure, STOP-file handling, JSON metrics, and research-log appends. The runner automates evaluation/logging only; it does not rewrite learning code or launch long Slumbot runs unless explicitly configured.

**Tech Stack:** Python stdlib (`argparse`, `json`, `subprocess`, `datetime`, `pathlib`), pytest, existing project unit tests.

---

### Task 1: Runner Unit Tests

**Files:**
- Create: `test/unit/test_poker_autoresearch.py`

- [ ] **Step 1: Write failing tests for state, readiness, gate, cycle, and log behavior.**
- [ ] **Step 2: Run `pytest -q test/unit/test_poker_autoresearch.py` and confirm it fails because the module does not exist.**

### Task 2: Core Autoresearch Module

**Files:**
- Create: `poker_ai/research/__init__.py`
- Create: `poker_ai/research/autoresearch.py`

- [ ] **Step 1: Implement minimal state initialization, readiness checks, command execution, cycle creation/closure, log append, and STOP-file detection.**
- [ ] **Step 2: Run `pytest -q test/unit/test_poker_autoresearch.py` and make it pass.**

### Task 3: CLI Wrapper And Local State Hygiene

**Files:**
- Create: `scripts/poker_autoresearch.py`
- Modify: `.gitignore`
- Create: `autoresearch-session/README.md`
- Create or modify: `RESEARCH_LOG.md`

- [ ] **Step 1: Add CLI subcommands for `init`, `status`, `gate`, `new-cycle`, `close-cycle`, and `continuous`.**
- [ ] **Step 2: Ignore generated local state while tracking a README for the state directory.**
- [ ] **Step 3: Add an initial research log entry explaining the workflow.**
- [ ] **Step 4: Run unit tests again.**

### Task 4: Workflow Documentation Update

**Files:**
- Modify: `docs/research_protocols/poker_autoresearch_workflow.md`

- [ ] **Step 1: Update the draft workflow with the implemented commands and approval status.**
- [ ] **Step 2: Include the safe startup command and stop-file behavior.**

### Task 5: Verification

**Files:**
- No new files.

- [ ] **Step 1: Run the new unit tests.**
- [ ] **Step 2: Run the Tier 0 gate through the new CLI.**
- [ ] **Step 3: Run `init`, `status`, `new-cycle`, `close-cycle`, and a bounded `continuous` dry run.**
- [ ] **Step 4: Inspect git status and confirm generated state is ignored except the README.**
