# Research Synchronization via Notion

This document defines the Notion-based synchronization workflow between
Claude.ai (research designer) and Claude Code (implementer).

**Read this file at the start of every session.**

---

## Notion Page Structure & Access

```
📂 논문연구 (331cfafa-6fc4-8117-a85f-e79eab941336)
└── 📋 [논문1] Resource-Elastic NPU Scheduling (331cfafa-6fc4-81ea-bd67-d563ea47af90) ← Hub
    │
    ├── 📂 연구 (332cfafa-6fc4-81b6-92ec-f4635444662c)
    │   │  Claude.ai: ✍️ Write / Claude Code: 📖 Read-only
    │   ├── 1. Introduction (332cfafa-6fc4-8127-81a2-f46ca8b6309c)
    │   ├── 2. Background (332cfafa-6fc4-810e-8642-c374c6065648)
    │   ├── 3. Related Work (332cfafa-6fc4-81df-bef8-d404d99e65bc)
    │   ├── 4. Methodology (331cfafa-6fc4-81c2-972e-d565b565882e) ← Primary reference
    │   ├── 5. Evaluation (331cfafa-6fc4-8159-96ad-ea07cebeafc8)
    │   └── 6. Discussion & Future Work (332cfafa-6fc4-81dc-a22c-c87776b3437b)
    │
    ├── 📂 구현 (332cfafa-6fc4-81c8-8233-d36b4756587f)
    │   │  Claude.ai: ✍️ Request & Summarize / Claude Code: ✍️ Results (구현 태스크 DB only)
    │   ├── 구현 태스크 (DB: 918d34e1b12c4d7faa6135282f0f9581, data_source: 63378bd3-1d1a-4a0f-966b-d4087c0d1f13)
    │   ├── 구현 현황 (331cfafa-6fc4-8107-bfad-d8d65baf1fce)
    │   └── 실험 결과 (331cfafa-6fc4-8100-a6b5-fae2bd8cb935)
    │
    └── 📂 관리 (332cfafa-6fc4-8145-9b67-ec9987e3ad7d)
        │  Claude.ai: ✍️ Write
        ├── 논문 기여 & 타겟 학회 (331cfafa-6fc4-81cd-bc6d-f1a0c52c863b)
        ├── 리뷰어 대응 전략 (331cfafa-6fc4-8148-bca2-c0a789625991)
        ├── 변경 이력 (DB: f347c198f95c4d0db925f31cdb7fb578, data_source: 5d1ba157-ca10-4339-afbe-b5a15fa55c21)
        └── 참고 문헌 & 리소스 (331cfafa-6fc4-81e0-96f2-dccaff84a4dd)
```

---

## Claude Code Access Rules

| Area | Permission | Notes |
|------|-----------|-------|
| 📂 연구 (1-6) | 📖 Read-only | Reference for implementation, especially 4. Methodology |
| 구현 태스크 DB — [요청] section | 📖 Read-only | Read task requests from Claude.ai |
| 구현 태스크 DB — [결과] section, status (진행중/완료), commit hash | ✍️ Write | Write implementation results |
| 구현 현황, 실험 결과 | 📖 Read-only | Claude.ai summarizes task results here |
| 📂 관리 | 📖 Read-only | Reference only |

---

## Task Workflow

Claude Code operates as the **implementer** in a feedback loop with
Claude.ai (the **designer**).

### Step 1 — Receive task

- Check 구현 태스크 DB for items with status "대기" (waiting).
- Read the [요청] section: goal, inputs, expected outputs, constraints.
- Read relevant 연구 pages (especially 4. Methodology) for formulas,
  hardware coefficients, and constraints.

### Step 2 — Implement & test

- Change task status to "진행중" (in progress).
- Implement according to the request. Internal planning and code
  modifications are autonomous — do NOT change the requested
  inputs/outputs.
- If a research-level issue is found (formula error, wrong coefficient,
  missing constraint), do NOT fix it independently. Report it in the
  [결과] section and let Claude.ai decide.
- Manage all code changes via git.

### Step 3 — Report results

- Write [결과] section in the same task item:
  - Execution results
  - Findings
  - Deviations from expectations (if any)
- Record the git commit hash.
- Change status to "완료" (complete).

### Task Status Reference

| Status | Meaning | Set by |
|--------|---------|--------|
| 대기 | Request ready, waiting for implementation | Claude.ai |
| 진행중 | Claude Code is working | Claude Code |
| 완료 | Results written, ready for review | Claude Code |
| 수정필요 | Implementation-level fix needed (same task) | Claude.ai |
| 설계변경필요 | Design change needed (new task will follow) | Claude.ai |

### Critical Principle — STRICT ENFORCEMENT

**These rules are STRICT and must be followed without exception.**

Research design decisions (formulas, coefficients, constraints) are
ALWAYS made by Claude.ai. Claude Code reports issues — never modifies
research pages directly.
