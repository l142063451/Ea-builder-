# 

## Title

**Operating Guide for GitHub Copilot Agent (Agent Mode, Web)** — Build & Operate the **Visual FX Strategy Builder** (original, fxDreema-like) with Admin Panel, Codegen (MQL4/5), Backtesting, CI/CD to DigitalOcean

---

## Purpose

These instructions tell GitHub Copilot Agent exactly how to autonomously plan, implement, test, secure, document, and deploy the full production system described in `REQUIREMENTS.md`. No mock or placeholder features—ship real, integrated functionality. Keep a live, human-readable trail in `STATUS.md`.

> **Legal/IP**: Original product only; no copying of proprietary code/assets/text. Display and require acceptance of the disclaimer: “For educational use; not financial advice.” Comply with licenses and MetaQuotes ToS.

---

## Operating Rules (Agent)

1. **Single Source of Truth**: Treat `REQUIREMENTS.md` as the spec. If conflicts arise, open an issue and propose the safest choice.
2. **Roadmap by PRs (1→18)**: Implement strictly via the PR sequence in `REQUIREMENTS.md` (“Roadmap via 18 Pull Requests”). Never batch features across unrelated PRs.
3. **Atomic Commits & Conventional Commits**:

   * Format: `feat(scope): …`, `fix(scope): …`, `chore(ci): …`, `refactor(core): …`, etc.
   * One logical change per commit; keep diffs focused.
4. **Branching**:

   * Main branch protected.
   * Work on feature branches: `feat/pr-XX-short-title`.
   * Each PR references the roadmap number and includes a comprehensive checklist + test evidence.
5. **No Placeholders**: Implement real logic, real validations, real codegen, real backtests, real admin features. Feature-flag optional components, but don’t stub interfaces.
6. **Self-Validation Before PR**: Run lint, typecheck, unit, integration, E2E, security scans, screenshot capture. Update `STATUS.md` before opening PR.
7. **Status Discipline**: On **every** PR open/merge, update `/STATUS.md` with: Summary, Completed, In-Progress, Upcoming, Risks, Decisions.
8. **Security First**: Enforce RBAC, input validation (Zod), rate limits, secure headers, CSRF, HTTPS, Argon2id passwords, and audit logs. Never commit secrets.
9. **Observability**: Instrument code (OpenTelemetry). Ship structured logs, traces, and metrics. Provide dashboards and alerts.
10. **Documentation**: Keep `/docs/*` in sync. Each PR must update relevant docs. Keep `README.md` quickstart current.
11. **Screenshots & Videos**: Playwright must capture page/component screenshots and store in DO Spaces per commit SHA. Link them in PRs.
12. **DigitalOcean Deploy**: Staging auto-deploys on merge to `main`; production deploy gated by manual approval and semver tag.

---

## Repository & Tooling (Initialize in PR1)

* **Node**: `v20.x` LTS; **pnpm** package manager.
* **Monorepo**: Turborepo + pnpm workspaces.
* **Apps**: `apps/web` (React+Vite), `apps/api` (NestJS), `apps/worker` (Node workers).
* **Packages**:

  * `packages/graph-core` (schema, validators, traversal)
  * `packages/codegen-mql` (templates, emitter)
  * `packages/backtest-core` (interfaces)
  * `packages/backtest-js` (engine)
  * `packages/ui-nodes` (React Flow custom nodes)
* **Testing**: Vitest (unit/integration), Playwright (E2E).
* **Lint/Format**: ESLint (TS strict, security plugins), Prettier, commitlint, Husky hooks.
* **Type Safety**: `tsconfig` strict in all packages.

**Commands (root scripts)**

```jsonc
{
  "scripts": {
    "dev": "turbo run dev --parallel",
    "build": "turbo run build",
    "lint": "turbo run lint",
    "typecheck": "turbo run typecheck",
    "test": "turbo run test",
    "e2e": "turbo run e2e",
    "ci": "pnpm lint && pnpm typecheck && pnpm test && pnpm e2e",
    "migrate": "pnpm --filter apps/api prisma migrate deploy"
  }
}
```

---

## Environment & Secrets (Create Before PR7)

Define these **GitHub Actions Secrets** and **DigitalOcean** resources:

* `DO_TOKEN` (Personal Access Token with write on Apps/Registry)
* `DO_SPACES_KEY`, `DO_SPACES_SECRET`, `DO_SPACES_REGION`, `DO_SPACES_BUCKET`
* `DOCR_REGISTRY` (e.g., `registry.digitalocean.com/your-registry`)
* `DATABASE_URL` (DigitalOcean Managed PG)
* `REDIS_URL` (DigitalOcean Managed Redis)
* `JWT_SECRET`, `ENCRYPTION_KEY`
* `SMTP_URL` (for email), `OAUTH_GITHUB_ID/SECRET`, `OAUTH_GOOGLE_ID/SECRET`
* `APP_BASE_URL_STAGING`, `APP_BASE_URL_PROD`
* `OTEL_EXPORTER_OTLP_ENDPOINT` (optional)
* `SENTRY_DSN` (optional)

**Runtime env (12-factor)**:

* API: `PORT`, `NODE_ENV`, `CORS_ORIGINS`, `RATE_LIMIT`, `SECURE_COOKIES=true`
* Web: `VITE_API_URL`, `VITE_WS_URL`, `VITE_I18N_DEFAULT=en`
* Worker: `JOBS_CONCURRENCY`, `ARTIFACTS_BUCKET`, `ARTIFACTS_PREFIX`

---

## GitHub Project Hygiene

* **Issue Templates**: Bug report, Feature request, Tech debt.
* **PR Template**: Description, Linked Issue/PR#, Screenshots, Checklists (see below), Test plan, Risk/rollback plan.
* **Labels**: `prio:high`, `type:feat`, `type:chore`, `area:web`, `area:api`, `area:worker`, `security`, `docs`, `blocked`, `ready-for-review`.
* **CODEOWNERS**: `apps/web/* @maintainers`, etc. For now, assign to `@maintainers` pseudo-team.

---

## STATUS.md Protocol (Agent MUST follow)

**Structure** (always overwrite with latest state; keep history at bottom):

```markdown
# STATUS

## Summary
<1-2 sentences of current state>

## Completed
- [PR1] <title> — merged <sha>, date — notes & links

## In Progress
- [PRX] <title> — branch, checklist progress, blockers

## Upcoming (Next 3)
1) [PRY] <title>
2) [PRZ] <title>
3) [PR…] <title>

## Risks & Mitigations
- <risk> — <mitigation>

## Decisions (Architecture/Policy)
- <decision> — rationale — date — link to PR/issue

---

### History
<append snapshots per milestone>
```

**On each PR open**: Add to **In Progress** with checklist.
**On each PR merge**: Move to **Completed**, update Summary, refresh Upcoming.

---

## CI/CD (GitHub Actions)

Create workflows (in PR15 & PR18):

1. `.github/workflows/ci.yml`

   * Triggers: PR, push to branches.
   * Steps: checkout → setup pnpm+Node → `pnpm install` (with turbo cache) → `pnpm ci` (lint+typecheck+tests+e2e) → upload artifacts (coverage, test results, screenshots).
2. `.github/workflows/screenshot-upload.yml`

   * Trigger: `workflow_run` from `ci.yml` success.
   * Upload `/apps/web/playwright-report/**.png` to DO Spaces path: `screenshots/${{ github.sha }}/`.
3. `.github/workflows/deploy-staging.yml`

   * Trigger: push to `main`.
   * Build images → push to DOCR → update DO App (or DOKS) with image tags → run DB migrations → smoke tests.
4. `.github/workflows/deploy-prod.yml`

   * Trigger: Git tag `v*.*.*` (manual approval).
   * Same steps as staging; announce deployment; health checks; notify on success/failure.

**Quality Gates (fail build if any fail)**

* ESLint errors, TypeScript errors, unit/integration failures, Playwright failures.
* Coverage threshold: **80%+** lines on `packages/*` and `apps/api`.
* Security: CodeQL scan (separate workflow), `pnpm audit --audit-level=high` (or OSV scanner) must pass or be documented as accepted risk.

---

## PR Checklists (attach to every PR)

**General**

* [ ] Linked issues & roadmap PR number.
* [ ] Updated docs (`/docs/*`, `README`, `STATUS.md`).
* [ ] Added/updated unit & E2E tests.
* [ ] Screenshots or terminal output attached.
* [ ] Security review: inputs validated, RBAC enforced, secrets not logged.
* [ ] Performance check (if applicable) — basic profiling notes.

**Feature-Specific (pick relevant)**

* **Graph Core**: schema version bump, migration path, validators proven with golden fixtures.
* **Node Editor**: invalid connections blocked, undo/redo works, JSON import/export stable.
* **Codegen**: emitted `.mq4/.mq5` compile in MetaEditor (evidence: CI Windows runner log or manual signed artifact).
* **Backtest**: deterministic on seed, perf meets targets, metrics validated.
* **Admin**: RBAC checks in API/UI, audit entries created, feature flags effective.
* **CI/CD**: cache works, artifacts uploaded, DO deploy successful, rollback path verified.

---

## Security Requirements (enforce across PRs)

* **Auth**: Argon2id, JWT rotation, refresh token revocation list; secure cookies; CSRF for web forms.
* **RBAC**: Route guards & API RBAC middleware; least privilege defaults.
* **Validation**: Zod on all inputs; length/enum/range constraints; sanitize HTML (CMS).
* **Headers**: Strict CSP, HSTS, X-Frame-Options DENY, X-Content-Type-Options, Referrer-Policy strict.
* **Rate Limits**: Per IP + user, sliding window; ban list integration.
* **File Uploads**: MIME validation; antivirus scan (ClamAV container) for datasets/templates.
* **Audit**: Immutable audit logs for admin operations; include `actorId`, `ip`, before/after snapshot.
* **Secrets**: From env only; never printed; redact in logs.
* **Dependencies**: Lockfiles committed; periodic audit; pin high-risk packages.

---

## Observability & Ops

* **Logging**: JSON logs with correlation IDs; Pino or Winston.
* **Tracing**: OpenTelemetry SDK; span web→api→worker; add key spans for codegen/backtest.
* **Metrics**: Request latency, error rate, queue depth, job duration, backtest throughput.
* **Dashboards**: Provision Grafana (or DO compatible).
* **Alerts**: Error rate > 2% for 5m, queue depth > threshold, deploy failures.

---

## Admin Panel Implementation (PR12–PR14)

* **RBAC**: `owner`, `admin`, `maintainer`, `user` with scoped permissions (org/project).
* **Feature Flags**: Create/edit, target by user/org/env; client & server evaluators.
* **Node Catalog**: CRUD nodes (key, category, enabled), versioned schemas (params, ports, validations), test compile against sample graphs.
* **Templates**: Upload/version MQL templates; compute checksum; diff view; compile check job.
* **Themes**: Token editor (colors, spacing, typography) + preview & publish.
* **CMS**: WYSIWYG (sanitized) with schedules, drafts, publish; preview links.
* **Audit Logs**: Filter/search/export; per-entity drilldowns.
* **Artifacts**: List/download generated EAs, reports, screenshots, with commit linkage.
* **Danger Zone**: Rotate keys (reseed JWT, invalidate sessions), purge caches, rebuild search index.

---

## Visual Node Editor (PR4–PR6)

* **Canvas**: React Flow with pan/zoom, mini-map, grid, snap, keyboard shortcuts (undo/redo/copy/paste/delete).
* **Palette**: Searchable, categorized; quick insert via `/` command palette.
* **Inspector**: Zod-driven forms, live validation, inline docs, tooltips.
* **Validation**: Type-checked ports; cycle detection; incompatible link prevention; actionability hints.
* **Subgraphs**: Save selection as reusable block; parameterized; versioned.
* **Diffing**: Visual graph diff between strategy versions.
* **I/O**: JSON import/export (semver with `graph-core`).

---

## Code Generation (PR8 + PR9)

* **Templates**: Handlebars/EJS per node & MQL target (4/5).
* **Traversal**: Topological walk; gather declarations; render lifecycle (`OnInit`, `OnTick`, `OnDeinit`).
* **Trading Safety**: Spread/slippage/max lot guards, freeze level checks, min stop distance, retry on transient errors with `GetLastError`.
* **Inputs**: Strategy tunables exposed as EA inputs.
* **Artifacts**: Store `.mq4`/`.mq5` + manifest JSON in DO Spaces; link to StrategyVersion.
* **Compile Check**: CI job on Windows self-hosted runner to compile EAs; upload logs.

---

## Backtesting (PR10–PR11)

* **Engine**: Candle loop, indicators (MA, RSI, MACD, ATR, Stoch), PnL accounting, equity curve.
* **Config**: Spread, slippage, commission, position sizing %, TP/SL, trailing stop.
* **Metrics**: Net PnL, drawdown, win rate, profit factor, Sharpe-like, exposure.
* **Determinism**: Seeded RNG for slippage; stable results across runs.
* **UI**: Run config form; progress via WebSocket; charts (equity, drawdown, distribution).
* **Exports**: JSON/CSV of trades and metrics.

---

## DigitalOcean Deployment (PR18)

**Option A — DO App Platform (default)**

* **Images**: Build & push to DOCR.
* **Apps**:

  * `web` → static deploy (Vite build) served via App Platform or Nginx image.
  * `api` → Node app with env wiring & healthcheck.
  * `worker` → Node worker with BullMQ and concurrency env.
* **Managed**: PG, Redis, Spaces.
* **CD**: `deploy-staging.yml` (auto on `main`), `deploy-prod.yml` (tag-gated).
* **Health Checks**: `/healthz` for api & worker readiness endpoints.

**Option B — DOKS (Kubernetes)**

* Helm charts for web, api, worker; HPA, Nginx Ingress, cert-manager.
* External Secrets for env; DO Load Balancer; Rolling updates.

**Rollback**

* App Platform: redeploy previous image tag.
* DOKS: `helm rollback <release> <rev>`.

---

## Smoke & E2E Testing (must pass before merge)

* **Smoke (post-deploy)**:

  * `GET /healthz` returns OK for api & worker.
  * Web loads; login works; node editor renders; admin link visible for admin account.
* **Playwright E2E**:

  * Sign up/sign in/out.
  * Create project & strategy; add nodes; validate connections; save.
  * Run backtest; view metrics; download report.
  * Export EA; download artifact; (if Windows runner available) compile check pass.
  * Admin: create feature flag; disable node category; verify UI change; view audit log.
* **Screenshots/Video**: All main screens (light/dark), modals, error states; uploaded to Spaces with `commit SHA` path.

---

## Performance Targets

* Web P95 `< 2.0s` load on DO staging; API P95 `< 250ms` @ 200 RPS; Backtest 1y M1 `< 10s` on standard DO droplet. Profile hot paths; memoize graph traversal; use worker queues for heavy jobs.

---

## Failure Handling & Recovery

* If any step fails:

  1. Stop and open a GitHub issue with precise logs and proposed fix.
  2. If deploy failed, perform rollback.
  3. Update `STATUS.md` with incident summary and next steps.
  4. Add automated test to prevent recurrence.

---

## Getting Started (Agent, PR1)

1. Initialize monorepo (Turborepo + pnpm).
2. Add linting, formatting, commit hooks, commitlint, editorconfig.
3. Create directories (`apps/*`, `packages/*`, `docs/*`).
4. Add base READMEs, `STATUS.md` initial scaffold.
5. CI skeleton for lint+typecheck+unit; mark E2E/Deploy to be added in later PRs.
6. Update `STATUS.md` and open **PR1** with checklist and screenshots (where applicable).

---

## Definition of Done (per PR)

* All acceptance criteria in `REQUIREMENTS.md` for that PR met.
* Code, tests, docs updated.
* CI green (lint, type, unit, integration, E2E, security scans).
* Screenshots uploaded & linked.
* `STATUS.md` updated.
* For deploy PRs: staging healthy; rollback verified.

---

## Command Reference (Agent)

* **Install**: `pnpm i`
* **Dev**: `pnpm dev`
* **Build**: `pnpm build`
* **Lint/Type**: `pnpm lint && pnpm typecheck`
* **Tests**: `pnpm test`
* **E2E**: `pnpm e2e` (headless; records)
* **Migrate**: `pnpm migrate`
* **Generate artifacts**: Trigger via API or worker queue; outputs to DO Spaces.
* **Deploy (CI)**: On push/merge per workflows; manual prod via tag.

---

## Mandatory Disclaimers (UI & Docs)

* Display on backtest/export screens: “For educational use; not financial advice.”
* Require checkbox acknowledgment before running backtests or exporting EAs.

---
