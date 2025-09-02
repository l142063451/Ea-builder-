

REQUIREMENTS.md

Title

Visual FX Strategy Builder (Original, fxDreema-like) — Production-Grade Web Platform with Admin Control, Live Testing, Codegen (MQL4/MQL5), and Full CI/CD to DigitalOcean

Overview

Build a secure, production-ready, web platform that lets users visually design algorithmic FX strategies via a node/block editor, backtest them, and export them as MQL4/MQL5 Expert Advisors (EAs). It includes a full admin panel to control everything (feature flags, node catalog, templates, theming, content, users, roles, audits). It ships with solid CI/CD, live testing, and observability, and deploys to DigitalOcean. No placeholders or mock-only implementations—use real integrations, compile-ready codegen, and runnable tests.

> Legal/IP: Create an original product; do not copy proprietary text/UI/code from fxDreema or others. Include disclaimers: “For educational use; not financial advice.” Comply with licenses and MetaQuotes ToS.




---

Objectives (Measurable)

Users can build a strategy with visual nodes, validate connections, backtest on uploaded OHLCV data, and export compilable .mq4/.mq5 within <10 minutes.

Codegen produces MQL that compiles without errors for included example strategies (MA crossover, RSI strategy, MACD momentum).

Backtest (JS engine) runs 1 year of M1 data < 10s locally on typical server hardware; results deterministic across runs (seeded).

Admin panel can create/update nodes, templates, feature flags, themes, content, users/roles; emits audit logs for all changes.

CI: All PRs gated by lint, typecheck, unit/integration/E2E tests, and auto screenshots of major UI.

Deployment: One-click auto deploy to DigitalOcean staging & production via GitHub Actions; rollbacks in < 2 minutes.



---

In Scope

Visual node editor (React-based) with type-safe ports, validation, undo/redo, snap-to-grid, keyboard shortcuts, search, mini-map, import/export JSON.

Node catalog covering events, indicators, conditions, math/transform, actions, flow (detailed below).

MQL4/MQL5 code generation via template modules (no placeholders); mapping indicators and orders with guards and error handling.

Backtest engine (JS) with indicators, PnL metrics, drawdown, Sharpe-like ratio, slippage/spread config.

Project/strategy management: versions, tags, descriptions, share/export.

Admin panel: RBAC, feature flags, node catalog CRUD, templates, theming, CMS-like content, audit logs, moderation, rate limits.

Full CI/CD, security hardening, observability, documentation, status tracking via status.md.

Deployment to DigitalOcean (Managed DB/Redis, Spaces, App Platform or DOKS).


Out of Scope (for MVP)

Broker order execution or live trading connectivity.

Copy trading/social graph.

Proprietary indicator packs (beyond standard set).



---

Personas

Quant Hobbyist: builds simple EAs visually, tests ideas quickly.

Quant Dev: needs codegen quality, validation, reproducible backtests.

Educator/Coach: needs tutorials, templates, shareable examples.

Admin/Owner: configures catalog, templates, policies, themes.



---

System Architecture

Tech Stack

Frontend (apps/web): React + TypeScript, Vite, React Flow, TailwindCSS, Zustand (or Redux Toolkit), Zod for forms, i18n (English default).

Backend (apps/api): Node.js + TypeScript, NestJS, PostgreSQL (Prisma), Redis (BullMQ), REST + WebSocket.

Workers (apps/worker): Node workers for codegen/backtests; job queue via BullMQ.

Packages: graph-core (schema/validators/traversal), codegen-mql (templates/emitter), backtest-core (interfaces), backtest-js (engine), ui-nodes (custom nodes).

Storage: PostgreSQL (managed), Redis (managed), S3-compatible (DO Spaces) for artifacts, screenshots, exports.

CI/CD: GitHub Actions, protected main branch, versioning via tags.

Observability: OpenTelemetry traces, structured logs, metrics; Grafana/Loki/Tempo (or DO-compatible stack).

Auth: Email/password + optional OAuth (GitHub/Google).

Secrets: GitHub OIDC → DO; managed via environment variables in DO.


Environments

Local (Docker Compose), Staging (DO), Production (DO).

Staging auto-deploys on main; Production via manual approval & tag.


Deployment (DigitalOcean)

Option A (App Platform): Apps for Web, API, Worker; DO Managed PG, DO Managed Redis, DO Spaces, DO Container Registry.

Option B (DOKS): Kubernetes with Helm charts, HPA, Nginx Ingress; use DO managed services; recommended for scale.



---

Data Model (ERD)

erDiagram
  User ||--o{ Session : has
  User ||--o{ ApiToken : has
  User ||--o{ OrgMember : has
  Organization ||--o{ OrgMember : has
  Organization ||--o{ Project : owns
  Project ||--o{ Strategy : contains
  Strategy ||--o{ StrategyVersion : versions
  StrategyVersion ||--o{ Artifact : generates
  StrategyVersion ||--o{ BacktestJob : runs
  BacktestJob ||--o{ BacktestResult : yields
  NodeDef ||--o{ NodeDefVersion : versions
  Template ||--o{ TemplateVersion : versions
  AuditLog
  FeatureFlag
  Theme
  CmsPage
  Screenshot

  User {
    uuid id PK
    text email
    text passwordHash
    text role  // admin|maintainer|user
    jsonb profile
    timestamptz createdAt
  }
  Organization {
    uuid id PK
    text name
    jsonb settings
    timestamptz createdAt
  }
  Project {
    uuid id PK
    uuid orgId FK
    text name
    text description
    jsonb settings
    timestamptz createdAt
  }
  Strategy {
    uuid id PK
    uuid projectId FK
    text name
    text target  // MQL4|MQL5
    text status  // draft|ready|deprecated
    jsonb metadata
  }
  StrategyVersion {
    uuid id PK
    uuid strategyId FK
    text version
    jsonb graph  // nodes+edges
    text changelog
    timestamptz createdAt
  }
  Artifact {
    uuid id PK
    uuid strategyVersionId FK
    text kind // mq4|mq5|json|report|zip
    text url  // DO Spaces
    jsonb meta // hashes, sizes
    timestamptz createdAt
  }
  NodeDef {
    uuid id PK
    text key  // e.g., "indicator.rsi"
    text category
    bool enabled
  }
  NodeDefVersion {
    uuid id PK
    uuid nodeDefId FK
    jsonb schema  // params, ports
    text codeGenKey
    timestamptz createdAt
  }
  Template {
    uuid id PK
    text key
    text target // mql4|mql5
  }
  TemplateVersion {
    uuid id PK
    uuid templateId FK
    text path
    text checksum
    timestamptz createdAt
  }
  BacktestJob {
    uuid id PK
    uuid strategyVersionId FK
    text status // queued|running|failed|done
    jsonb config // slippage, spread, fees
    text engine // js|mt5
    timestamptz startedAt
    timestamptz finishedAt
  }
  BacktestResult {
    uuid id PK
    uuid backtestJobId FK
    jsonb metrics // pnl, winrate, dd, sharpeLike
    jsonb equityCurve
    jsonb trades
  }
  FeatureFlag {
    uuid id PK
    text key
    bool enabled
    jsonb rules
  }
  Theme {
    uuid id PK
    text name
    jsonb tokens // colors, typography
  }
  CmsPage {
    uuid id PK
    text slug
    jsonb content
    bool published
  }
  AuditLog {
    uuid id PK
    uuid actorId
    text action
    text entity
    uuid entityId
    jsonb before
    jsonb after
    timestamptz at
    text ip
  }
  Screenshot {
    uuid id PK
    text path
    text page
    jsonb meta // viewport, hash, semver
  }


---

Visual Editor: Nodes, Blocks, Sub-Blocks

Editor Features

Canvas: pan/zoom, mini-map, grid, snap-to-grid, guides, selection box.

Searchable palette grouped by categories, keyboard quick insert.

Type-checked ports (e.g., series<number>, number, bool, orderHandle, flow).

Validation: cycle detection, dead-end flows, missing params, incompatible connections; inline error badges.

Undo/redo, copy/paste, templates (user-defined blocks), subgraphs (reusable composite nodes).

Inspector: form-driven params with Zod schemas, tooltips, live docs.

Version diff: graph diff visualizer (added/removed/changed nodes/edges).

Import/Export: JSON schema with semantic versioning.


Node Categories & Definitions (MVP)

Events:

OnTick, OnTimer(interval), OnTrade(eventType)

Outputs: flow, optional event payload.


Indicators:

MA(period, method, price) -> series<number>

RSI(period, price) -> series<number>

MACD(fast, slow, signal) -> {macd, signal, hist}

ATR(period) -> series<number>

Stochastic(k, d, slowing) -> {k, d}


Math/Transform:

Add/Sub/Mul/Div(a,b) -> number|series (broadcast aware)

EMA(input, period)

Lag(series, shift)

Normalize(series)

ZScore(series, period)


Conditions:

CrossOver(a,b) / CrossUnder(a,b)

GreaterThan(a,b) / LessThan(a,b)

InRange(value, low, high)

SlopeAbove(series, threshold)


Actions/Orders:

Buy(market|limit|stop, size%, tp, sl, trailing, magic)

Sell(...)

ModifyOrder(orderHandle, tp, sl)

CloseAll, CloseByCondition

Outputs: orderHandle, bool success.


Flow Control:

If(cond) → then/else

Switch(value)

Throttle(period)

Debounce(period)

Once

ForEach(barRange)



> Each node has: params schema, input/output port types, validate(), and a codeGenKey mapping to MQL templates. Admin can version these.



Connections

Edges connect compatible port types; runtime ensures symbol/timeframe context compatibility.

Flow edges orchestrate execution order; data edges pass values/series.

Guard conditions on edges (optional) to short-circuit flow.



---

Code Generation (MQL4/MQL5)

Strategy

Traversal: topological traversal of flow graph; gather declarations; emit code segments into lifecycle functions (OnInit, OnTick, OnDeinit).

Templates: Handlebars/EJS templates per node & per target (MQL4/MQL5); include indicator function calls (iMA, iRSI, iMACD, iATR, iStochastic), order ops, error handling (GetLastError), spread/slippage guards, magic numbers, symbol/timeframe, shift indexing.

Safety: respect minimum stop distance, freeze levels, market status, max spread, max lot; retry on transient trade errors.

Parameters: expose extern/inputs for tunables (periods, risk %) as EA inputs.

Artifacts: emit compilable .mq4/.mq5 plus a readme with inputs and known limitations.


> Acceptance: sample strategies compile in MetaEditor without errors.




---

Backtesting

JS Engine (MVP)

Inputs: OHLCV CSV/Parquet, symbol, timeframe, backtest window, spread/slippage, commissions.

Indicators via performant JS lib (e.g., native implementations in the engine package).

Event loop: candle-based; apply strategy logic each tick; position tracking, PnL, equity curve.

Metrics: net PnL, max drawdown, win rate, profit factor, basic Sharpe-like (simple standardization), exposure time.

Deterministic with seed; runs 1y M1 < 10s target on standard DO droplet.


MT5 Strategy Tester (Optional, Post-MVP)

Adapter to Windows self-hosted runner (or Wine) to invoke Strategy Tester CLI with generated EA and data; parse HTML/XML/CSV results.

Admin can enable/disable MT5 adapter via feature flag.



---

Admin Panel (Full Control)

RBAC: Roles (owner, admin, maintainer, user), org/project scoping, permissions matrix.

Node Catalog Management: CRUD node defs & versions, params, ports, validation logic, enable/disable.

Template Management: upload and version templates (MQL4/MQL5) with checksums, diff view, test compile on sample graphs.

Feature Flags: per-env, rule-based (by org/user).

Theming: token-based color/typography, dark mode, brand assets; preview & publish.

CMS: editable pages (landing, docs, tutorials), banners, announcements; scheduled publish.

Users & Orgs: invites, SSO config, quotas.

Moderation: rate limits, IP bans, content flags, spam detection toggles.

Artifacts: browse generated EAs, backtest reports, download links.

Audit Logs: immutable record of all admin actions; search & export.

Screenshots: view auto-captured screenshots from CI (per page/component).

Danger Zone: rotate secrets, purge caches, regen search index.



---

API (Representative Endpoints)

POST /auth/login, POST /auth/register, POST /auth/oauth/callback

GET /projects, POST /projects, PATCH /projects/:id

GET /strategies?projectId=, POST /strategies, PATCH /strategies/:id

POST /strategies/:id/version (create version from current graph)

POST /codegen/:strategyVersionId → jobId

GET /artifacts/:id (presigned URL)

POST /backtests → jobId, GET /backtests/:id

WS /jobs/:id/stream progress

Admin-only:

GET/POST/PATCH /admin/node-defs, /admin/templates, /admin/feature-flags, /admin/themes, /admin/audit-logs, /admin/cms, /admin/users




---

Security & Compliance

HTTPS everywhere; HSTS; secure cookies; CSRF protection; strict CORS.

Argon2id password hashing; JWT with rotation & short TTL; refresh tokens in DB.

RBAC enforcement across API and UI routes; input validation with Zod; output typing.

Rate limiting (IP+user), request size limits, file scanning for uploads (MIME & antivirus).

Secrets via DO environment / OIDC; no secrets in repo.

Trading disclaimers mandatory acceptance for export/backtest features.

Full audit logging for admin actions.



---

Performance & Reliability

P95 page load < 2.0s, TTI < 3.0s on median hardware.

Backtest queue resilient to worker restarts; jobs idempotent; exponential backoff.

DB migrations transactional; backups daily with 7/30 retention.

Horizontal scaling for API & Worker; health checks & graceful shutdown.



---

Observability

Structured logs (JSON) with correlation IDs.

Tracing (OpenTelemetry) across web→api→worker→db.

Metrics (CPU, mem, queue depth, request latency).

Alerting on error rate spikes, job failures, slow queries.



---

UX & Design Requirements (Psychological, Impactful, Futuristic)

Design System: Tokenized (colors, spacing, typography), glassmorphism accents, motion micro-interactions (<150ms), meaningful empty states, progress affordances.

Persuasive UX: progress steps, streaks for learning, contextual nudges (“You’re 1 node away from a runnable strategy”).

Accessibility: WCAG 2.1 AA; keyboard nav; focus rings; high contrast toggle.

Internationalization: English default; i18n-ready.

Screenshot Coverage: Playwright autogenerates screenshots of every page, modal, and primary component state; stored in DO Spaces with commit hash.



---

Testing

Unit: graph validators, codegen mappers, indicator functions.

Integration: codegen→compile smoke tests (syntax check stubs or containerized compiler step), worker job lifecycle.

E2E (Playwright): sign-in, node editor flows (add, connect, validate, save), backtest run, artifact download, admin edits; auto-screenshot & video on CI.

Security: authz tests, rate limit tests, input fuzzing.

Load: k6 for API/backtest endpoints.



---

Documentation Deliverables

/docs/PRD.md, /docs/ARCHITECTURE.md, /docs/NODES.md, /docs/CODEGEN.md, /docs/BACKTESTING.md, /docs/SECURITY.md, /docs/DISCLAIMER.md

/STATUS.md: living status tracker (completed, in-progress, upcoming, blockers) — must be updated by CI on each PR merge.

/INSTRUCTIONS.md: Copilot Agent operating guide (to be delivered next)

README.md with quickstart, environment vars, deployment.



---

Acceptance Criteria (MVP)

User creates MA crossover strategy, backtests on sample data, and exports .mq5 that compiles.

Node editor prevents invalid connections; validation messages are clear.

Admin disables a node category; it disappears instantly for non-admins.

CI runs full test matrix, publishes screenshots & artifacts; staging auto-deploys.

Production rollout/rollback functional; observability dashboards live.



---

Roadmap via 18 Pull Requests (PR 1 → PR 18)

1. PR1 – Monorepo & Tooling: Turborepo/pnpm, linting, formatting, Husky, commitlint, TypeScript baseline.


2. PR2 – Graph Core: JSON schema, Zod models, validation, traversal, tests.


3. PR3 – UI Shell & Design System: Tailwind tokens, themes, layout, auth shell.


4. PR4 – Node Editor (Base): React Flow canvas, palette, drag/drop, selection, mini-map.


5. PR5 – Node Params & Validation: Inspector forms, port typing, cycle detection, guard rails.


6. PR6 – Node Catalog (MVP): Events, Indicators, Math, Conditions, Actions, Flow nodes wired.


7. PR7 – API & Persistence: NestJS + Prisma models for users, projects, strategies, versions.


8. PR8 – Codegen Package: MQL templates (MA, RSI, MACD, ATR, Stochastic), emitter, snapshot tests.


9. PR9 – Worker & Jobs: BullMQ queue, codegen job, artifacts to DO Spaces.


10. PR10 – Backtest JS Engine: indicators, loop, metrics, deterministic runs.


11. PR11 – Backtest UI & Reports: run config UI, result charts, export CSV/JSON.


12. PR12 – Admin Panel (RBAC + Flags): roles, feature flags, audit logs.


13. PR13 – Admin Node/Template Mgmt: CRUD node defs & templates, versioning, test compile.


14. PR14 – CMS & Theming: editable content, themes, preview & publish.


15. PR15 – CI/CD & Screenshots: GitHub Actions, Playwright E2E, screenshot capture to Spaces.


16. PR16 – Observability: logs, metrics, traces, dashboards, alerts.


17. PR17 – Security Hardening: rate limits, CSP, headers, fuzz tests, DAST.


18. PR18 – DigitalOcean Deploy: App Platform or DOKS manifests, env wiring, rollout & rollback.




---

Risks & Mitigations

MT5 CLI constraints: keep optional, feature-flagged; default to JS engine.

Indicator parity differences: document exact MQL signatures; unit-test templates.

Performance on large graphs: memoize traversal, virtualize UI lists, profile.

User misuse (financial risk): strong disclaimers; tutorial-first UX; block direct broker links.



---

Glossary

EA: Expert Advisor (MQL4/5 automated strategy).

Graph: Node/edge JSON representing strategy logic.

Artifact: Generated file (EA, report, screenshot).

RBAC: Role-Based Access Control.

Do Spaces: DigitalOcean S3-compatible object store.



---

