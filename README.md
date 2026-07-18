# Visual Image Search (pharmacy catalog)

Self-hosted **image-to-image** visual search added to an existing pharmacy
e-commerce catalog (~80k products, ~320k images), fused with the existing
keyword + semantic text search. This is **additive**: the existing text search
is a black box we call and merge from.

## Architecture (two tiers)

- **`ml-service/`** — Python FastAPI GPU service: image encoder
  (SigLIP2 default / DINOv3 gated), PaddleOCR, and Qdrant vector search.
  Endpoints: `/search`, `/ocr`, `/embed` (diagnostics), `/healthz`.
- **`worker/`** — Cloudflare Worker edge: upload handling (ImageKit), D1
  metadata (via binding), authenticated calls to the ML service, hybrid fusion,
  response shaping, rate limiting.
- **`infra/`** — Docker Compose (Qdrant + ml-service) and Qdrant config.
- **`db/`** — D1 schema migrations + dev seed.
- **`eval/`** — recall@k / latency / threshold-calibration harness.
- **`docs/`** — provider alternatives, cost model, DINOv3 license review, runbook.

See `.plans/v2-image-search.md` (repo-external) for the full task breakdown and
`docs/` for operational detail.

## Secrets

All secrets are referenced by **name only** — never commit literal values.
Copy `.env.example` to `.env` (git-ignored) for the ML service / Compose stack.
Worker secrets are set with `wrangler secret put <NAME>`.

## Run it

### ML service + Qdrant (GPU host)

```bash
cp .env.example .env            # fill real values (git-ignored)
docker compose -f infra/docker-compose.yml up --build
# health check:
curl http://localhost:8000/healthz
# GPU verification:
docker compose -f infra/docker-compose.yml run --rm ml-service \
    python -c "import torch; assert torch.cuda.is_available()"
```

Local (CPU / no Docker) for tests:

```bash
cd ml-service
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt        # heavy GPU deps; see Dockerfile notes
pytest tests/
```

### Cloudflare Worker (edge)

```bash
cd worker
npm install
npx wrangler d1 execute image-search --local --file ../db/migrations/0001_init.sql
npm run dev                # wrangler dev (local D1 + mock bindings)
npm test                   # vitest + @cloudflare/vitest-pool-workers
npx wrangler deploy --dry-run
```

Greenfield: local tests use a **local D1** and mock bindings; `--dry-run`
validates config without provisioned remote resources.
