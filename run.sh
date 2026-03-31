#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  run.sh — Local development launcher for Climate AI v2
#  Starts the FastAPI backend and Streamlit frontend in parallel.
#
#  Usage:
#    chmod +x run.sh
#    ./run.sh              # Start API + Frontend
#    ./run.sh --api-only   # Start only the FastAPI backend
#    ./run.sh --ui-only    # Start only the Streamlit frontend
#    ./run.sh --train      # Run model training then start services
#    ./run.sh --init       # Load initial data from GEE then train then start
# ─────────────────────────────────────────────────────────────────────────────

set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# ── Colours ──
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${CYAN}[climate-ai]${NC} $*"; }
ok()   { echo -e "${GREEN}[   OK   ]${NC} $*"; }
warn() { echo -e "${YELLOW}[  WARN  ]${NC} $*"; }
err()  { echo -e "${RED}[  ERR   ]${NC} $*"; exit 1; }

# ── Load env ──
if [ -f ".env" ]; then
  export $(grep -v '^#' .env | xargs)
  log "Loaded .env"
else
  warn ".env not found — using defaults. Copy .env.example to .env and fill in keys."
fi

# ── Check Python ──
python3 --version >/dev/null 2>&1 || err "python3 not found."

# ── Virtual env ──
if [ -f "venv/bin/activate" ]; then
  log "Activating local venv ..."
  source "venv/bin/activate"
elif [ -f "../../sprint/bin/activate" ]; then
  log "Activating legacy sprint environment ..."
  source "../../sprint/bin/activate"
else
  warn "No virtual environment activated. You may need to run 'python -m venv venv' first."
fi

# ── Install deps ──
log "Checking dependencies ..."
pip install -q -r requirements.txt

# ── Parse args ──
MODE="both"
for arg in "$@"; do
  case $arg in
    --api-only)  MODE="api"   ;;
    --ui-only)   MODE="ui"    ;;
    --train)     MODE="train_and_start" ;;
    --init)      MODE="init_and_start"  ;;
  esac
done

# ── Initial data load ──
if [ "$MODE" = "init_and_start" ]; then
  log "Running initial GEE data load ..."
  python3 data/pipeline.py --mode init
  ok "Initial data load complete."
  log "Running model training ..."
  python3 scripts/train_all.py
  ok "Training complete."
  MODE="both"
fi

# ── Training only then start ──
if [ "$MODE" = "train_and_start" ]; then
  log "Running model training ..."
  python3 scripts/train_all.py
  ok "Training complete."
  MODE="both"
fi

# ── API Backend ──
start_api() {
  log "Starting FastAPI backend on http://0.0.0.0:${API_PORT:-8000} ..."
  uvicorn api.main:app \
    --host "${API_HOST:-0.0.0.0}" \
    --port "${API_PORT:-8000}" \
    --reload \
    --log-level info &
  API_PID=$!
  ok "API started (PID $API_PID)"
}

# ── Frontend ──
start_ui() {
  log "Starting Streamlit frontend on http://localhost:8501 ..."
  streamlit run frontend/app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false &
  UI_PID=$!
  ok "Frontend started (PID $UI_PID)"
}

# ── Trap for cleanup ──
cleanup() {
  log "Shutting down services ..."
  [ -n "$API_PID" ] && kill "$API_PID" 2>/dev/null
  [ -n "$UI_PID"  ] && kill "$UI_PID"  2>/dev/null
  log "Done."
}
trap cleanup EXIT SIGINT SIGTERM

# ── Launch ──
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  🌍  Climate AI v2 — Launch"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

case $MODE in
  api)  start_api ;;
  ui)   start_ui  ;;
  both)
    start_api
    sleep 2
    start_ui
    ;;
esac

echo ""
log "Services running. Press Ctrl+C to stop."
echo ""
log "  📡 API:      http://localhost:${API_PORT:-8000}"
log "  📊 Docs:     http://localhost:${API_PORT:-8000}/docs"
log "  🖥️  Frontend: http://localhost:8501"
echo ""

wait
