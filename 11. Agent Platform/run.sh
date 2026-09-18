#!/usr/bin/env bash
# run.sh — start all three services.
#
# Three PROCESSES, not three threads. That's the point: each can be
# restarted, scaled, or killed independently — which is exactly what the
# exercises have you do.
set -m
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export VERIFY_MODE="${VERIFY_MODE:-local}"

echo "VERIFY_MODE=$VERIFY_MODE"
uvicorn auth_service:app  --port 8001 --log-level warning &
AUTH=$!
uvicorn agent_service:app --port 8002 --log-level warning &
AGENT=$!
sleep 2
uvicorn gateway:app       --port 8000 --log-level info &
GW=$!

echo ""
echo "  gateway  http://localhost:8000   ← open this"
echo "  auth     http://localhost:8001"
echo "  agent    http://localhost:8002"
echo ""
echo "  kill auth only:  kill $AUTH"
echo "  Ctrl-C to stop all"

trap "kill $AUTH $AGENT $GW 2>/dev/null" EXIT
wait