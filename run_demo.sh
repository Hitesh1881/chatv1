#!/bin/zsh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"

if ! "$PYTHON" -c "import torch" >/dev/null 2>&1; then
  echo "Python/PyTorch environment not ready."
  echo "Create/activate your .venv, then install the project with:"
  echo "  python3 -m pip install -e '.[dev,image]'"
  exit 1
fi

if ! "$PYTHON" -c "import chatv1" >/dev/null 2>&1; then
  "$PYTHON" -m pip install -e ".[dev,image]"
fi

if [[ ! -f "$ROOT/artifacts/tiny_chatv1.pt" ]]; then
  echo "No local checkpoint found. Training the real ChatV1 tiny model..."
  "$PYTHON" "$ROOT/scripts/train_tiny.py" --steps 400
fi

echo "Starting ChatV1 local services..."
"$PYTHON" "$ROOT/scripts/serve_chat.py" >"$ROOT/artifacts/chat.log" 2>&1 &
CHAT_PID=$!
"$PYTHON" "$ROOT/scripts/serve_image.py" >"$ROOT/artifacts/image.log" 2>&1 &
IMAGE_PID=$!

cleanup() {
  kill "$CHAT_PID" "$IMAGE_PID" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

sleep 2
echo ""
echo "Chat API:  http://127.0.0.1:8001"
echo "Image API: http://127.0.0.1:8000"
echo "Demo:      $ROOT/web/chatv1-demo.html"
echo ""
echo "Open the Demo URL in Brave/Chrome."
echo "Image model loads only when you click Generate Image."
echo "Press Ctrl+C to stop both services."

if command -v open >/dev/null 2>&1; then
  open "$ROOT/web/chatv1-demo.html"
fi

wait
