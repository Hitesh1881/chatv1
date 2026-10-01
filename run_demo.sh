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

CHECKPOINT="$ROOT/artifacts/tiny_chatv1.pt"
NEEDS_TRAINING=0

if [[ ! -f "$CHECKPOINT" ]]; then
  NEEDS_TRAINING=1
else
  if ! "$PYTHON" - "$CHECKPOINT" <<'PY'
import sys
import torch
from chatv1.tokenizer import CharTokenizer

checkpoint = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
tokenizer = CharTokenizer.from_state_dict(checkpoint["tokenizer"])
tokenizer.encode("ok")
tokenizer.encode("User: hello\nAssistant:")\nif int(checkpoint.get("training_version", 0)) < 2:\n    raise ValueError("training corpus version is stale")
PY
  then
    echo "Existing checkpoint tokenizer is stale for the current demo."
    NEEDS_TRAINING=1
  fi
fi

if [[ "$NEEDS_TRAINING" -eq 1 ]]; then
  echo "Training the real ChatV1 tiny model with the current tokenizer..."
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
