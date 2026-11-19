#!/usr/bin/env bash
# Make sure an Ollama server answers on localhost:11434, then pull the models given as args.
#
# On WSL the server should be the *Windows* Ollama app: running the model inside WSL took
# 5.4 GB of the VM's 7 GB RAM and got it OOM-killed. The Windows app uses the GPU directly
# and is reachable from WSL on localhost (mirrored networking). On native Linux/macOS the
# script starts a locally installed `ollama` instead.
set -euo pipefail

URL="http://localhost:11434"
up() { curl -sf -m 2 "$URL/api/version" >/dev/null; }

if ! up; then
  win_dir=$(ls -d /mnt/c/Users/*/AppData/Local/Programs/Ollama 2>/dev/null | head -1 || true)
  if [ -n "$win_dir" ]; then
    echo "starting Ollama for Windows"
    # Start the server itself, hidden and detached. (Launching the tray app from WSL does
    # not bring the server up, and `cmd.exe /c start` blocks the calling shell.)
    (cd /mnt/c && timeout 30 powershell.exe -NoProfile -Command \
      "Start-Process -WindowStyle Hidden -FilePath (Join-Path \$env:LOCALAPPDATA 'Programs\Ollama\ollama.exe') -ArgumentList 'serve'" \
      >/dev/null 2>&1) || true
  elif command -v ollama >/dev/null && ! grep -qi microsoft /proc/version; then
    # Native Linux/macOS (not WSL): run the server locally.
    echo "starting Ollama"
    setsid nohup ollama serve >"$HOME/.ollama-serve.log" 2>&1 </dev/null &
  else
    echo "No Ollama server found. On WSL install the Windows app: winget install Ollama.Ollama" >&2
    exit 1
  fi
  for _ in $(seq 1 20); do up && break; sleep 2; done
fi
up || { echo "Ollama did not start on $URL" >&2; exit 1; }

for model in "$@"; do
  printf 'pull %s: ' "$model"
  curl -sf -m 1800 "$URL/api/pull" -d "{\"model\":\"$model\",\"stream\":false}"
  echo
done
