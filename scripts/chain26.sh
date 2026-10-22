#!/bin/bash
# Run 26: after implant B finishes, copy it to adapters/, then train D and copy it (one process at a time).
cd "$(dirname "$0")"
while kill -0 707 2>/dev/null; do sleep 20; done
[ -f /tmp/implant_B/adapter_model.safetensors ] && mkdir -p adapters/implant_B && cp /tmp/implant_B/adapter_config.json /tmp/implant_B/adapter_model.safetensors /tmp/implant_B/stage1.json adapters/implant_B/ 2>/dev/null
echo "B copied $(date -u +%H:%M:%SZ)"
HF_HUB_OFFLINE=1 /usr/local/bin/python3 implant.py train /tmp/implant_D --facts D > /tmp/implant_D.log 2>&1
[ -f /tmp/implant_D/adapter_model.safetensors ] && mkdir -p adapters/implant_D && cp /tmp/implant_D/adapter_config.json /tmp/implant_D/adapter_model.safetensors /tmp/implant_D/stage1.json adapters/implant_D/ 2>/dev/null
echo "D copied $(date -u +%H:%M:%SZ)"
