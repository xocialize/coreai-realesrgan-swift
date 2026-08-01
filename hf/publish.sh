#!/bin/zsh
# Publish the .aimodel assets + model card to HF coreai-community.
# Prerequisites (both were missing on 2026-07-31, blocking the first attempt):
#   1. coreai-community membership APPROVED (huggingface.co/coreai-community — role must not be None)
#   2. The fine-grained token granted repo.write on coreai-community
#      (huggingface.co/settings/tokens -> edit -> add organization: coreai-community)
set -e
cd "$(dirname "$0")"
REPO=coreai-community/Real-ESRGAN-CoreAI
hf repo create $REPO --repo-type model || true
STAGE=$(mktemp -d)
cp README.md "$STAGE/"
cp ../scripts/srvgg_export.py "$STAGE/"
for m in realesr_general_x4v3 realesr_general_wdn_x4v3 realesr_animevideov3; do
  cp -R "../Sources/RealESRGANCoreAI/Resources/${m}_float16_static128.aimodel" "$STAGE/"
done
hf upload $REPO "$STAGE" . --repo-type model --commit-message "Real-ESRGAN SRVGG fp16 static128 — first SR model in coreai-community"
rm -rf "$STAGE"
echo "published: https://huggingface.co/$REPO"
