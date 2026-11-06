#!/usr/bin/env bash
# سكربت رفع تحديثات Englishawy v4 إلى مستودع GitHub الخاص بك:
# https://github.com/mohammedsabryalfiqi/Englishawy.git
# ملاحظة: الأصول الثقيلة (.cache/) مستثناة عبر .gitignore ولا تُرفع.

set -e

echo "🚀 جاري تجهيز ورفع تحديثات Englishawy v4 إلى GitHub..."

if [ ! -d ".git" ]; then
  git init
  git remote add origin https://github.com/mohammedsabryalfiqi/Englishawy.git || true
fi

BRANCH="$(git rev-parse --abbrev-ref HEAD)"

git add .gitignore app.py assets_runtime.py bot.py config.py font_analysis.py font_catalog.py \
        font_store.py owner_auth.py services.py telegram_webhook.py zero_gpu.py \
        pipeline.py preprocessor.py ocr_engine.py font_matcher.py neural_reranker.py \
        comparison_card.py setup_assets.py identify_cli.py font_manifest.json \
        requirements.txt requirements-dev.txt .env.example README.md push_to_github.sh tests
git commit -m "feat(v4): Gradio ZeroGPU Space, owner font library and Telegram webhook" || true
git push -u origin "$BRANCH"

echo "✅ تم رفع التحديثات على الفرع: $BRANCH"
