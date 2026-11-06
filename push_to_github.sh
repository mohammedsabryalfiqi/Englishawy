#!/usr/bin/env bash
# سكربت رفع تحديثات Englishawy v3 إلى مستودع GitHub الخاص بك:
# https://github.com/mohammedsabryalfiqi/Englishawy.git
# ملاحظة: الأصول الثقيلة (.cache/) مستثناة عبر .gitignore ولا تُرفع.

set -e

echo "🚀 جاري تجهيز ورفع تحديثات Englishawy v3 إلى GitHub..."

if [ ! -d ".git" ]; then
  git init
  git remote add origin https://github.com/mohammedsabryalfiqi/Englishawy.git || true
fi

BRANCH="$(git rev-parse --abbrev-ref HEAD)"

git add .gitignore bot.py pipeline.py preprocessor.py ocr_engine.py font_matcher.py \
        neural_reranker.py comparison_card.py setup_assets.py identify_cli.py \
        font_manifest.json requirements.txt .env.example README.md push_to_github.sh
git commit -m "feat(v3): local AI font identification (Tesseract voting + Google Fonts shape match + DINOv2 rerank) with visual comparison card, no download links" || true
git push -u origin "$BRANCH"

echo "✅ تم رفع التحديثات على الفرع: $BRANCH"
