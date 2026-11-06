#!/usr/bin/env bash
# سكربت رفع التحديثات مباشرة إلى مستودع GitHub الخاص بك:
# https://github.com/mohammedsabryalfiqi/Englishawy.git

set -e

echo "🚀 جاري تجهيز ورفع تحديثات Englishawy v2.4 إلى GitHub..."

if [ ! -d ".git" ]; then
  git init
  git branch -M main
  git remote add origin https://github.com/mohammedsabryalfiqi/Englishawy.git || true
fi

git add bot.py preprocessor.py whatfontis.py dafont.py external_vault.py fonts_vault.json requirements.txt .env.example README.md
git commit -m "feat(v2.4): upgrade font recognition with image preprocessing, external server font vault, smart name normalizer, and interactive Telegram re-analysis"
git push -u origin main

echo "✅ تم تحديث المستودع على GitHub بنجاح!"
