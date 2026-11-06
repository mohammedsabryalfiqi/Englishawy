"""
external_vault.py — محرك التخزين الداخلي للخطوط على سيرفر خارجي + المطابقة الهجينة
==================================================================================
يربط البوت بقاعدة بيانات الخطوط الخاصة بك (fonts_vault.json + السيرفر الخارجي)
بحيث يعطي المستخدم رابط تحميل مباشر للخط (.ttf / .otf) من سيرفرك الخارجي بدقة عالية جداً.
"""

import json
import os
from dafont import normalize_font_name


class ExternalFontVault:
    def __init__(self, vault_json_path: str, external_base_url: str = ""):
        self.vault_json_path = vault_json_path
        self.external_base_url = external_base_url.rstrip("/")
        self.fonts = self._load_vault()

    def _load_vault(self) -> list[dict]:
        if not os.path.exists(self.vault_json_path):
            return []
        try:
            with open(self.vault_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except Exception:
            return []

    def find_in_vault_by_name(self, raw_title: str) -> dict | None:
        clean = normalize_font_name(raw_title).lower()
        if not clean:
            return None

        for font in self.fonts:
            f_name = font.get("name", "").lower()
            f_family = font.get("family", "").lower()
            if clean == f_family or clean == f_name or clean in f_name or f_family in clean:
                return font
        return None

    def rank_vault_by_vector(self, extracted_vector: dict, limit: int = 5) -> list[dict]:
        scored = []
        for font in self.fonts:
            vec = font.get("vector_8d", {})
            serif_diff = abs(vec.get("serif_score", 0.1) - extracted_vector.get("serif_score", 0.1))
            contrast_diff = abs(
                vec.get("stroke_contrast", 0.15) - extracted_vector.get("stroke_contrast", 0.15)
            )
            weight_diff = abs(vec.get("weight_score", 600) - extracted_vector.get("weight_score", 600)) / 800.0

            score = 98.5 - (serif_diff * 35.0 + contrast_diff * 30.0 + weight_diff * 20.0)
            confidence = round(max(55.0, min(99.2, score)), 1)
            scored.append((confidence, font))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [{"confidence": conf, "font": f} for conf, f in scored[:limit]]

    def match_and_enrich(
        self,
        whatfontis_matches: list[dict],
        extracted_vector: dict,
        dafont_search_fn,
        limit: int = 5,
    ) -> list[dict]:
        results = []
        seen_families = set()

        # 1. إذا رجعت نتائج من WhatFontIs API، نربط كل نتيجة بمستودع السيرفر الخارجي + DaFont المطبّع
        for idx, m in enumerate(whatfontis_matches[:limit]):
            raw_title = m.get("title", "Unknown Font")
            clean_family = normalize_font_name(raw_title)
            family_key = clean_family.lower()
            if family_key in seen_families:
                continue
            seen_families.add(family_key)

            vault_hit = self.find_in_vault_by_name(raw_title)
            df_info = dafont_search_fn(clean_family)

            base_conf = round(max(78.0, 98.4 - idx * 2.7), 1)
            if vault_hit:
                base_conf = min(99.4, round(base_conf + 1.2, 1))

            results.append(
                {
                    "title": vault_hit["name"] if vault_hit else raw_title,
                    "clean_family": clean_family,
                    "confidence": base_conf,
                    "category": vault_hit.get("category", "English Font") if vault_hit else "Identified",
                    "license_type": m.get("type") or (vault_hit.get("license_type") if vault_hit else "Check License"),
                    "external_server_url": vault_hit.get("external_server_url") if vault_hit else None,
                    "file_format": vault_hit.get("file_format", "TTF / OTF") if vault_hit else None,
                    "key_glyph_notes": vault_hit.get("key_glyph_notes") if vault_hit else None,
                    "whatfontis_url": m.get("url") or (vault_hit.get("whatfontis_url") if vault_hit else None),
                    "dafont": df_info,
                }
            )

        # 2. استكمال النتائج أو العمل الكامل من مستودع السيرفر الخارجي عبر البصمة الهندسية 8D Vector
        if len(results) < limit:
            vector_ranked = self.rank_vault_by_vector(extracted_vector, limit=limit * 2)
            for item in vector_ranked:
                font = item["font"]
                clean_family = normalize_font_name(font["family"])
                if clean_family.lower() in seen_families:
                    continue
                seen_families.add(clean_family.lower())

                results.append(
                    {
                        "title": font["name"],
                        "clean_family": clean_family,
                        "confidence": item["confidence"],
                        "category": font.get("category", "Sans-Serif"),
                        "license_type": font.get("license_type", "Free / OFL"),
                        "external_server_url": font.get("external_server_url"),
                        "file_format": font.get("file_format", "TTF"),
                        "key_glyph_notes": font.get("key_glyph_notes"),
                        "whatfontis_url": font.get("whatfontis_url"),
                        "dafont": {
                            "clean_query": clean_family,
                            "search_url": font.get("dafont_url"),
                            "font_page": font.get("dafont_url"),
                            "download_url": None,
                        },
                    }
                )
                if len(results) >= limit:
                    break

        return results[:limit]
