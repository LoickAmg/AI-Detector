from __future__ import annotations

import json
import math
from pathlib import Path

VIEWER = Path(__file__).parent / "static" / "viewer.html"


def json_safe(obj):
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def to_json(result: dict) -> str:
    return json.dumps(json_safe(result), ensure_ascii=False)


def render_html(result: dict) -> str:
    payload = to_json(result).replace("</", "<\\/").replace("<!--", "<\\u0021--")
    script = f"<script>window.__REPORT__ = {payload};</script>\n<script>"
    return VIEWER.read_text(encoding="utf-8").replace("<script>", script, 1)
