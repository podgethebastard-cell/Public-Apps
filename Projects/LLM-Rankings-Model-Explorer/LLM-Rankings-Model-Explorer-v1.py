# llm_rankings_app.py
# =============================================================================
# LLM Rankings & Model Explorer (Streamlit) — FIXED
# -----------------------------------------------------------------------------
# Fixes in this revision:
# ✅ Prevent AttributeError when a column is missing (e.g., Aider disabled or failed fetch)
#    - view.get("col") can return None -> None.apply crashes
#    - Now we always use a Series fallback with the correct length.
# ✅ Same safety added for all optional metric columns (Aider/LCB/SWE/ctx/price/etc.)
#
# Features included:
# ✅ “Top Picks” panel (best overall / best coding / best budget / best open-weights)
# ✅ Multipage feel in a SINGLE .py (Rankings → click model → dedicated detail page with Back/Next)
# ✅ More leaderboards (incl. SWE-bench + LiveCodeBench) via plug-in “source adapters”
# =============================================================================

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import requests
import streamlit as st

# =========================
# Page config + basic style
# =========================
st.set_page_config(
    page_title="LLM Rankings • Model Explorer",
    page_icon="🧠",
    layout="wide",
)

st.markdown(
    """
<style>
.block-container { padding-top: 1.0rem; padding-bottom: 3rem; }
small, .muted { color: rgba(255,255,255,0.70); }
hr { border: none; border-top: 1px solid rgba(255,255,255,0.10); margin: 0.75rem 0; }
.card { border: 1px solid rgba(255,255,255,0.12); border-radius: 16px; padding: 14px; background: rgba(255,255,255,0.03); }
.card h4 { margin: 0 0 6px 0; font-size: 15px; }
.card .kpi { font-size: 22px; font-weight: 750; margin-top: 2px; }
.card .sub { font-size: 12px; opacity: 0.85; }
.table-wrap { border: 1px solid rgba(255,255,255,0.10); border-radius: 16px; overflow: hidden; }
.table-wrap table { width: 100%; border-collapse: collapse; font-size: 13px; }
.table-wrap th, .table-wrap td { padding: 10px; border-bottom: 1px solid rgba(255,255,255,0.08); vertical-align: middle; }
.table-wrap th { background: rgba(255,255,255,0.04); text-align: left; font-weight: 650; }
.table-wrap tr:hover td { background: rgba(255,255,255,0.03); }
.badge { display: inline-block; padding: 2px 8px; border-radius: 999px; border: 1px solid rgba(255,255,255,0.16);
         background: rgba(255,255,255,0.04); font-size: 12px; margin-right: 6px; }
a.inline { color: inherit; text-decoration: none; border-bottom: 1px dotted rgba(255,255,255,0.35); }
a.inline:hover { border-bottom-color: rgba(255,255,255,0.7); }
</style>
""",
    unsafe_allow_html=True,
)

# ==========
# Utilities
# ==========
USER_AGENT = "LLM-Rankings-Streamlit/1.1"
DEFAULT_TIMEOUT = 20


def _now_ts() -> float:
    return time.time()


def normalize_key(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9\.\-\s_:/]", "", s)
    return s


def safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float, np.integer, np.floating)):
            if isinstance(x, float) and np.isnan(x):
                return None
            return float(x)
        s = str(x).strip()
        if s == "" or s.lower() in {"nan", "none", "null", "n/a"}:
            return None
        s = s.replace("%", "").replace(",", "")
        return float(s)
    except Exception:
        return None


def series_or_nan(df: pd.DataFrame, col: str) -> pd.Series:
    """Return df[col] if exists, else a NaN Series of correct length."""
    if col in df.columns:
        return df[col]
    return pd.Series([np.nan] * len(df), index=df.index)


def http_get_json(url: str) -> Any:
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    return r.json()


def http_get_text(url: str) -> str:
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    return r.text


def github_contents(owner: str, repo: str, path: str = "", ref: str = "main") -> List[Dict[str, Any]]:
    api = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={ref}"
    r = requests.get(api, headers={"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"}, timeout=DEFAULT_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict):
        return [data]
    return data


def github_raw_url(owner: str, repo: str, path: str, ref: str = "main") -> str:
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{ref}/{path}"


def pct(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.2f}%"


def money_per_million(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"${x:.2f} / 1M"


# ======================
# Baseline directory
# ======================
BASE_MODELS: List[Dict[str, Any]] = [
    {"display_name": "GPT-5", "provider": "OpenAI", "open_weights": False, "links": {"provider": "https://openai.com", "search": "https://openai.com/search?q=gpt-5"}},
    {"display_name": "GPT-4.1", "provider": "OpenAI", "open_weights": False, "links": {"provider": "https://openai.com", "search": "https://openai.com/search?q=gpt-4.1"}},
    {"display_name": "o3", "provider": "OpenAI", "open_weights": False, "links": {"provider": "https://openai.com", "search": "https://openai.com/search?q=o3"}},
    {"display_name": "o4-mini", "provider": "OpenAI", "open_weights": False, "links": {"provider": "https://openai.com", "search": "https://openai.com/search?q=o4-mini"}},
    {"display_name": "Claude Opus", "provider": "Anthropic", "open_weights": False, "links": {"provider": "https://www.anthropic.com"}},
    {"display_name": "Claude Sonnet", "provider": "Anthropic", "open_weights": False, "links": {"provider": "https://www.anthropic.com"}},
    {"display_name": "Gemini 2.5 Pro", "provider": "Google", "open_weights": False, "links": {"provider": "https://ai.google"}},
    {"display_name": "Grok-4", "provider": "xAI", "open_weights": False, "links": {"provider": "https://x.ai"}},
    {"display_name": "DeepSeek-R1", "provider": "DeepSeek", "open_weights": True, "links": {"provider": "https://www.deepseek.com"}},
    {"display_name": "Kimi (Moonshot)", "provider": "Moonshot AI", "open_weights": False, "links": {"provider": "https://www.moonshot.cn"}},
    {"display_name": "Llama 3.1 405B", "provider": "Meta", "open_weights": True, "links": {"provider": "https://ai.meta.com/llama/"}},
    {"display_name": "Qwen 2.5", "provider": "Alibaba", "open_weights": True, "links": {"provider": "https://qwenlm.ai"}},
    {"display_name": "Mistral Large", "provider": "Mistral", "open_weights": False, "links": {"provider": "https://mistral.ai"}},
]

ALIASES: Dict[str, str] = {
    normalize_key("gpt-5"): normalize_key("gpt 5"),
    normalize_key("o4 mini"): normalize_key("o4-mini"),
    normalize_key("deepseek r1"): normalize_key("deepseek-r1"),
    normalize_key("kimi"): normalize_key("kimi (moonshot)"),
}


def canonical_key(name: str) -> str:
    k = normalize_key(name)
    return ALIASES.get(k, k)


# =====================
# Source adapter system
# =====================
@dataclass
class SourceResult:
    df: pd.DataFrame
    meta: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class SourceAdapter:
    id: str = "base"
    label: str = "Base"
    homepage: str = ""
    description: str = ""

    def fetch(self, cfg: Dict[str, Any]) -> SourceResult:
        raise NotImplementedError


class BaselineDirectoryAdapter(SourceAdapter):
    id = "baseline"
    label = "Baseline Directory"
    description = "Built-in list so the app always runs."

    def fetch(self, cfg: Dict[str, Any]) -> SourceResult:
        rows = []
        for m in BASE_MODELS:
            rows.append(
                {
                    "model_key": canonical_key(m["display_name"]),
                    "display_name": m["display_name"],
                    "provider": m.get("provider"),
                    "open_weights": bool(m.get("open_weights", False)),
                    "link_provider": (m.get("links") or {}).get("provider"),
                }
            )
        return SourceResult(df=pd.DataFrame(rows))


class OpenRouterModelsAdapter(SourceAdapter):
    id = "openrouter_models"
    label = "OpenRouter Models (metadata + pricing)"
    homepage = "https://openrouter.ai"
    description = "Fetch model list, context length and per-token pricing from OpenRouter."

    def fetch(self, cfg: Dict[str, Any]) -> SourceResult:
        try:
            url = "https://openrouter.ai/api/v1/models"
            data = http_get_json(url)
            rows = []
            for item in data.get("data", []) if isinstance(data, dict) else []:
                mid = item.get("id")
                name = item.get("name") or mid
                prov = (mid.split("/")[0] if isinstance(mid, str) and "/" in mid else None) or item.get("provider") or "OpenRouter"
                ctx = safe_float(item.get("context_length"))
                pricing = item.get("pricing") or {}
                p_in = safe_float(pricing.get("prompt"))
                p_out = safe_float(pricing.get("completion"))
                p_in_m = (p_in * 1_000_000) if p_in is not None else None
                p_out_m = (p_out * 1_000_000) if p_out is not None else None

                open_guess = False
                if isinstance(mid, str):
                    open_guess = any(kw in mid.lower() for kw in ["llama", "mistral", "qwen", "deepseek", "mixtral", "gemma", "phi", "olmo", "yi-"])

                rows.append(
                    {
                        "model_key": canonical_key(str(name)),
                        "display_name": str(name),
                        "provider": prov,
                        "openrouter_id": mid,
                        "context_length": ctx,
                        "price_in_1m_usd": p_in_m,
                        "price_out_1m_usd": p_out_m,
                        "open_weights": open_guess,
                        "link_openrouter": f"https://openrouter.ai/models/{mid}" if mid else None,
                    }
                )
            return SourceResult(df=pd.DataFrame(rows), meta={"url": url, "rows": len(rows)})
        except Exception as e:
            return SourceResult(df=pd.DataFrame(), error=f"{type(e).__name__}: {e}")


class AiderLeaderboardAdapter(SourceAdapter):
    id = "aider"
    label = "Aider Coding Leaderboard"
    homepage = "https://aider.chat/docs/leaderboards/"
    description = "Best-effort HTML table parse of Aider coding leaderboard."

    def fetch(self, cfg: Dict[str, Any]) -> SourceResult:
        try:
            url = self.homepage
            html = http_get_text(url)
            tables = pd.read_html(html)
            best = None
            for t in tables:
                cols = [c.lower() for c in t.columns.astype(str)]
                if any("model" == c or "model" in c for c in cols) and any("pass" in c for c in cols):
                    best = t
                    break
            if best is None:
                return SourceResult(df=pd.DataFrame(), error="No suitable table found on Aider page.")

            df = best.copy()
            model_col = next((c for c in df.columns if str(c).lower().strip() == "model"), None) or df.columns[0]
            pass_cols = [c for c in df.columns if "pass" in str(c).lower()]
            pass2_col = next((c for c in pass_cols if "@2" in str(c).lower()), None) or (pass_cols[0] if pass_cols else None)

            rows = []
            for _, r in df.iterrows():
                name = str(r.get(model_col, "")).strip()
                if not name:
                    continue
                rows.append(
                    {
                        "model_key": canonical_key(name),
                        "display_name": name,
                        "aider_pass_metric": safe_float(r.get(pass2_col)) if pass2_col else None,
                        "aider_metric_name": str(pass2_col) if pass2_col else None,
                        "link_aider": url,
                    }
                )
            out = pd.DataFrame(rows)
            if out.empty:
                return SourceResult(df=out, error="Parsed 0 rows from Aider table.")
            out = out.groupby("model_key", as_index=False).agg({"display_name": "first", "aider_pass_metric": "max", "aider_metric_name": "first", "link_aider": "first"})
            return SourceResult(df=out, meta={"url": url, "rows": len(out)})
        except Exception as e:
            return SourceResult(df=pd.DataFrame(), error=f"{type(e).__name__}: {e}")


class SweBenchLeaderboardsAdapter(SourceAdapter):
    id = "swebench"
    label = "SWE-bench Leaderboards"
    homepage = "https://www.swebench.com/"
    description = "Fetch SWE-bench leaderboards.json from the official site repo and extract resolved %."

    def fetch(self, cfg: Dict[str, Any]) -> SourceResult:
        try:
            owner = "SWE-bench"
            repo = "swe-bench.github.io"
            ref = cfg.get("ref", "master")
            path = "data/leaderboards.json"
            url = github_raw_url(owner, repo, path, ref=ref)
            data = http_get_json(url)

            rows = []
            for lb in data.get("leaderboards", []):
                lb_name = str(lb.get("name", "")).strip()
                for item in lb.get("results", []):
                    if item.get("warning"):
                        continue
                    model = item.get("model") or item.get("name") or item.get("system") or ""
                    resolved = safe_float(item.get("resolved"))
                    rows.append(
                        {
                            "model_key": canonical_key(str(model)),
                            "display_name": str(model),
                            f"swebench_{normalize_key(lb_name)}_resolved_pct": resolved,
                            "link_swebench": self.homepage,
                        }
                    )

            df = pd.DataFrame(rows)
            if df.empty:
                return SourceResult(df=df, error="No rows extracted (schema may have changed).")

            agg: Dict[str, Any] = {"display_name": "first", "link_swebench": "first"}
            for c in df.columns:
                if c.startswith("swebench_") and c.endswith("_resolved_pct"):
                    agg[c] = "max"

            out = df.groupby("model_key", as_index=False).agg(agg)
            return SourceResult(df=out, meta={"url": url, "ref": ref, "rows": len(out)})
        except Exception as e:
            return SourceResult(df=pd.DataFrame(), error=f"{type(e).__name__}: {e}")


class LiveCodeBenchAdapter(SourceAdapter):
    id = "livecodebench"
    label = "LiveCodeBench (auto-discovery)"
    homepage = "https://livecodebench.github.io/"
    description = "Auto-discovers a JSON/CSV leaderboard file in the official repo and extracts Pass@1."

    def fetch(self, cfg: Dict[str, Any]) -> SourceResult:
        try:
            owner = "LiveCodeBench"
            repo = "livecodebench.github.io"
            ref = cfg.get("ref", "main")

            candidate_paths = [
                "public/leaderboard.json",
                "public/leaderboard_data.json",
                "public/leaderboard.csv",
                "public/data/leaderboard.json",
                "public/data/leaderboard.csv",
                "public/data/leaderboard_data.json",
            ]

            found_url = None
            found_path = None

            def try_path(p: str) -> bool:
                nonlocal found_url, found_path
                try:
                    u = github_raw_url(owner, repo, p, ref=ref)
                    r = requests.get(u, headers={"User-Agent": USER_AGENT}, timeout=10)
                    if r.status_code == 200 and len(r.content) > 10:
                        found_url, found_path = u, p
                        return True
                except Exception:
                    return False
                return False

            for p in candidate_paths:
                if try_path(p):
                    break

            if found_url is None:
                scan_paths = ["public", "public/data"]
                assets: List[str] = []
                for sp in scan_paths:
                    try:
                        items = github_contents(owner, repo, sp, ref=ref)
                        for it in items:
                            if it.get("type") == "file":
                                assets.append(f"{sp}/{it.get('name','')}")
                    except Exception:
                        continue

                leaderboardish = [p for p in assets if ("leaderboard" in p.lower() or "lcb" in p.lower()) and (p.lower().endswith(".json") or p.lower().endswith(".csv"))]
                for p in leaderboardish:
                    if try_path(p):
                        break

            if found_url is None:
                return SourceResult(df=pd.DataFrame(), error="Could not auto-discover a leaderboard asset in the repo (repo structure may have changed).")

            if found_url.endswith(".json"):
                data = http_get_json(found_url)
                rows_in = None
                if isinstance(data, list):
                    rows_in = data
                elif isinstance(data, dict):
                    for k in ["rows", "data", "leaderboard", "results"]:
                        if k in data and isinstance(data[k], list):
                            rows_in = data[k]
                            break
                if not rows_in:
                    return SourceResult(df=pd.DataFrame(), error=f"JSON schema not recognized for {found_path}")

                rows = []
                for rr in rows_in:
                    if not isinstance(rr, dict):
                        continue
                    name = rr.get("model") or rr.get("Model") or rr.get("name") or rr.get("system") or ""
                    pass1 = safe_float(rr.get("pass@1")) or safe_float(rr.get("Pass@1")) or safe_float(rr.get("pass1")) or safe_float(rr.get("score"))
                    if not str(name).strip():
                        continue
                    rows.append(
                        {
                            "model_key": canonical_key(str(name)),
                            "display_name": str(name),
                            "livecodebench_pass1": pass1,
                            "link_livecodebench": self.homepage,
                        }
                    )

                df = pd.DataFrame(rows)
                if df.empty:
                    return SourceResult(df=df, error=f"No rows extracted from {found_path}")
                out = df.groupby("model_key", as_index=False).agg({"display_name": "first", "livecodebench_pass1": "max", "link_livecodebench": "first"})
                return SourceResult(df=out, meta={"asset": found_path, "url": found_url, "rows": len(out)})

            # CSV
            text = http_get_text(found_url)
            dfc = pd.read_csv(pd.io.common.StringIO(text))
            cols_low = {c.lower(): c for c in dfc.columns}
            model_col = cols_low.get("model") or cols_low.get("name") or cols_low.get("system") or dfc.columns[0]
            pass1_col = None
            for k in ["pass@1", "pass1", "score"]:
                if k in cols_low:
                    pass1_col = cols_low[k]
                    break

            rows = []
            for _, rr in dfc.iterrows():
                name = str(rr.get(model_col, "")).strip()
                if not name:
                    continue
                rows.append(
                    {
                        "model_key": canonical_key(name),
                        "display_name": name,
                        "livecodebench_pass1": safe_float(rr.get(pass1_col)) if pass1_col else None,
                        "link_livecodebench": self.homepage,
                    }
                )
            out = pd.DataFrame(rows)
            if out.empty:
                return SourceResult(df=out, error=f"No rows extracted from {found_path}")
            out = out.groupby("model_key", as_index=False).agg({"display_name": "first", "livecodebench_pass1": "max", "link_livecodebench": "first"})
            return SourceResult(df=out, meta={"asset": found_path, "url": found_url, "rows": len(out)})

        except Exception as e:
            return SourceResult(df=pd.DataFrame(), error=f"{type(e).__name__}: {e}")


# =========================
# Merge + ranking
# =========================
def merge_sources(results: List[Tuple[SourceAdapter, SourceResult]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    status_rows = []
    dfs = []

    for ad, res in results:
        status_rows.append(
            {"Source": ad.label, "Adapter ID": ad.id, "OK": res.error is None, "Rows": 0 if res.df is None else len(res.df), "Error": res.error or "", "Homepage": ad.homepage or ""}
        )
        if res.df is not None and not res.df.empty and "model_key" in res.df.columns:
            d = res.df.copy()
            d["model_key"] = d["model_key"].astype(str).map(canonical_key)
            dfs.append(d)

    if not dfs:
        merged = pd.DataFrame(columns=["model_key", "display_name"])
    else:
        merged = dfs[0]
        for d in dfs[1:]:
            merged = merged.merge(d, on="model_key", how="outer", suffixes=("", "_r"))
            if "display_name" in merged.columns and "display_name_r" in merged.columns:
                merged["display_name"] = merged["display_name"].fillna(merged["display_name_r"])
                merged = merged.drop(columns=["display_name_r"])

    if "open_weights" not in merged.columns:
        merged["open_weights"] = False
    else:
        merged["open_weights"] = merged["open_weights"].fillna(False).astype(bool)

    pin = pd.to_numeric(series_or_nan(merged, "price_in_1m_usd"), errors="coerce")
    pout = pd.to_numeric(series_or_nan(merged, "price_out_1m_usd"), errors="coerce")
    merged["avg_price_1m_usd"] = (pin + pout) / 2.0

    merged["display_name"] = series_or_nan(merged, "display_name").astype(str).replace("nan", "")
    merged.loc[merged["display_name"].astype(str).str.strip() == "", "display_name"] = merged["model_key"]

    return merged, pd.DataFrame(status_rows)


def zscore(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mu = s.mean(skipna=True)
    sd = s.std(skipna=True)
    if sd is None or sd == 0 or np.isnan(sd):
        return pd.Series([0.0] * len(s), index=s.index)
    return (s - mu) / sd


def compute_composite(df: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    total = pd.Series([0.0] * len(out), index=out.index, dtype=float)
    used_any = False
    for col, w in weights.items():
        if w == 0:
            continue
        zs = zscore(series_or_nan(out, col)).fillna(0.0)
        total = total + (zs * float(w))
        used_any = True
    out["composite_score"] = total if used_any else 0.0
    out = out.sort_values(["composite_score", "display_name"], ascending=[False, True]).reset_index(drop=True)
    out["rank_overall"] = np.arange(1, len(out) + 1)
    return out


# =========================
# Navigation (query params)
# =========================
def qp_get(key: str, default: str = "") -> str:
    try:
        v = st.query_params.get(key)
        if isinstance(v, list):
            return v[0] if v else default
        return v if v is not None else default
    except Exception:
        try:
            v = st.experimental_get_query_params().get(key, [default])
            return v[0] if v else default
        except Exception:
            return default


def qp_set(**kwargs: str) -> None:
    try:
        st.query_params.clear()
        for k, v in kwargs.items():
            st.query_params[k] = v
    except Exception:
        st.experimental_set_query_params(**kwargs)


# =========================
# UI helpers
# =========================
def badge(text: str) -> str:
    return f'<span class="badge">{text}</span>'


def html_table(df: pd.DataFrame, max_rows: int = 50) -> str:
    view = df.head(max_rows).copy()
    cols = list(view.columns)

    def cell(v: Any) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        return str(v)

    html = ['<div class="table-wrap"><table>']
    html.append("<thead><tr>")
    for c in cols:
        html.append(f"<th>{c}</th>")
    html.append("</tr></thead><tbody>")
    for _, r in view.iterrows():
        html.append("<tr>")
        for c in cols:
            html.append(f"<td>{cell(r.get(c))}</td>")
        html.append("</tr>")
    html.append("</tbody></table></div>")
    return "\n".join(html)


def external_search_link(q: str) -> str:
    return f"https://www.google.com/search?q={quote_plus(q)}"


# =========================
# Sidebar
# =========================
st.sidebar.title("🧠 LLM Explorer")

page = qp_get("page", "rankings")
if page not in {"rankings", "model", "sources"}:
    page = "rankings"

nav = st.sidebar.radio("Navigate", options=["Rankings", "Model Detail", "Sources"], index={"rankings": 0, "model": 1, "sources": 2}[page])
if nav == "Rankings" and page != "rankings":
    qp_set(page="rankings")
    st.rerun()
if nav == "Model Detail" and page != "model":
    qp_set(page="model", model=qp_get("model", ""))
    st.rerun()
if nav == "Sources" and page != "sources":
    qp_set(page="sources")
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("Data Sources")

ALL_ADAPTERS: List[SourceAdapter] = [
    BaselineDirectoryAdapter(),
    OpenRouterModelsAdapter(),
    AiderLeaderboardAdapter(),
    SweBenchLeaderboardsAdapter(),
    LiveCodeBenchAdapter(),
]

default_enabled = {"baseline": True, "openrouter_models": True, "aider": True, "swebench": True, "livecodebench": True}
enabled: Dict[str, bool] = {}
adapter_cfg: Dict[str, Dict[str, Any]] = {}

for ad in ALL_ADAPTERS:
    enabled[ad.id] = st.sidebar.checkbox(ad.label, value=default_enabled.get(ad.id, False), help=ad.description)
    cfg: Dict[str, Any] = {}
    if ad.id in {"swebench", "livecodebench"}:
        cfg["ref"] = st.sidebar.text_input(
            f"{ad.label} • Git ref",
            value="master" if ad.id == "swebench" else "main",
            help="Branch/tag/commit used for GitHub raw fetch.",
            key=f"ref::{ad.id}",
        ).strip() or ("master" if ad.id == "swebench" else "main")
    adapter_cfg[ad.id] = cfg

st.sidebar.markdown("---")
st.sidebar.subheader("Weights")

with st.sidebar.expander("Composite score settings", expanded=True):
    w_aider = st.slider("Coding (Aider)", 0.0, 5.0, 3.0, 0.5)
    w_lcb = st.slider("LiveCodeBench Pass@1", 0.0, 5.0, 3.0, 0.5)
    w_swe_v = st.slider("SWE-bench Verified resolved %", 0.0, 5.0, 3.0, 0.5)
    w_ctx = st.slider("Context length", 0.0, 5.0, 1.0, 0.5)
    w_price = st.slider("Avg price ($/1M) (lower better)", -5.0, 0.0, -2.0, 0.5)

weights = {
    "aider_pass_metric": float(w_aider),
    "livecodebench_pass1": float(w_lcb),
    "swebench_verified_resolved_pct": float(w_swe_v),
    "context_length": float(w_ctx),
    "avg_price_1m_usd": float(w_price),
}

st.sidebar.markdown("---")
st.sidebar.subheader("Filters")
only_open = st.sidebar.checkbox("Open-weights only", value=False)
max_price = st.sidebar.number_input("Max avg price ($/1M)", min_value=0.0, value=0.0, step=0.5, help="0 = no filter")
min_ctx = st.sidebar.number_input("Min context length", min_value=0, value=0, step=1024, help="0 = no filter")
search = st.sidebar.text_input("Search models", value="")
top_n = st.sidebar.slider("Show top N", 10, 200, 50, 10)

# =========================
# Fetch sources (with session cache)
# =========================
def fetch_with_cache(adapter: SourceAdapter, cfg: Dict[str, Any]) -> SourceResult:
    key = f"adapter_cache::{adapter.id}::{json.dumps(cfg, sort_keys=True)}"
    ttl_s = 60 * 30
    cache = st.session_state.setdefault("_adapter_cache", {})
    item = cache.get(key)
    if item and (_now_ts() - item["ts"] < ttl_s):
        return item["res"]
    res = adapter.fetch(cfg)
    cache[key] = {"ts": _now_ts(), "res": res}
    return res


active_adapters = [ad for ad in ALL_ADAPTERS if enabled.get(ad.id, False)]
results: List[Tuple[SourceAdapter, SourceResult]] = []
with st.spinner("Fetching sources…"):
    for ad in active_adapters:
        results.append((ad, fetch_with_cache(ad, adapter_cfg.get(ad.id, {}))))

merged, status_df = merge_sources(results)

# SWE Verified column best-effort
if "swebench_verified_resolved_pct" not in merged.columns:
    cand = [c for c in merged.columns if c.startswith("swebench_") and "verified" in c and c.endswith("_resolved_pct")]
    if cand:
        merged["swebench_verified_resolved_pct"] = merged[cand].max(axis=1)

ranked = compute_composite(merged, weights=weights)

# Filters
f = ranked.copy()
if only_open:
    f = f[f.get("open_weights", False) == True].copy()
if max_price and max_price > 0:
    f = f[pd.to_numeric(series_or_nan(f, "avg_price_1m_usd"), errors="coerce").fillna(np.inf) <= float(max_price)].copy()
if min_ctx and min_ctx > 0:
    f = f[pd.to_numeric(series_or_nan(f, "context_length"), errors="coerce").fillna(0) >= float(min_ctx)].copy()
if search.strip():
    s = search.strip().lower()
    f = f[
        f["display_name"].astype(str).str.lower().str.contains(s)
        | series_or_nan(f, "provider").astype(str).str.lower().str.contains(s)
        | f["model_key"].astype(str).str.lower().str.contains(s)
    ].copy()

f = f.sort_values(["rank_overall"], ascending=True).reset_index(drop=True)
st.session_state["last_rank_list"] = f["display_name"].tolist()

# =========================
# Top Picks
# =========================
def top_pick_best_overall(df: pd.DataFrame) -> Optional[pd.Series]:
    return df.iloc[0] if len(df) else None


def top_pick_best_coding(df: pd.DataFrame) -> Optional[pd.Series]:
    v = pd.to_numeric(series_or_nan(df, "aider_pass_metric"), errors="coerce")
    d = df.copy()
    d["_v"] = v
    d = d.sort_values(["_v", "rank_overall"], ascending=[False, True]).dropna(subset=["_v"])
    return d.iloc[0] if len(d) else None


def top_pick_best_budget(df: pd.DataFrame) -> Optional[pd.Series]:
    v = pd.to_numeric(series_or_nan(df, "avg_price_1m_usd"), errors="coerce")
    d = df.copy()
    d["_v"] = v
    d = d.sort_values(["_v", "rank_overall"], ascending=[True, True]).dropna(subset=["_v"])
    return d.iloc[0] if len(d) else None


def top_pick_best_open(df: pd.DataFrame) -> Optional[pd.Series]:
    d = df[df.get("open_weights", False) == True].copy()
    d = d.sort_values(["rank_overall"], ascending=True)
    return d.iloc[0] if len(d) else None


def render_top_picks(df: pd.DataFrame) -> None:
    best_overall = top_pick_best_overall(df)
    best_coding = top_pick_best_coding(df)
    best_budget = top_pick_best_budget(df)
    best_open = top_pick_best_open(df)

    c1, c2, c3, c4 = st.columns(4)

    def card(col, title: str, item: Optional[pd.Series], kpi: str, sub: str, btn_key: str):
        with col:
            st.markdown(
                f"""
<div class="card">
  <h4>{title}</h4>
  <div class="kpi">{kpi}</div>
  <div class="sub">{sub}</div>
</div>
""",
                unsafe_allow_html=True,
            )
            if item is not None:
                if st.button("View model →", use_container_width=True, key=btn_key):
                    qp_set(page="model", model=str(item["display_name"]))
                    st.rerun()

    card(c1, "🏆 Best overall", best_overall, (str(best_overall["display_name"]) if best_overall is not None else "—"), ("Composite rank #1" if best_overall is not None else "No models"), "tp_overall")
    card(c2, "👨‍💻 Best coding", best_coding, (str(best_coding["display_name"]) if best_coding is not None else "—"), ("Top Aider in view" if best_coding is not None else "Aider missing"), "tp_coding")
    card(c3, "💸 Best budget", best_budget, (str(best_budget["display_name"]) if best_budget is not None else "—"), (f"Avg {money_per_million(best_budget.get('avg_price_1m_usd'))}" if best_budget is not None else "Pricing missing"), "tp_budget")
    card(c4, "🔓 Best open-weights", best_open, (str(best_open["display_name"]) if best_open is not None else "—"), ("Top open-weights by composite" if best_open is not None else "None in view"), "tp_open")


# =========================
# Pages
# =========================
def render_rankings(df: pd.DataFrame) -> None:
    st.title("📊 LLM Rankings")
    st.caption("Click a model name to open its detail page. Optional sources can fail safely without breaking the UI.")

    render_top_picks(df)
    st.markdown("---")

    view = df.copy()

    # IMPORTANT FIX:
    # Never call .apply on df.get("col") because it may return None.
    view["Open"] = series_or_nan(view, "open_weights").apply(lambda x: "✅" if bool(x) else "")
    view["Avg $/1M"] = series_or_nan(view, "avg_price_1m_usd").apply(lambda x: money_per_million(safe_float(x)))
    view["Ctx"] = series_or_nan(view, "context_length").apply(lambda x: "—" if safe_float(x) is None else f"{int(float(x)):,}")
    view["Aider"] = series_or_nan(view, "aider_pass_metric").apply(lambda x: "—" if safe_float(x) is None else f"{safe_float(x):.2f}")
    view["LCB Pass@1"] = series_or_nan(view, "livecodebench_pass1").apply(lambda x: "—" if safe_float(x) is None else f"{safe_float(x):.2f}")
    view["SWE-Verified"] = series_or_nan(view, "swebench_verified_resolved_pct").apply(lambda x: "—" if safe_float(x) is None else f"{safe_float(x):.2f}%")

    rows = []
    for _, r in view.head(top_n).iterrows():
        name = str(r["display_name"])
        link = f'?page=model&model={quote_plus(name)}'
        prov = str(r.get("provider") or "—")
        comp = safe_float(r.get("composite_score"))
        comp_s = f"{comp:.2f}" if comp is not None else "—"
        rows.append(
            {
                "Rank": int(r["rank_overall"]),
                "Model": f'<a class="inline" href="{link}">{name}</a>',
                "Provider": prov,
                "Open": str(r.get("Open")),
                "Composite": comp_s,
                "Aider": str(r.get("Aider")),
                "LCB Pass@1": str(r.get("LCB Pass@1")),
                "SWE-Verified": str(r.get("SWE-Verified")),
                "Ctx": str(r.get("Ctx")),
                "Avg $/1M": str(r.get("Avg $/1M")),
            }
        )

    tdf = pd.DataFrame(rows)
    st.markdown(html_table(tdf, max_rows=top_n), unsafe_allow_html=True)


def render_model_detail(df_all: pd.DataFrame) -> None:
    model_name = qp_get("model", "").strip()
    st.title("🧾 Model Detail")

    if not model_name:
        st.info("Pick a model from Rankings.")
        return

    row = df_all[df_all["display_name"].astype(str) == model_name]
    if row.empty:
        ck = canonical_key(model_name)
        row = df_all[df_all["model_key"].astype(str) == ck]
    if row.empty:
        st.warning("Model not found (try clearing filters).")
        if st.button("Back to rankings"):
            qp_set(page="rankings")
            st.rerun()
        return

    r = row.iloc[0]

    rank_list = st.session_state.get("last_rank_list") or df_all.sort_values("rank_overall")["display_name"].tolist()
    idx = rank_list.index(model_name) if model_name in rank_list else None
    prev_name = rank_list[idx - 1] if (idx is not None and idx > 0) else None
    next_name = rank_list[idx + 1] if (idx is not None and idx < len(rank_list) - 1) else None

    navc1, navc2 = st.columns([1.2, 2.0])
    with navc1:
        if st.button("⬅ Back", use_container_width=True):
            qp_set(page="rankings")
            st.rerun()
    with navc2:
        pcol, ncol = st.columns(2)
        with pcol:
            if prev_name and st.button("Prev", use_container_width=True):
                qp_set(page="model", model=prev_name)
                st.rerun()
        with ncol:
            if next_name and st.button("Next", use_container_width=True):
                qp_set(page="model", model=next_name)
                st.rerun()

    st.markdown("---")
    st.subheader(str(r["display_name"]))

    meta_bits = []
    if r.get("provider"):
        meta_bits.append(badge(f"Provider: {r.get('provider')}"))
    if bool(r.get("open_weights", False)):
        meta_bits.append(badge("Open-weights"))
    if safe_float(r.get("context_length")) is not None:
        meta_bits.append(badge(f"Context: {int(float(r.get('context_length'))):,}"))
    if safe_float(r.get("avg_price_1m_usd")) is not None:
        meta_bits.append(badge(f"Avg price: {money_per_million(r.get('avg_price_1m_usd'))}"))
    st.markdown(" ".join(meta_bits), unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Overall rank", int(r.get("rank_overall", 0)))
    k2.metric("Composite", f"{safe_float(r.get('composite_score')):.2f}" if safe_float(r.get("composite_score")) is not None else "—")
    k3.metric("LiveCodeBench Pass@1", f"{safe_float(r.get('livecodebench_pass1')):.2f}" if safe_float(r.get("livecodebench_pass1")) is not None else "—")
    k4.metric("SWE-bench Verified", pct(safe_float(r.get("swebench_verified_resolved_pct"))))

    st.markdown("---")
    st.markdown("#### Links")
    if r.get("link_openrouter"):
        st.link_button("OpenRouter model page", str(r.get("link_openrouter")), use_container_width=True)
    if r.get("link_swebench"):
        st.link_button("SWE-bench", str(r.get("link_swebench")), use_container_width=True)
    if r.get("link_livecodebench"):
        st.link_button("LiveCodeBench", str(r.get("link_livecodebench")), use_container_width=True)
    if r.get("link_provider"):
        st.link_button("Provider site", str(r.get("link_provider")), use_container_width=True)
    st.link_button("Web search (model)", external_search_link(str(r["display_name"])), use_container_width=True)

    st.markdown("---")
    st.subheader("All fields")
    st.dataframe(pd.DataFrame([r.to_dict()]), use_container_width=True, hide_index=True)


def render_sources(status: pd.DataFrame) -> None:
    st.title("🔌 Sources & Health")
    st.dataframe(status, use_container_width=True, hide_index=True)


# =========================
# Render
# =========================
if page == "rankings":
    render_rankings(f)
elif page == "model":
    render_model_detail(f)
else:
    render_sources(status_df)
