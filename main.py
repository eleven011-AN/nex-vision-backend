import os
import json
import uuid
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(title="DataSense AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# In-memory dataset store: dataset_id -> DataFrame
uploaded_datasets: dict[str, pd.DataFrame] = {}

# ─── Color Palette ────────────────────────────────────────────────────────────
COLOR_LIST = ["#00c9a7", "#ff6b6b", "#339af0", "#ffd166", "#845ef7", "#4a5568", "#f06595", "#74c0fc"]

def col_palette(n: int) -> list[str]:
    return [COLOR_LIST[i % len(COLOR_LIST)] for i in range(n)]

def safe_val(v):
    """Convert numpy/pandas types to JSON-safe Python types."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.ndarray,)):
        return v.tolist()
    if pd.isna(v):
        return 0
    return v


# ─── Chart Builders ───────────────────────────────────────────────────────────

def build_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str, agg: str = "sum") -> dict:
    grouped = df.groupby(x_col)[y_col].agg(agg).reset_index()
    grouped = grouped.sort_values(y_col, ascending=False).head(20)
    return {
        "type": "bar",
        "title": title,
        "labels": [str(v) for v in grouped[x_col].tolist()],
        "datasets": [{
            "label": y_col.replace("_", " "),
            "data": [round(safe_val(v), 2) for v in grouped[y_col].tolist()],
            "backgroundColor": col_palette(len(grouped)),
        }],
    }


def build_line_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str, agg: str = "sum") -> dict:
    grouped = df.groupby(x_col)[y_col].agg(agg).reset_index()
    grouped = grouped.sort_values(x_col)
    return {
        "type": "line",
        "title": title,
        "labels": [str(v) for v in grouped[x_col].tolist()],
        "datasets": [{
            "label": y_col.replace("_", " "),
            "data": [round(safe_val(v), 2) for v in grouped[y_col].tolist()],
            "borderColor": "#00c9a7",
            "backgroundColor": "rgba(0,201,167,0.1)",
            "fill": True,
            "tension": 0.4,
        }],
    }


def build_area_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str, agg: str = "sum") -> dict:
    grouped = df.groupby(x_col)[y_col].agg(agg).reset_index()
    grouped = grouped.sort_values(x_col)
    return {
        "type": "area",
        "title": title,
        "labels": [str(v) for v in grouped[x_col].tolist()],
        "datasets": [{
            "label": y_col.replace("_", " "),
            "data": [round(safe_val(v), 2) for v in grouped[y_col].tolist()],
        }],
    }


def build_pie_chart(df: pd.DataFrame, group_col: str, value_col: str, title: str, agg: str = "sum") -> dict:
    grouped = df.groupby(group_col)[value_col].agg(agg).reset_index()
    grouped = grouped.sort_values(value_col, ascending=False).head(10)
    return {
        "type": "pie",
        "title": title,
        "labels": [str(v) for v in grouped[group_col].tolist()],
        "datasets": [{
            "data": [round(safe_val(v), 2) for v in grouped[value_col].tolist()],
            "backgroundColor": col_palette(len(grouped)),
        }],
    }


def build_stacked_bar(df: pd.DataFrame, x_col: str, stack_col: str, value_col: str, title: str) -> dict:
    pivot = df.groupby([x_col, stack_col])[value_col].sum().unstack(fill_value=0)
    pivot = pivot.head(20)
    labels = [str(v) for v in pivot.index.tolist()]
    categories = pivot.columns.tolist()
    return {
        "type": "stackedBar",
        "title": title,
        "labels": labels,
        "datasets": [
            {
                "label": str(cat),
                "data": [round(safe_val(v), 2) for v in pivot[cat].tolist()],
                "backgroundColor": COLOR_LIST[i % len(COLOR_LIST)],
            }
            for i, cat in enumerate(categories)
        ],
    }


def build_horizontal_bar(df: pd.DataFrame, x_col: str, y_col: str, title: str, agg: str = "sum") -> dict:
    grouped = df.groupby(x_col)[y_col].agg(agg).reset_index()
    grouped = grouped.sort_values(y_col, ascending=True).head(15)
    return {
        "type": "horizontalBar",
        "title": title,
        "labels": [str(v) for v in grouped[x_col].tolist()],
        "datasets": [{
            "label": y_col.replace("_", " "),
            "data": [round(safe_val(v), 2) for v in grouped[y_col].tolist()],
            "backgroundColor": col_palette(len(grouped)),
        }],
    }


def build_composed_chart(df: pd.DataFrame, x_col: str, y_col_area: str, y_col_line: str, title: str, agg: str = "sum") -> dict:
    grouped = df.groupby(x_col)[[y_col_area, y_col_line]].agg(agg).reset_index()
    grouped = grouped.sort_values(x_col)
    return {
        "type": "composed",
        "title": title,
        "labels": [str(v) for v in grouped[x_col].tolist()],
        "datasets": [
            {
                "label": y_col_area.replace("_", " "),
                "type": "area",
                "data": [round(safe_val(v), 2) for v in grouped[y_col_area].tolist()],
            },
            {
                "label": y_col_line.replace("_", " "),
                "type": "line",
                "data": [round(safe_val(v), 2) for v in grouped[y_col_line].tolist()],
            }
        ],
    }


def build_radar_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str, agg: str = "sum") -> dict:
    grouped = df.groupby(x_col)[y_col].agg(agg).reset_index()
    grouped = grouped.sort_values(y_col, ascending=False).head(8)
    return {
        "type": "radar",
        "title": title,
        "labels": [str(v) for v in grouped[x_col].tolist()],
        "datasets": [{
            "label": y_col.replace("_", " "),
            "data": [round(safe_val(v), 2) for v in grouped[y_col].tolist()],
        }],
    }


# ─── Column Type Detection ────────────────────────────────────────────────────

def detect_columns(df: pd.DataFrame):
    """Detect numeric, categorical, and date columns from the dataframe."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Try to detect date columns from object columns
    date_cols = []
    for col in categorical_cols[:]:
        try:
            parsed = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
            if parsed.notna().sum() > len(df) * 0.5:
                date_cols.append(col)
        except Exception:
            pass

    # Filter out date cols from categorical
    categorical_cols = [c for c in categorical_cols if c not in date_cols]

    return numeric_cols, categorical_cols, date_cols


def smart_default_charts(df: pd.DataFrame) -> tuple[list, list]:
    """Auto-generate a default set of charts and KPIs for any uploaded dataset."""
    numeric_cols, categorical_cols, date_cols = detect_columns(df)
    charts = []
    kpis = []

    # ── KPIs: use first 4 numeric columns ──
    kpi_colors = ["teal", "coral", "purple", "amber"]
    kpi_icons = ["bar", "rupee", "trending", "target"]
    for i, col in enumerate(numeric_cols[:4]):
        total = df[col].sum()
        avg = df[col].mean()
        
        # Format as INR
        if total >= 1e7:
            formatted = f"₹{total/1e7:.2f}Cr"
        elif total >= 1e5:
            formatted = f"₹{total/1e5:.2f}L"
        elif total >= 1000:
            formatted = f"₹{total/1000:.1f}K"
        else:
            formatted = f"₹{total:,.0f}" if total > 0 else "0"

        kpis.append({
            "label": col.replace("_", " ").title(),
            "value": formatted,
            "trend": f"+{((avg / (total / len(df))) * 0.1):.1f}%" if total != 0 else "N/A",
            "up": True,
            "iconName": kpi_icons[i % len(kpi_icons)],
        })

    # Ensure we always have exactly 4 KPIs
    if len(kpis) < 4:
        if not any(k["label"] == "Total Rows" for k in kpis):
            kpis.append({"label": "Total Rows", "value": str(len(df)), "trend": "Dataset Size", "up": True, "iconName": "database"})
    if len(kpis) < 4:
        if not any(k["label"] == "Total Columns" for k in kpis):
            kpis.append({"label": "Total Columns", "value": str(len(df.columns)), "trend": "Features", "up": True, "iconName": "layers"})
    if len(kpis) < 4:
        kpis.append({"label": "Categories", "value": str(len(categorical_cols)), "trend": "Segments", "up": True, "iconName": "pie"})
    if len(kpis) < 4:
        missing = int(df.isna().sum().sum())
        kpis.append({"label": "Missing Values", "value": str(missing), "trend": "Data Quality", "up": missing == 0, "iconName": "warning" if missing > 0 else "check"})

    # ── Chart 1: If date + 2+ numeric → Composed chart ──
    if date_cols and len(numeric_cols) >= 2:
        x = date_cols[0]
        y1 = numeric_cols[0]
        y2 = numeric_cols[1]
        df[x] = pd.to_datetime(df[x], errors="coerce")
        df["_period"] = df[x].dt.to_period("M").astype(str)
        charts.append(build_composed_chart(df, "_period", y1, y2, f"{y1.replace('_',' ').title()} & {y2.replace('_',' ').title()} Trend (Composed)"))
    elif date_cols and len(numeric_cols) == 1:
        x = date_cols[0]
        y = numeric_cols[0]
        df[x] = pd.to_datetime(df[x], errors="coerce")
        df["_period"] = df[x].dt.to_period("M").astype(str)
        charts.append(build_area_chart(df, "_period", y, f"{y.replace('_',' ').title()} Trend (Area)"))

    # ── Chart 2: Radar chart for multi-category breakdown ──
    if categorical_cols and len(numeric_cols) >= 1:
        cat = categorical_cols[0]
        num1 = numeric_cols[0]
        if df[cat].nunique() > 2 and df[cat].nunique() <= 12:
            charts.append(build_radar_chart(df, cat, num1, f"{num1.replace('_',' ').title()} by {cat.replace('_',' ').title()} (Radar)"))

    # ── Chart 3: Categorical + numeric → Bar ──
    if categorical_cols and numeric_cols:
        cat = categorical_cols[0]
        num = numeric_cols[0]
        charts.append(build_bar_chart(df, cat, num, f"{num.replace('_',' ').title()} by {cat.replace('_',' ').title()}"))

    # ── Chart 4: Second categorical → Pie ──
    if len(categorical_cols) >= 1 and numeric_cols:
        cat = categorical_cols[min(1, len(categorical_cols)-1)]
        num = numeric_cols[0]
        charts.append(build_pie_chart(df, cat, num, f"{num.replace('_',' ').title()} by {cat.replace('_',' ').title()} (Pie)"))

    # ── Chart 5: Second numeric + categorical → Horizontal bar ──
    if categorical_cols and len(numeric_cols) >= 2:
        cat = categorical_cols[0]
        num = numeric_cols[1]
        charts.append(build_horizontal_bar(df, cat, num, f"{num.replace('_',' ').title()} by {cat.replace('_',' ').title()} (Ranked)"))

    # ── Chart 6: Stacked bar if 2+ categoricals + numeric ──
    if len(categorical_cols) >= 2 and numeric_cols:
        x_col = categorical_cols[0]
        stack_col = categorical_cols[1]
        num = numeric_cols[0]
        if df[x_col].nunique() <= 30 and df[stack_col].nunique() <= 10:
            charts.append(build_stacked_bar(df, x_col, stack_col, num, f"{num.replace('_',' ').title()} by {x_col.replace('_',' ').title()} & {stack_col.replace('_',' ').title()}"))

    return charts, kpis


# ─── NLP Query Parser ─────────────────────────────────────────────────────────

def parse_query(prompt: str, df: pd.DataFrame) -> dict:
    p = prompt.lower()
    numeric_cols, categorical_cols, date_cols = detect_columns(df)

    # ─── Apply Row Filters from Prompt ───
    # If the user mentions a specific value (e.g., "east"), filter the dataframe
    filters_applied = []
    for col in categorical_cols:
        if df[col].nunique() < 50:
            for val in df[col].dropna().unique():
                val_str = str(val).lower()
                # Check for whole word match to avoid partial overlaps
                if len(val_str) >= 2 and (val_str in p.split() or f" {val_str} " in f" {p} "):
                    df = df[df[col].astype(str).str.lower() == val_str]
                    filters_applied.append(f"{col} = '{val}'")
                    break

    if len(df) == 0:
        raise Exception("The filters applied based on your prompt resulted in 0 matching rows.")

    # Detect intent
    wants_line   = any(k in p for k in ["trend", "over time", "monthly", "by month", "time series", "timeline", "per month", "per year", "line"])
    wants_area   = any(k in p for k in ["area"])
    wants_pie    = any(k in p for k in ["percentage", "share", "proportion", "breakdown", "pie", "donut", "distribution"])
    wants_hbar   = any(k in p for k in ["horizontal", "rank", "ranking", "top", "bottom", "least", "most", "compare"])
    wants_stacked = any(k in p for k in ["stacked", "grouped", "by stage", "by category", "by type"])
    wants_radar  = any(k in p for k in ["radar", "spider", "star", "web"])
    wants_avg    = any(k in p for k in ["average", "avg", "mean"])
    wants_count  = any(k in p for k in ["count", "how many", "number of", "frequency"])
    wants_sum    = any(k in p for k in ["total", "sum", "revenue", "sales", "amount", "value"])

    agg = "mean" if wants_avg else "sum"

    # Figure out which numeric column to use (Y axis)
    y_col = None
    for col in numeric_cols:
        col_lower = col.lower().replace("_", " ")
        if any(word in p for word in col_lower.split()):
            y_col = col
            break
    if y_col is None and numeric_cols:
        y_col = numeric_cols[0]

    # Figure out X column (grouping)
    x_col = None

    # Check if date column matches
    for col in date_cols:
        col_lower = col.lower().replace("_", " ")
        if any(word in p for word in col_lower.split()) or wants_line:
            x_col = col
            break

    # Check categorical columns
    if x_col is None:
        for col in categorical_cols:
            col_lower = col.lower().replace("_", " ")
            if any(word in p for word in col_lower.split()):
                x_col = col
                break

    if x_col is None:
        if wants_line and date_cols:
            x_col = date_cols[0]
        elif categorical_cols:
            x_col = categorical_cols[0]
        elif date_cols:
            x_col = date_cols[0]
        else:
            x_col = df.columns[0]

    # Stack column for stacked charts
    stack_col = None
    if wants_stacked and len(categorical_cols) >= 2:
        for col in categorical_cols:
            if col != x_col:
                stack_col = col
                break

    title = prompt.strip().capitalize()[:70]
    charts = []

    # Convert date col for line chart
    if x_col in date_cols:
        df = df.copy()
        df[x_col] = pd.to_datetime(df[x_col], errors="coerce")
        df["_period"] = df[x_col].dt.to_period("M").astype(str)
        x_col = "_period"

    # Build chart
    if wants_stacked and stack_col and y_col:
        charts.append(build_stacked_bar(df, x_col, stack_col, y_col, title))
    elif wants_pie and y_col:
        charts.append(build_pie_chart(df, x_col, y_col, title, agg))
    elif wants_radar and y_col:
        charts.append(build_radar_chart(df, x_col, y_col, title, agg))
    elif wants_area and y_col:
        charts.append(build_area_chart(df, x_col, y_col, title, agg))
    elif wants_line and y_col:
        charts.append(build_line_chart(df, x_col, y_col, title, agg))
    elif wants_hbar and y_col:
        charts.append(build_horizontal_bar(df, x_col, y_col, title, agg))
    else:
        # Smart auto-selection
        if x_col in (date_cols + ["_period"]):
            charts.append(build_area_chart(df, x_col, y_col, title, agg))
        elif wants_line:
            charts.append(build_line_chart(df, x_col, y_col, title, agg))
        else:
            unique_vals = df[x_col].nunique() if x_col in df.columns else 0
            if unique_vals > 2 and unique_vals <= 8 and not wants_hbar:
                charts.append(build_radar_chart(df, x_col, y_col, f"{title} (Radar)", agg))
                charts.append(build_bar_chart(df, x_col, y_col, title, agg))
            elif unique_vals <= 8:
                charts.append(build_pie_chart(df, x_col, y_col, f"{title} (Distribution)", agg))
                charts.append(build_bar_chart(df, x_col, y_col, title, agg))
            else:
                charts.append(build_horizontal_bar(df, x_col, y_col, title, agg))

    # Add a secondary chart
    if y_col and x_col and not wants_pie:
        if not wants_line and categorical_cols:
            cat2 = next((c for c in categorical_cols if c != x_col), None)
            if cat2:
                charts.append(build_pie_chart(df, cat2, y_col, f"{y_col.replace('_',' ').title()} by {cat2.replace('_',' ').title()}", agg))

    # KPIs
    kpis = []
    if y_col and y_col in df.columns:
        total = float(df[y_col].sum())
        avg_v = float(df[y_col].mean())
        max_v = float(df[y_col].max())
        min_v = float(df[y_col].min())
        def fmt(n):
            v = abs(n)
            if v >= 1e7: return f"₹{n/1e7:.2f}Cr"
            if v >= 1e5: return f"₹{n/1e5:.2f}L"
            if v >= 1000: return f"₹{n/1000:.1f}K"
            return f"₹{n:,.0f}" if n > 0 else "0"

        kpis = [
            {"label": f"Total {y_col.replace('_',' ').title()}", "value": fmt(total), "trend": "+", "up": True, "iconName": "rupee"},
            {"label": f"Avg {y_col.replace('_',' ').title()}",   "value": fmt(avg_v),  "trend": "avg", "up": True, "iconName": "trending"},
            {"label": f"Max {y_col.replace('_',' ').title()}",   "value": fmt(max_v),  "trend": "peak", "up": True, "iconName": "trophy"},
            {"label": "Total Records",                            "value": f"{len(df):,}",   "trend": "rows", "up": True, "iconName": "database"},
        ]

    result = {
        "success": True,
        "charts": charts,
        "kpis": kpis,
        "rows_analyzed": len(df),
        "columns_used": {"x": x_col, "y": y_col},
        "filters_applied": filters_applied,
    }

    return result


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "DataSense AI API", "datasets_loaded": len(uploaded_datasets)}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    filename = file.filename or "upload.csv"
    ext = Path(filename).suffix.lower()

    if ext not in [".csv", ".xlsx", ".json"]:
        raise HTTPException(status_code=400, detail="Only CSV, Excel (.xlsx), or JSON files are supported.")

    dataset_id = str(uuid.uuid4())[:8]
    save_path = UPLOAD_DIR / f"{dataset_id}{ext}"

    contents = await file.read()
    with open(save_path, "wb") as f:
        f.write(contents)

    try:
        if ext == ".csv":
            try:
                df = pd.read_csv(save_path)
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(save_path, encoding="utf-8-sig")
                except UnicodeDecodeError:
                    df = pd.read_csv(save_path, encoding="latin1")
        elif ext == ".xlsx":
            df = pd.read_excel(save_path)
        elif ext == ".json":
            df = pd.read_json(save_path)

        # Clean column names
        df.columns = [str(c).strip().replace(" ", "_") for c in df.columns]

        # Drop fully-empty columns/rows
        df = df.dropna(how="all", axis=1).dropna(how="all", axis=0)

        uploaded_datasets[dataset_id] = df
        numeric_cols, categorical_cols, date_cols = detect_columns(df)

        return {
            "success": True,
            "dataset_id": dataset_id,
            "filename": filename,
            "rows": len(df),
            "columns": list(df.columns),
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "date_columns": date_cols,
            "preview": df.head(3).fillna("").to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse file: {str(e)}")


@app.get("/dataset/{dataset_id}/overview")
def dataset_overview(dataset_id: str):
    """Return auto-generated charts for the uploaded dataset without any query."""
    if dataset_id not in uploaded_datasets:
        raise HTTPException(status_code=404, detail="Dataset not found. Please upload a file first.")

    df = uploaded_datasets[dataset_id].copy()
    charts, kpis = smart_default_charts(df)

    return {
        "success": True,
        "charts": charts,
        "kpis": kpis,
        "rows": len(df),
        "columns": list(df.columns),
    }


class QueryRequest(BaseModel):
    prompt: str
    dataset_id: Optional[str] = None


@app.post("/query")
def query_data(request: QueryRequest):
    prompt = request.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    if not request.dataset_id or request.dataset_id not in uploaded_datasets:
        return JSONResponse(content={
            "success": False,
            "charts": [],
            "kpis": [],
            "ai_text": "⚠️ No dataset loaded. Please **upload a CSV file first** using the upload button, then ask your question.",
            "rows_analyzed": 0,
        })

    df_original = uploaded_datasets[request.dataset_id]
    df = df_original.copy()

    # ── Step 1: Build dynamic dataset schema from the actual uploaded file ─────
    numeric_cols, cat_cols, date_cols = detect_columns(df_original)

    schema_lines = [
        f"Total rows: {len(df_original)}",
        f"Total columns: {len(df_original.columns)}",
        f"Numeric columns: {', '.join(numeric_cols) or 'none'}",
        f"Date columns: {', '.join(date_cols) or 'none'}",
        "Categorical columns and their unique values:",
    ]
    for col in cat_cols:
        unique_vals = df_original[col].dropna().unique()
        if len(unique_vals) <= 50:
            schema_lines.append(f"  - {col}: {', '.join(str(v) for v in sorted(unique_vals, key=str))}")
        else:
            schema_lines.append(f"  - {col}: ({len(unique_vals)} unique values)")

    dataset_schema = "\n".join(schema_lines)

    # ── Step 2: Try to parse query & build charts (best-effort, always runs) ───
    chart_result = None
    try:
        chart_result = parse_query(prompt, df)
    except Exception:
        pass  # Non-data queries (chat, greetings, etc.) won't produce charts — that's fine

    # ── Step 3: Build chart data context for Gemini ───────────────────────────
    chart_context = ""
    if chart_result and chart_result.get("charts"):
        chart_context += "\nAggregated chart data computed from the dataset:\n"
        for c in chart_result.get("charts", []):
            chart_context += f"Chart: '{c.get('title', '')}'\n"
            for ds in c.get("datasets", []):
                data_dict = dict(zip(c.get("labels", []), ds.get("data", [])))
                chart_context += f"  {ds.get('label', 'Value')}: {data_dict}\n"

    kpi_context = ""
    if chart_result and chart_result.get("kpis"):
        kpi_context = "\nKey Metrics:\n"
        for kpi in chart_result["kpis"]:
            kpi_context += f"  - {kpi['label']}: {kpi['value']} ({kpi['trend']})\n"

    applied = (chart_result or {}).get("filters_applied", [])
    filter_str = ", ".join(applied) if applied else "None (full dataset)"

    # ── Step 4: Call Gemini with full context — handles ALL query types ────────
    if not GEMINI_API_KEY:
        ai_text = "⚠️ *AI summary unavailable* — Gemini API key not configured."
    else:
        try:
            model = genai.GenerativeModel("gemini-2.5-flash")

            sys_prompt = f"""You are DataSense AI, a smart and friendly data analytics assistant.
The user has uploaded a dataset. Here is its complete schema:

{dataset_schema}

Filters currently applied: {filter_str}
{kpi_context}{chart_context}

Instructions:
1. If the user is making casual conversation (greeting, saying goodbye, asking how you are, saying thanks, etc.) — respond warmly and naturally in 1-2 sentences like a professional assistant, without mentioning data.
2. If the user is asking about data, analysis, charts, or specific values — answer directly using ONLY the numbers from the chart data and metrics above. Use Markdown bolding for key numbers.
3. If the user asks about a value or category that does NOT exist in the schema above (e.g., a region or product not listed) — clearly tell them it's not in the dataset, and suggest what IS available based on the schema.
4. Never hallucinate or invent data.
5. Never mention "JSON", "chart data", or "schema" — answer as if you analyzed the data yourself.
6. Keep responses concise: 1-3 sentences for chat, 1-2 paragraphs for data questions.

User message: \"{prompt}\""""

            response = model.generate_content(sys_prompt)
            ai_text = response.text or "I analyzed the data, but could not generate a response."

        except Exception as e:
            ai_text = f"⚠️ *AI summary unavailable:* {str(e)}"

    # Build final response
    result = chart_result or {
        "success": True,
        "charts": [],
        "kpis": [],
        "rows_analyzed": 0,
        "columns_used": {},
        "filters_applied": [],
    }
    result["ai_text"] = ai_text
    return JSONResponse(content=result)

