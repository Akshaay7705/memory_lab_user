"""Cross-participant results page for the four Memory Lab modes."""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st
from pymongo import MongoClient

MODES = ["no_memory", "rag", "rolling_summary", "graph_memory"]
MODE_COLORS = {"no_memory": "#8c8c8c", "rag": "#4c78a8", "rolling_summary": "#f2a541", "graph_memory": "#16864b"}


@st.cache_resource
def _get_results_db():
    uri = st.secrets.get("MONGODB_URI") or st.secrets.get("MONGO_URI")
    if not uri:
        raise RuntimeError("Database configuration is unavailable")
    name = st.secrets.get("MONGODB_DATABASE") or st.secrets.get("MONGO_DATABASE", "memory_lab")
    return MongoClient(uri, serverSelectionTimeoutMS=5000)[name]


def _collect_results():
    db = _get_results_db()
    results = list(db.pilot_results.find({}, {"_id": 0}))
    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    feedback = list(db.pilot_feedback.find({"result_id": {"$exists": True}}, {"_id": 0}))
    if feedback:
        fdf = pd.DataFrame(feedback)
        if "result_id" in fdf:
            fdf = fdf.drop_duplicates("result_id", keep="last").set_index("result_id")
            for field in ("correct_label", "accuracy", "usefulness", "intrusive", "comment", "timestamp"):
                if field in fdf:
                    fb = df.result_id.map(fdf[field])
                    if field not in df: df[field] = fb
                    else: df[field] = fb.combine_first(df[field])
    if "accuracy" not in df: df["accuracy"] = pd.NA
    labels = {"Correct": 1.0, "Partly correct": 0.5, "Incorrect": 0.0}
    calculated = df.get("correct_label", pd.Series(index=df.index, dtype="object")).map(labels)
    df["accuracy"] = pd.to_numeric(df["accuracy"], errors="coerce").combine_first(calculated)
    if "memory_mode" in df:
        df = df[df.memory_mode.isin(MODES)].copy()
    return df


def render_results():
    st.title("Memory Lab · Results")
    st.caption("Aggregated across all anonymous participants. This is an informal pilot, not a controlled study.")
    try:
        df = _collect_results()
    except Exception:
        st.warning("Results are temporarily unavailable. Please try again later.")
        return
    if df.empty:
        st.info("No participant results have been recorded yet.")
        return

    for col in ("latency_s", "cost_usd", "usefulness"):
        if col not in df: df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "participant_id" not in df: df["participant_id"] = "unknown"
    rated = df.dropna(subset=["accuracy"]).copy()
    if "timestamp" in rated:
        rated["timestamp"] = pd.to_datetime(rated.timestamp, errors="coerce", utc=True)
    else:
        rated["timestamp"] = pd.NaT

    st.metric("Participants", int(df.participant_id.nunique()))
    st.metric("Generated answers", len(df))
    st.metric("Rated answers", len(rated))
    st.subheader("Which memory mode performs best?")
    if rated.empty:
        st.info("Accuracy comparisons will appear after participants rate answers.")
        best_mode = None
    else:
        stats = rated.groupby("memory_mode", as_index=False).agg(
            accuracy=("accuracy", "mean"), latency_s=("latency_s", "mean"), cost_usd=("cost_usd", "mean"),
            rated_answers=("accuracy", "count"))
        stats = stats.set_index("memory_mode").reindex(MODES).reset_index()
        ranked = stats.dropna(subset=["accuracy"]).sort_values(
            ["accuracy", "latency_s", "cost_usd"], ascending=[False, True, True], na_position="last")
        best_mode = ranked.iloc[0].memory_mode if not ranked.empty else None
        if best_mode:
            st.success(f"Current leader by average participant-rated accuracy: **{best_mode}**. Ties are broken by lower average latency, then lower estimated cost.")

        bar = alt.Chart(stats.dropna(subset=["accuracy"])).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("memory_mode:N", sort=MODES, title="Memory mode"),
            y=alt.Y("accuracy:Q", scale=alt.Scale(domain=[0, 1]), title="Average rated accuracy (0–1)"),
            color=alt.condition(alt.datum.memory_mode == best_mode, alt.value(MODE_COLORS["graph_memory"]), alt.value("#9ca3af")),
            tooltip=["memory_mode", alt.Tooltip("accuracy:Q", format=".3f"), "rated_answers"])
        st.altair_chart(bar.properties(title="Average rated accuracy by memory mode", height=300), use_container_width=True)

        if not rated.timestamp.isna().all():
            trend = rated.dropna(subset=["timestamp"]).sort_values("timestamp").copy()
            trend["answer_number"] = trend.groupby("memory_mode").cumcount() + 1
            line = alt.Chart(trend).mark_line(point=True).encode(
                x=alt.X("answer_number:Q", title="Rated answer number"),
                y=alt.Y("accuracy:Q", scale=alt.Scale(domain=[0, 1]), title="Participant rating (0–1)"),
                color=alt.Color("memory_mode:N", scale=alt.Scale(domain=MODES, range=[MODE_COLORS[m] for m in MODES]), title="Memory mode"),
                detail="memory_mode:N", tooltip=["memory_mode", "question", alt.Tooltip("accuracy:Q", format=".2f"), "timestamp"])
            st.altair_chart(line.properties(title="Rated answer accuracy over collection order", height=300), use_container_width=True)
        st.subheader("Mode metrics")
        st.dataframe(stats.rename(columns={"memory_mode": "Memory mode", "accuracy": "Avg accuracy", "latency_s": "Avg latency (s)",
            "cost_usd": "Avg estimated cost ($)", "rated_answers": "Rated answers"}), use_container_width=True, hide_index=True)

    st.subheader("All participants by memory mode")
    participant_table = df.groupby(["participant_id", "memory_mode"], as_index=False).agg(
        questions=("result_id", "nunique"), rated_answers=("accuracy", "count"), avg_accuracy=("accuracy", "mean"),
        avg_usefulness=("usefulness", "mean"), avg_latency_s=("latency_s", "mean"), avg_cost_usd=("cost_usd", "mean"))
    if "preferred_mode" in df:
        prefs = df.dropna(subset=["preferred_mode"]).drop_duplicates(["participant_id", "query_id"])
        pref_counts = prefs.groupby(["participant_id", "preferred_mode"]).size().rename("times_chosen_best").reset_index()
        participant_table = participant_table.merge(pref_counts, left_on=["participant_id", "memory_mode"],
                                                     right_on=["participant_id", "preferred_mode"], how="left")
        participant_table = participant_table.drop(columns=["preferred_mode"], errors="ignore")
        participant_table["times_chosen_best"] = participant_table.times_chosen_best.fillna(0).astype(int)
    st.dataframe(participant_table, use_container_width=True, hide_index=True)
