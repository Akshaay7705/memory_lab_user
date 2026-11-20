"""Memory Lab informal pilot Streamlit app."""
from __future__ import annotations

import json
import random
import time
import uuid
from datetime import datetime, timezone

import pandas as pd
import streamlit as st
from pymongo import MongoClient

from prompts import ANSWER_PROMPT, EXTRACTION_PROMPT

st.set_page_config(page_title="Memory Lab Pilot", page_icon="🧠", layout="wide")
st.title("Memory Lab · Pilot")
st.caption("Informal pilot, not a controlled study.")

CATEGORIES = ["Location", "Work/School", "People", "Interests", "Preferences", "Other"]
TOPICS = [
    ("city", "What city do you live in?"),
    ("workplace or school", "Where do you work or study?"),
    ("role or field", "What is your role or field?"),
    ("manager or mentor", "What is your manager or mentor's name?"),
    ("hobby", "What is a hobby you enjoy?"),
    ("diet", "Any food preferences you’d like me to remember?"),
    ("pet", "Do you have a pet? If so, what is their name?"),
    ("anything else", "What else would you like me to know about you?"),
    ("anything else", "Anything else you’d like to add?")]
SUGGESTIONS = {
    "Recall": "Where do I work?", "Update": "Which city do I live in now?",
    "History": "Where did I live before?",
    "Preference use": "Suggest something I might enjoy, based on what you know about me."}
MEMORY_MODES = ["no_memory", "rag", "rolling_summary", "graph_memory"]

def safe_notice(message: str):
    st.warning(message)

@st.cache_resource
def get_db():
    # Accept the project's existing secret names and the documented aliases.
    uri = st.secrets.get("MONGODB_URI") or st.secrets.get("MONGO_URI")
    if not uri:
        raise RuntimeError("MongoDB URI is not configured")
    database = st.secrets.get("MONGODB_DATABASE") or st.secrets.get("MONGO_DATABASE", "memory_lab")
    return MongoClient(uri, serverSelectionTimeoutMS=5000)[database]

@st.cache_resource
def get_llm():
    # Groq credentials and model selection come only from Streamlit secrets.
    from groq import Groq
    return Groq(api_key=st.secrets["GROQ_API_KEY"])

def llm_json(prompt: str):
    response = get_llm().chat.completions.create(
        model=st.secrets.get("GROQ_MODEL", "openai/gpt-oss-120b"),
        messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"})
    text = response.choices[0].message.content or "{}"
    return json.loads(text), response

def llm_text(prompt: str):
    result = get_llm().chat.completions.create(
        model=st.secrets.get("GROQ_MODEL", "openai/gpt-oss-120b"),
        messages=[{"role": "user", "content": prompt}])
    return result.choices[0].message.content or "I don’t know from the facts I have.", result

def state_init():
    defaults = {"phase": "consent", "participant_id": str(uuid.uuid4()), "session_id": str(uuid.uuid4()),
                "topic_idx": 0, "facts": [], "messages": [], "llm_calls": 0, "answers": []}
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

def call_ok():
    if st.session_state.llm_calls >= 40:
        safe_notice("You’ve reached this pilot session’s assistant limit. You can still review and rate the answers already generated.")
        return False
    st.session_state.llm_calls += 1
    return True

def init_runtime():
    state_init()
    try:
        get_db().command("ping")
        get_llm()
        return True
    except Exception:
        safe_notice("The pilot services are temporarily unavailable. Please try again later.")
        return False

def run_consent():
    st.write("This is an informal pilot. Your answers are stored anonymously under a random participant ID. Please don’t enter sensitive personal information such as health details, government IDs, passwords, or financial details.")
    if st.checkbox("I understand and agree to continue"):
        if st.button("Continue", type="primary"):
            st.session_state.phase = "intake"; st.rerun()

def run_intake():
    idx = st.session_state.topic_idx
    st.progress(min(idx / len(TOPICS), 1.0), text=f"{min(idx + 1, len(TOPICS))} of {len(TOPICS)} topics")
    if idx >= len(TOPICS):
        if st.button("Review what I shared", type="primary"):
            st.session_state.phase = "review"; st.rerun()
        return
    topic, question = TOPICS[idx]
    st.markdown(f"**Pilot:** {question} *(You can say “skip”.)*")
    with st.form(f"intake_{idx}"):
        msg = st.text_input("Your response", key=f"intake_input_{idx}")
        submitted = st.form_submit_button("Send")
    if submitted:
        if not msg.strip():
            safe_notice("Please enter an answer or type ‘skip’."); return
        if msg.strip().lower() == "skip":
            st.session_state.topic_idx += 1; st.rerun()
        if not call_ok(): return
        try:
            parsed, _ = llm_json(EXTRACTION_PROMPT.format(topic=topic, message=msg))
            value = parsed.get("value")
            if not value or not parsed.get("attribute"):
                safe_notice("I couldn’t identify a fact from that. Please rephrase, or type ‘skip’."); return
            cat = parsed.get("category")
            if cat not in CATEGORIES: cat = "Other"
            st.session_state.facts.append({"entity": str(parsed.get("entity") or "participant"),
                "attribute": str(parsed["attribute"]), "value": str(value), "category": cat, "superseded": False})
            st.session_state.topic_idx += 1; st.rerun()
        except Exception:
            safe_notice("I couldn’t process that answer. Please rephrase, or type ‘skip’.")

def run_review():
    st.subheader("Review your memory")
    facts = st.session_state.facts
    if not facts: st.info("No facts yet. You can add one below.")
    for cat in CATEGORIES:
        subset = [(i, f) for i, f in enumerate(facts) if f.get("category") == cat]
        if subset:
            st.markdown(f"**{cat}**")
            for i, fact in subset:
                a, b, c, d, e = st.columns([1, 1, 2, 1, .7])
                a.text_input("Entity", fact["entity"], key=f"e{i}")
                b.text_input("Attribute", fact["attribute"], key=f"a{i}")
                c.text_input("Value", fact["value"], key=f"v{i}")
                d.selectbox("Category", CATEGORIES, index=CATEGORIES.index(fact["category"]), key=f"c{i}")
                if e.button("Delete", key=f"d{i}"): st.session_state.facts.pop(i); st.rerun()
    with st.expander("Add a fact"):
        with st.form("add_fact"):
            entity = st.text_input("Entity", "participant"); attr = st.text_input("Attribute")
            value = st.text_input("Value"); cat = st.selectbox("Category", CATEGORIES)
            if st.form_submit_button("Add"):
                if attr and value:
                    facts.append({"entity": entity or "participant", "attribute": attr, "value": value, "category": cat, "superseded": False}); st.rerun()
    # Apply in-place edits from current widgets.
    for i, fact in enumerate(facts):
        for key, statekey in (("entity", f"e{i}"), ("attribute", f"a{i}"), ("value", f"v{i}"), ("category", f"c{i}")):
            if statekey in st.session_state: fact[key] = st.session_state[statekey]
    st.markdown("**Change a fact**")
    if facts:
        choice = st.selectbox("Choose a fact to update", range(len(facts)), format_func=lambda i: f"{facts[i]['attribute']}: {facts[i]['value']}")
        with st.form("change_fact"):
            new_value = st.text_input("New value")
            if st.form_submit_button("Save change"):
                if new_value.strip():
                    facts[choice]["superseded"] = True
                    replacement = {**facts[choice], "value": new_value.strip(), "superseded": False}
                    facts.append(replacement); st.success("Updated. The previous value will remain as history.")
    if st.button("Confirm and continue", type="primary"):
        try:
            db = get_db(); pid = st.session_state.participant_id
            db.graph_facts.delete_many({"participant_id": pid})
            now = datetime.now(timezone.utc)
            docs = [{**f, "participant_id": pid, "session_id": st.session_state.session_id, "created_at": now} for f in facts]
            if docs: db.graph_facts.insert_many(docs)
            st.session_state.phase = "test"; st.rerun()
        except Exception: safe_notice("We couldn’t save your memory right now. Please try again.")

def facts_for_answer(question):
    allfacts = list(get_db().graph_facts.find({"participant_id": st.session_state.participant_id}, {"_id": 0}))
    # Include all facts in this compact pilot so historical questions can retrieve superseded values.
    return allfacts

def answer_question(question, qtype):
    try:
        facts = facts_for_answer(question)
        query_id = str(uuid.uuid4())
        st.session_state.active_query_id = query_id
        for mode in MEMORY_MODES:
            if not call_ok(): break
            started = time.perf_counter()
            labeled = [{"entity": f["entity"], "attribute": f["attribute"], "value": f["value"],
                        "category": f["category"], "status": "superseded" if f.get("superseded") else "current"} for f in facts]
            if mode == "no_memory": used = []
            elif mode == "rag":
                terms = set(question.lower().replace("?", "").split())
                used = sorted(labeled, key=lambda f: sum(t in (f["attribute"] + " " + f["value"]).lower() for t in terms), reverse=True)[:3]
            elif mode == "rolling_summary":
                used = [{"summary": "; ".join(f"{f['attribute']}: {f['value']} ({f['status']})" for f in labeled)}] if labeled else []
            else: used = labeled
            answer, res = llm_text(ANSWER_PROMPT.format(question=question, facts=json.dumps(used, ensure_ascii=False)))
            usage = getattr(res, "usage", None); ti = getattr(usage, "prompt_tokens", 0) or 0; to = getattr(usage, "completion_tokens", 0) or 0
            cost = (ti * float(st.secrets.get("COST_PER_1K_INPUT_USD", 0)) + to * float(st.secrets.get("COST_PER_1K_OUTPUT_USD", 0))) / 1000
            row = {"result_id": str(uuid.uuid4()), "query_id": query_id, "participant_id": st.session_state.participant_id,
                "session_id": st.session_state.session_id, "question": question, "question_type": qtype,
                "memory_mode": mode, "answer": answer, "facts_used": used, "latency_s": time.perf_counter() - started,
                "tokens_in": ti, "tokens_out": to, "cost_usd": cost, "accuracy": None, "correct_label": None,
                "usefulness": None, "intrusive": None, "comment": None, "timestamp": datetime.now(timezone.utc)}
            get_db().pilot_results.insert_one(row)
            st.session_state.answers.append(row)
    except Exception: safe_notice("I couldn’t answer that just now. Please try another question.")

def answer_card(i, item):
    st.markdown(f"**Question:** {item['question']}\n\n**Answer:** {item['answer']}")
    st.caption(f"Memory mode: {item['memory_mode']}")
    with st.expander("Memory inspector"):
        if item["facts_used"]:
            st.dataframe(pd.DataFrame(item["facts_used"]), use_container_width=True, hide_index=True)
        else: st.write("No stored facts were available.")
    if item.get("rated"):
        st.success("Feedback saved")
        return
    with st.form(f"feedback_{i}"):
        correct = st.radio("Was this answer correct?", ["Correct", "Partly correct", "Incorrect", "Not sure"], horizontal=True, key=f"correct_{i}")
        useful = st.slider("How useful was it?", 1, 5, 3, key=f"useful_{i}")
        intrusive = st.checkbox("This felt intrusive or wrong to me", key=f"intrusive_{i}")
        comment = st.text_area("Optional comment", key=f"comment_{i}")
        if st.form_submit_button("Save rating"):
            try:
                accuracy = {"Correct": 1.0, "Partly correct": 0.5, "Incorrect": 0.0, "Not sure": None}[correct]
                get_db().pilot_feedback.insert_one({"participant_id": st.session_state.participant_id,
                    "result_id": item["result_id"], "question": item["question"], "question_type": item["question_type"],
                    "memory_mode": item["memory_mode"], "answer": item["answer"], "correct_label": correct,
                    "accuracy": accuracy, "usefulness": useful, "intrusive": intrusive, "comment": comment,
                    "latency_s": item["latency_s"], "tokens_in": item["tokens_in"], "tokens_out": item["tokens_out"],
                    "cost_usd": item["cost_usd"], "timestamp": datetime.now(timezone.utc)})
                get_db().pilot_results.update_one({"result_id": item["result_id"], "participant_id": st.session_state.participant_id},
                    {"$set": {"correct_label": correct, "accuracy": accuracy, "usefulness": useful, "intrusive": intrusive, "comment": comment}})
                item["rated"] = True; st.success("Thanks, your rating was saved.")
            except Exception: safe_notice("We couldn’t save that rating. Please try again.")

def run_test():
    st.subheader("Try asking your memory")
    st.write("Each question is answered in all four memory modes so you can review and rate every result.")
    typ = st.selectbox("Suggested question type", list(SUGGESTIONS))
    if st.button("Try a suggested question"):
        answer_question(SUGGESTIONS[typ], typ.lower().replace(" ", "_"))
        st.rerun()
    with st.form("free_question"):
        q = st.text_input("Or ask your own question")
        if st.form_submit_button("Ask"):
            if q.strip():
                answer_question(q.strip(), "free")
                st.rerun()
    current_items = [(i, item) for i, item in enumerate(st.session_state.answers)
                     if item.get("query_id") == st.session_state.get("active_query_id")]
    for i, item in current_items:
        with st.container(border=True): answer_card(i, item)
    render_best_mode_choices("test", st.session_state.get("active_query_id"))
    if st.button("Finish pilot", type="primary"):
        st.session_state.phase = "done"; st.rerun()

def render_best_mode_choices(area, active_query_id=None):
    groups = {}
    for item in st.session_state.answers:
        if active_query_id and item.get("query_id") != active_query_id: continue
        groups.setdefault(item.get("query_id"), []).append(item)
    for query_id, items in groups.items():
        if not query_id or len(items) != len(MEMORY_MODES): continue
        current = items[0].get("preferred_mode")
        with st.form(f"best_mode_{area}_{query_id}"):
            best = st.radio(f"Which memory mode gave the best answer to: {items[0]['question']}", MEMORY_MODES,
                            index=MEMORY_MODES.index(current) if current in MEMORY_MODES else 0)
            if st.form_submit_button("Save best answer"):
                try:
                    get_db().pilot_results.update_many(
                        {"participant_id": st.session_state.participant_id, "query_id": query_id},
                        {"$set": {"preferred_mode": best, "preference_timestamp": datetime.now(timezone.utc)}})
                    for item in items: item["preferred_mode"] = best
                    st.success(f"Saved preference: {best}")
                except Exception: safe_notice("We couldn’t save that preference. Please try again.")

def run_chat():
    st.subheader("Chat and update your memory")
    st.caption("Tell the chatbot something new, correct an existing fact, or ask a question. Say “skip” if you do not want to add a fact.")
    for entry in st.session_state.get("chat_messages", []):
        with st.chat_message(entry["role"]): st.write(entry["content"])
    active_chat_query_id = st.session_state.get("active_chat_query_id")
    chat_results = [(i, item) for i, item in enumerate(st.session_state.answers)
                    if item.get("question_type") == "chat" and item.get("query_id") == active_chat_query_id]
    for i, item in chat_results:
        with st.container(border=True): answer_card(i, item)
    render_best_mode_choices("chat", active_chat_query_id)
    with st.form("chat_turn", clear_on_submit=True):
        message = st.text_input("Message")
        send = st.form_submit_button("Send")
    if not send or not message.strip(): return
    if not call_ok(): return
    st.session_state.chat_messages = st.session_state.get("chat_messages", [])
    st.session_state.chat_messages.append({"role": "user", "content": message.strip()})
    try:
        if not call_ok(): return
        parsed, _ = llm_json(EXTRACTION_PROMPT.format(topic="new or changed personal information", message=message.strip()))
        value, attribute = parsed.get("value"), parsed.get("attribute")
        if message.strip().lower() != "skip" and value and attribute:
            db = get_db(); pid = st.session_state.participant_id
            current = db.graph_facts.find_one({"participant_id": pid, "attribute": attribute, "superseded": False})
            if not current or str(current.get("value")) != str(value):
                if current: db.graph_facts.update_one({"_id": current["_id"]}, {"$set": {"superseded": True}})
                cat = parsed.get("category") if parsed.get("category") in CATEGORIES else "Other"
                fact = {"participant_id": pid, "session_id": st.session_state.session_id, "entity": str(parsed.get("entity") or "participant"),
                    "attribute": str(attribute), "value": str(value), "category": cat, "superseded": False, "created_at": datetime.now(timezone.utc)}
                db.graph_facts.insert_one(fact)
                st.session_state.facts.append({k: fact[k] for k in ("entity", "attribute", "value", "category", "superseded")})
        question = message.strip()
        answer_question(question, "chat")
        st.session_state.active_chat_query_id = st.session_state.get("active_query_id")
        recent = st.session_state.answers[-4:]
        reply = next((x["answer"] for x in recent if x["memory_mode"] == "graph_memory"), "I couldn’t answer that from memory.")
        st.session_state.chat_messages.append({"role": "assistant", "content": reply})
        st.rerun()
    except Exception: safe_notice("I couldn’t process that chat message. Please try again.")

def run_summary():
    st.subheader("Your pilot summary")
    try:
        pid = st.session_state.participant_id
        results = list(get_db().pilot_results.find({"participant_id": pid}, {"_id": 0}))
        if not results: st.info("No answers have been processed yet."); return
        df = pd.DataFrame(results)
        st.write(f"Processed answer results: **{len(df)}** across **{df.memory_mode.nunique()}** memory modes.")
        summary = df.groupby("memory_mode").agg(
            answers=("result_id", "count"), avg_latency_s=("latency_s", "mean"),
            avg_cost_usd=("cost_usd", "mean"), avg_accuracy=("accuracy", "mean")).reset_index()
        st.dataframe(summary, use_container_width=True, hide_index=True)
        if "preferred_mode" in df and df.preferred_mode.notna().any():
            st.subheader("Modes chosen as best answer")
            choices = df.dropna(subset=["preferred_mode"]).drop_duplicates("query_id")
            st.dataframe(choices.preferred_mode.value_counts().rename("Questions"))
        st.caption("Accuracy uses your ratings: Correct = 1, Partly correct = 0.5, Incorrect = 0; Not sure is excluded. Cost is estimated from the per-token rates in Streamlit secrets.")
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Download all processed results (CSV)", df.to_csv(index=False), "my_pilot_results.csv", "text/csv")
        st.download_button("Download all processed results (JSON)", df.to_json(orient="records", date_format="iso", force_ascii=False), "my_pilot_results.json", "application/json")
        facts = list(get_db().graph_facts.find({"participant_id": pid}, {"_id": 0}))
        st.subheader("Your stored memory facts")
        if facts:
            fact_df = pd.DataFrame(facts); st.dataframe(fact_df, use_container_width=True, hide_index=True)
            st.download_button("Download memory facts (CSV)", fact_df.to_csv(index=False), "my_memory_facts.csv", "text/csv")
        else: st.info("No memory facts are stored yet.")
        feedback = list(get_db().pilot_feedback.find({"participant_id": pid}, {"_id": 0}))
        if feedback:
            feedback_df = pd.DataFrame(feedback)
            st.subheader("Saved feedback")
            st.dataframe(feedback_df, use_container_width=True, hide_index=True)
            st.download_button("Download feedback (CSV)", feedback_df.to_csv(index=False), "my_pilot_feedback.csv", "text/csv")
    except Exception: safe_notice("We couldn’t load your summary right now. Please try again.")

def run_done():
    st.subheader("Thanks for taking part")
    with st.form("wrapup"):
        remember = st.radio("Did it remember what you told it?", ["Yes", "Partly", "No"])
        off = st.text_area("Did anything feel off or creepy?")
        change = st.text_area("What would you change?")
        if st.form_submit_button("Submit closing feedback"):
            try:
                get_db().pilot_feedback.insert_one({"participant_id": st.session_state.participant_id, "record_type": "wrapup",
                    "remembered": remember, "off_comment": off, "change_comment": change, "timestamp": datetime.now(timezone.utc)})
                st.success("Thank you. Your closing feedback was saved.")
            except Exception: safe_notice("We couldn’t save your closing feedback. Please try again.")
def main():
    if not init_runtime(): return
    page = st.sidebar.radio("Page", ["Pilot", "Chat", "Summary"])
    if page == "Summary": run_summary(); return
    if page == "Chat":
        if st.session_state.phase in ("test", "done"): run_chat()
        else: st.info("Complete memory review before starting chat mode.")
        return
    {"consent": run_consent, "intake": run_intake, "review": run_review,
     "test": run_test, "done": run_done}.get(st.session_state.phase, run_consent)()

if __name__ == "__main__": main()
