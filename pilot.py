"""Memory Lab informal pilot Streamlit app (v2).

Changes vs v1 (see notes in the reply):
  - RAG = embedding retrieval over raw chat rounds with LongMemEval-style indexing
    (round-level values + fact-augmented keys). Time-aware query expansion not implemented.
  - Rolling summary = LLM-written summary of the raw chat, updated chunk by chunk.
  - Graph memory = structured facts with current/superseded status (unchanged).
  - Same context budget for every strategy; all four modes answer every question or none do.
  - Answers are shown blinded (Answer A-D, shuffled); mode names and the inspector are revealed after rating.
  - Raw chat is kept in session_state.chat_log so RAG/summary can see it.
  - Fixes: widget keys use a fact id (deleting a fact no longer shifts edits onto the wrong fact),
    call_ok(n) counts calls once, rows carry impl_version="v2".
Requires: fastembed (add to requirements.txt).
"""
from __future__ import annotations

import hashlib
import json
import random
import time
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st
from pymongo import MongoClient

from prompts import ANSWER_PROMPT, EXTRACTION_PROMPT
from results_page import render_results

st.set_page_config(page_title="Memory Lab Pilot", page_icon="🧠", layout="wide")

IMPL_VERSION = "v2"
MAX_CALLS = 60
MAX_CONTEXT_CHARS = 1500
RAG_TOP_K = 6
SUMMARY_CHUNK = 4

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

SUMMARY_UPDATE_PROMPT = """You maintain a short running summary of what a user has said about themselves.

Current summary:
{summary}

New messages:
{new}

Return the updated summary in at most 120 words. Keep what the user stated. If something changed, \
note both the new value and the earlier one. Do not invent anything."""

FACT_KEY_PROMPT = """For each numbered chat round below, list the personal facts the user states, \
as a short plain-text phrase (for example "lives in Pune; works at Infosys"). \
If a round states no personal fact, use an empty string.

Return ONLY a JSON array of exactly {n} strings, in the same order as the rounds.

{rounds}"""


def safe_notice(message: str):
    st.warning(message)


@st.cache_resource
def get_db():
    uri = st.secrets.get("MONGODB_URI") or st.secrets.get("MONGO_URI")
    if not uri:
        raise RuntimeError("MongoDB URI is not configured")
    database = st.secrets.get("MONGODB_DATABASE") or st.secrets.get("MONGO_DATABASE", "memory_lab")
    return MongoClient(uri, serverSelectionTimeoutMS=5000)[database]


@st.cache_resource
def get_llm():
    from groq import Groq
    return Groq(api_key=st.secrets["GROQ_API_KEY"])


@st.cache_resource
def _embedder():
    from fastembed import TextEmbedding
    return TextEmbedding("BAAI/bge-small-en-v1.5")


def embed(texts):
    """One vector per text (local ONNX model; Groq has no embeddings endpoint)."""
    return [v.tolist() for v in _embedder().embed(texts)]


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
                "topic_idx": 0, "facts": [], "messages": [], "llm_calls": 0, "answers": [], "chat_log": []}
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def call_ok(n: int = 1):
    """Check the per-session budget for n calls and reserve them (counted once)."""
    if st.session_state.llm_calls + n > MAX_CALLS:
        safe_notice("You’ve reached this pilot session’s assistant limit. You can still review and rate the answers already generated.")
        return False
    st.session_state.llm_calls += n
    return True


def _counted_llm_text(prompt: str):
    st.session_state.llm_calls += 1
    return llm_text(prompt)


def log_pair(bot: str, user: str):
    """Raw history for RAG / rolling summary: what the bot asked and what the user said."""
    st.session_state.chat_log.append({"bot": bot, "user": user})


def forget(fact: dict):
    """If a participant deletes a fact, drop raw messages that mention its value."""
    v = str(fact["value"]).lower()
    st.session_state.chat_log = [p for p in st.session_state.chat_log if v not in p["user"].lower()]


def new_fact(entity, attribute, value, category, superseded=False):
    return {"fid": uuid.uuid4().hex[:8], "entity": str(entity or "participant"), "attribute": str(attribute),
            "value": str(value), "category": category if category in CATEGORIES else "Other",
            "superseded": superseded, "orig": str(value)}


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
            st.session_state.phase = "intake"
            st.rerun()


def run_intake():
    idx = st.session_state.topic_idx
    st.progress(min(idx / len(TOPICS), 1.0), text=f"{min(idx + 1, len(TOPICS))} of {len(TOPICS)} topics")
    if idx >= len(TOPICS):
        if st.button("Review what I shared", type="primary"):
            st.session_state.phase = "review"
            st.rerun()
        return
    topic, question = TOPICS[idx]
    st.markdown(f"**Pilot:** {question} *(You can say “skip”.)*")
    with st.form(f"intake_{idx}"):
        msg = st.text_input("Your response", key=f"intake_input_{idx}")
        submitted = st.form_submit_button("Send")
    if submitted:
        if not msg.strip():
            safe_notice("Please enter an answer or type ‘skip’.")
            return
        if msg.strip().lower() == "skip":
            st.session_state.topic_idx += 1
            st.rerun()
        if not call_ok():
            return
        try:
            parsed, _ = llm_json(EXTRACTION_PROMPT.format(topic=topic, message=msg))
            value = parsed.get("value")
            if not value or not parsed.get("attribute"):
                safe_notice("I couldn’t identify a fact from that. Please rephrase, or type ‘skip’.")
                return
            st.session_state.facts.append(new_fact(parsed.get("entity"), parsed["attribute"], value, parsed.get("category")))
            log_pair(question, msg.strip())
            st.session_state.topic_idx += 1
            st.rerun()
        except Exception:
            safe_notice("I couldn’t process that answer. Please rephrase, or type ‘skip’.")


def run_review():
    st.subheader("Review your memory")
    facts = st.session_state.facts
    if not facts:
        st.info("No facts yet. You can add one below.")
    for cat in CATEGORIES:
        subset = [f for f in facts if f.get("category") == cat]
        if subset:
            st.markdown(f"**{cat}**")
            for fact in subset:
                fid = fact["fid"]
                a, b, c, d, e = st.columns([1, 1, 2, 1, .7])
                a.text_input("Entity", fact["entity"], key=f"e_{fid}")
                b.text_input("Attribute", fact["attribute"], key=f"a_{fid}")
                c.text_input("Value", fact["value"], key=f"v_{fid}")
                d.selectbox("Category", CATEGORIES, index=CATEGORIES.index(fact["category"]), key=f"c_{fid}")
                if e.button("Delete", key=f"d_{fid}"):
                    forget(fact)
                    st.session_state.facts = [x for x in facts if x["fid"] != fid]
                    st.rerun()
    with st.expander("Add a fact"):
        with st.form("add_fact"):
            entity = st.text_input("Entity", "participant")
            attr = st.text_input("Attribute")
            value = st.text_input("Value")
            cat = st.selectbox("Category", CATEGORIES)
            if st.form_submit_button("Add"):
                if attr and value:
                    facts.append(new_fact(entity, attr, value, cat))
                    log_pair("Add a fact", f"Please remember: my {attr} is {value}")
                    st.rerun()
    # Apply in-place edits from the current widgets (keyed by fact id).
    for fact in facts:
        fid = fact["fid"]
        for key, statekey in (("entity", f"e_{fid}"), ("attribute", f"a_{fid}"), ("value", f"v_{fid}"), ("category", f"c_{fid}")):
            if statekey in st.session_state:
                fact[key] = st.session_state[statekey]
    st.markdown("**Change a fact**")
    if facts:
        choice = st.selectbox("Choose a fact to update", range(len(facts)),
                              format_func=lambda i: f"{facts[i]['attribute']}: {facts[i]['value']}")
        with st.form("change_fact"):
            new_value = st.text_input("New value")
            if st.form_submit_button("Save change"):
                if new_value.strip():
                    old = facts[choice]
                    old["superseded"] = True
                    facts.append(new_fact(old["entity"], old["attribute"], new_value.strip(), old["category"]))
                    log_pair("Update a fact", f"My {old['attribute']} changed from {old['value']} to {new_value.strip()}")
                    st.success("Updated. The previous value will remain as history.")
    if st.button("Confirm and continue", type="primary"):
        try:
            # Log manual edits as raw messages so RAG / summary can see them too.
            for f in facts:
                if not f["superseded"] and str(f["value"]) != str(f.get("orig", f["value"])):
                    log_pair("Correct a fact", f"Correction: my {f['attribute']} is {f['value']}")
                    f["orig"] = f["value"]
            db = get_db()
            pid = st.session_state.participant_id
            db.graph_facts.delete_many({"participant_id": pid})
            now = datetime.now(timezone.utc)
            docs = [{**{k: v for k, v in f.items() if k != "orig"}, "participant_id": pid,
                     "session_id": st.session_state.session_id, "created_at": now} for f in facts]
            if docs:
                db.graph_facts.insert_many(docs)
            st.session_state.phase = "test"
            st.rerun()
        except Exception:
            safe_notice("We couldn’t save your memory right now. Please try again.")


def facts_for_answer(question):
    allfacts = list(get_db().graph_facts.find({"participant_id": st.session_state.participant_id}, {"_id": 0}))
    # Include all facts in this compact pilot so historical questions can retrieve superseded values.
    return allfacts


# ----- strategy contexts (v2) -------------------------------------------------------------

def _clip(items, limit=MAX_CONTEXT_CHARS):
    """Same context budget for every strategy."""
    out, used = [], 0
    for it in items:
        size = len(json.dumps(it, ensure_ascii=False))
        if used + size > limit:
            break
        out.append(it)
        used += size
    return out


def _chat_pairs():
    return [f"Bot asked: {p['bot']}\nUser said: {p['user']}" for p in st.session_state.chat_log]


def _key(pairs):
    return hashlib.md5("\n".join(pairs).encode()).hexdigest()


def _expanded_keys(pairs):
    """Fact-augmented key expansion (LongMemEval): index KEY = round + extracted facts,
    VALUE returned to the model = the raw round. Falls back to plain rounds on parse failure."""
    numbered = "\n\n".join(f"[{i}]\n{p}" for i, p in enumerate(pairs))
    try:
        text, _ = _counted_llm_text(FACT_KEY_PROMPT.format(rounds=numbered, n=len(pairs)))
        facts = json.loads(text[text.index("["): text.rindex("]") + 1])
        if isinstance(facts, list) and len(facts) == len(pairs):
            return [f"{p}\nFacts: {f}" for p, f in zip(pairs, facts)]
    except Exception:
        pass
    return list(pairs)


def _rag_context(question):
    """Round-level values + fact-augmented keys. Time-aware query expansion: not implemented."""
    pairs = _chat_pairs()
    if not pairs:
        return []
    cache = st.session_state.setdefault("rag_cache", {})
    k = _key(pairs)
    if k not in cache:
        cache[k] = np.array(embed(_expanded_keys(pairs)))
    q = np.array(embed([question])[0])
    M = cache[k]
    sims = M @ q / (np.linalg.norm(M, axis=1) * np.linalg.norm(q) + 1e-9)
    top = np.argsort(-sims)[:RAG_TOP_K]
    return _clip([{"message": pairs[i]} for i in top])


def _summary_context():
    """LLM-written rolling summary of the raw chat, updated chunk by chunk."""
    pairs = _chat_pairs()
    if not pairs:
        return []
    cache = st.session_state.setdefault("summary_cache", {})
    k = _key(pairs)
    if k not in cache:
        summary = "(empty)"
        for i in range(0, len(pairs), SUMMARY_CHUNK):
            chunk = "\n\n".join(pairs[i:i + SUMMARY_CHUNK])
            summary, _ = _counted_llm_text(SUMMARY_UPDATE_PROMPT.format(summary=summary, new=chunk))
        cache[k] = summary
    return [{"summary": cache[k]}]


def _graph_context(facts):
    labeled = [{"entity": f["entity"], "attribute": f["attribute"], "value": f["value"],
                "category": f["category"], "status": "superseded" if f.get("superseded") else "current"}
               for f in facts]
    return _clip(labeled)


def answer_question(question, qtype):
    try:
        # Reserve budget for all four answers BEFORE starting, so every question gets every mode.
        st.session_state.last_error = None
        if not call_ok(len(MEMORY_MODES)):
            return False
        facts = facts_for_answer(question)
        query_id = str(uuid.uuid4())

        contexts = {}
        for mode in MEMORY_MODES:
            if mode == "no_memory":
                contexts[mode] = []
            elif mode == "rag":
                contexts[mode] = _rag_context(question)
            elif mode == "rolling_summary":
                contexts[mode] = _summary_context()
            else:
                contexts[mode] = _graph_context(facts)

        order = random.sample(MEMORY_MODES, len(MEMORY_MODES))
        labels = {m: f"Answer {chr(65 + i)}" for i, m in enumerate(order)}
        in_rate = float(st.secrets.get("COST_PER_1K_INPUT_USD", 0))
        out_rate = float(st.secrets.get("COST_PER_1K_OUTPUT_USD", 0))

        rows = []
        for mode in MEMORY_MODES:
            used = contexts[mode]
            started = time.perf_counter()
            answer, res = llm_text(ANSWER_PROMPT.format(question=question, facts=json.dumps(used, ensure_ascii=False)))
            latency = time.perf_counter() - started
            usage = getattr(res, "usage", None)
            ti = getattr(usage, "prompt_tokens", 0) or 0
            to = getattr(usage, "completion_tokens", 0) or 0
            rows.append({"result_id": str(uuid.uuid4()), "query_id": query_id,
                         "participant_id": st.session_state.participant_id,
                         "session_id": st.session_state.session_id, "question": question, "question_type": qtype,
                         "memory_mode": mode, "impl_version": IMPL_VERSION, "blind_label": labels[mode],
                         "answer": answer, "facts_used": used, "latency_s": latency,
                         "tokens_in": ti, "tokens_out": to, "cost_usd": (ti * in_rate + to * out_rate) / 1000,
                         "accuracy": None, "correct_label": None, "usefulness": None, "intrusive": None,
                         "comment": None, "timestamp": datetime.now(timezone.utc)})
        get_db().pilot_results.insert_many(rows)  # all four modes or none
        st.session_state.answers.extend(rows)
        st.session_state.active_query_id = query_id  # set only once answers exist
        return True
    except Exception:
        import traceback
        st.session_state.last_error = traceback.format_exc()
        safe_notice("I couldn’t answer that just now. Please try another question.")
        return False


def show_last_error():
    """Shows why the last question failed (remove or hide before a wider rollout)."""
    err = st.session_state.get("last_error")
    if err:
        with st.expander("Technical details (for the developer)", expanded=True):
            st.code(err)


def answer_card(i, item):
    label = item.get("blind_label", item["memory_mode"])
    st.markdown(f"**{label}**\n\n**Question:** {item['question']}\n\n**Answer:** {item['answer']}")
    if item.get("rated"):
        # Reveal the mode and what it saw only after the participant has rated it.
        st.caption(f"Memory mode: {item['memory_mode']}")
        with st.expander("Memory inspector"):
            if item["facts_used"]:
                st.dataframe(pd.DataFrame(item["facts_used"]), use_container_width=True, hide_index=True)
            else:
                st.write("No stored facts were available.")
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
                    "memory_mode": item["memory_mode"], "impl_version": IMPL_VERSION, "answer": item["answer"],
                    "correct_label": correct, "accuracy": accuracy, "usefulness": useful, "intrusive": intrusive,
                    "comment": comment, "latency_s": item["latency_s"], "tokens_in": item["tokens_in"],
                    "tokens_out": item["tokens_out"], "cost_usd": item["cost_usd"], "timestamp": datetime.now(timezone.utc)})
                get_db().pilot_results.update_one({"result_id": item["result_id"], "participant_id": st.session_state.participant_id},
                    {"$set": {"correct_label": correct, "accuracy": accuracy, "usefulness": useful, "intrusive": intrusive, "comment": comment}})
                item["rated"] = True
                st.rerun()
            except Exception:
                safe_notice("We couldn’t save that rating. Please try again.")


def run_test():
    st.subheader("Try asking your memory")
    st.write("Each question is answered four different ways. The answers are labeled A–D in random order; rate each one, and the method behind it is shown after you rate it.")
    typ = st.selectbox("Suggested question type", list(SUGGESTIONS))
    if st.button("Try a suggested question"):
        with st.spinner("Asking all four memory modes…"):
            ok = answer_question(SUGGESTIONS[typ], typ.lower().replace(" ", "_"))
        if ok:
            st.rerun()
    with st.form("free_question"):
        q = st.text_input("Or ask your own question")
        asked = st.form_submit_button("Ask")
    if asked and q.strip():
        with st.spinner("Asking all four memory modes…"):
            ok = answer_question(q.strip(), "free")
        if ok:
            st.rerun()
    show_last_error()
    current_items = [(i, item) for i, item in enumerate(st.session_state.answers)
                     if item.get("query_id") == st.session_state.get("active_query_id")]
    current_items.sort(key=lambda t: t[1].get("blind_label", ""))
    for i, item in current_items:
        with st.container(border=True):
            answer_card(i, item)
    render_best_mode_choices("test", st.session_state.get("active_query_id"))
    if st.button("Finish pilot", type="primary"):
        st.session_state.phase = "done"
        st.rerun()


def render_best_mode_choices(area, active_query_id=None):
    groups = {}
    for item in st.session_state.answers:
        if active_query_id and item.get("query_id") != active_query_id:
            continue
        groups.setdefault(item.get("query_id"), []).append(item)
    for query_id, items in groups.items():
        if not query_id or len(items) != len(MEMORY_MODES):
            continue
        label_to_mode = {x.get("blind_label", x["memory_mode"]): x["memory_mode"] for x in items}
        labels = sorted(label_to_mode)
        current_mode = items[0].get("preferred_mode")
        current_label = next((l for l, m in label_to_mode.items() if m == current_mode), None)
        with st.form(f"best_mode_{area}_{query_id}"):
            best_label = st.radio(f"Which answer was best for: {items[0]['question']}", labels,
                                  index=labels.index(current_label) if current_label in labels else 0)
            if st.form_submit_button("Save best answer"):
                try:
                    best = label_to_mode[best_label]
                    get_db().pilot_results.update_many(
                        {"participant_id": st.session_state.participant_id, "query_id": query_id},
                        {"$set": {"preferred_mode": best, "preference_timestamp": datetime.now(timezone.utc)}})
                    for item in items:
                        item["preferred_mode"] = best
                    st.success(f"Saved preference: {best_label}")
                except Exception:
                    safe_notice("We couldn’t save that preference. Please try again.")


def run_chat():
    st.subheader("Chat and update your memory")
    st.caption("Tell the chatbot something new, correct an existing fact, or ask a question. Say “skip” if you do not want to add a fact.")
    for entry in st.session_state.get("chat_messages", []):
        with st.chat_message(entry["role"]):
            st.write(entry["content"])
    active_chat_query_id = st.session_state.get("active_chat_query_id")
    chat_results = [(i, item) for i, item in enumerate(st.session_state.answers)
                    if item.get("question_type") == "chat" and item.get("query_id") == active_chat_query_id]
    chat_results.sort(key=lambda t: t[1].get("blind_label", ""))
    for i, item in chat_results:
        with st.container(border=True):
            answer_card(i, item)
    render_best_mode_choices("chat", active_chat_query_id)
    with st.form("chat_turn", clear_on_submit=True):
        message = st.text_input("Message")
        send = st.form_submit_button("Send")
    if not send or not message.strip():
        return
    msg = message.strip()
    st.session_state.chat_messages = st.session_state.get("chat_messages", [])
    st.session_state.chat_messages.append({"role": "user", "content": msg})
    try:
        if not call_ok():
            return
        parsed, _ = llm_json(EXTRACTION_PROMPT.format(topic="new or changed personal information", message=msg))
        value, attribute = parsed.get("value"), parsed.get("attribute")
        if msg.lower() != "skip" and value and attribute:
            db = get_db()
            pid = st.session_state.participant_id
            current = db.graph_facts.find_one({"participant_id": pid, "attribute": attribute, "superseded": False})
            if not current or str(current.get("value")) != str(value):
                if current:
                    db.graph_facts.update_one({"_id": current["_id"]}, {"$set": {"superseded": True}})
                fact = new_fact(parsed.get("entity"), attribute, value, parsed.get("category"))
                doc = {k: v for k, v in fact.items() if k != "orig"}
                doc.update({"participant_id": pid, "session_id": st.session_state.session_id,
                            "created_at": datetime.now(timezone.utc)})
                db.graph_facts.insert_one(doc)
                st.session_state.facts.append(fact)
        if msg.lower() != "skip":
            log_pair("Tell me something new or ask a question.", msg)  # RAG / summary see it too
        if not answer_question(msg, "chat"):
            show_last_error()
            return
        st.session_state.active_chat_query_id = st.session_state.get("active_query_id")
        recent = st.session_state.answers[-4:]
        reply = next((x["answer"] for x in recent if x["memory_mode"] == "graph_memory"), "I couldn’t answer that from memory.")
        st.session_state.chat_messages.append({"role": "assistant", "content": reply})
        st.rerun()
    except Exception:
        safe_notice("I couldn’t process that chat message. Please try again.")


def run_summary():
    st.subheader("Your pilot summary")
    try:
        pid = st.session_state.participant_id
        results = list(get_db().pilot_results.find({"participant_id": pid}, {"_id": 0}))
        if not results:
            st.info("No answers have been processed yet.")
            return
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
            fact_df = pd.DataFrame(facts)
            st.dataframe(fact_df, use_container_width=True, hide_index=True)
            st.download_button("Download memory facts (CSV)", fact_df.to_csv(index=False), "my_memory_facts.csv", "text/csv")
        else:
            st.info("No memory facts are stored yet.")
        feedback = list(get_db().pilot_feedback.find({"participant_id": pid}, {"_id": 0}))
        if feedback:
            feedback_df = pd.DataFrame(feedback)
            st.subheader("Saved feedback")
            st.dataframe(feedback_df, use_container_width=True, hide_index=True)
            st.download_button("Download feedback (CSV)", feedback_df.to_csv(index=False), "my_pilot_feedback.csv", "text/csv")
    except Exception:
        safe_notice("We couldn’t load your summary right now. Please try again.")


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
            except Exception:
                safe_notice("We couldn’t save your closing feedback. Please try again.")


def main():
    st.title("Memory Lab · Pilot")
    st.caption("Informal pilot, not a controlled study.")
    if not init_runtime():
        return
    page = st.sidebar.radio("Page", ["Pilot", "Chat", "Summary"])
    if page == "Summary":
        run_summary()
        return
    if page == "Chat":
        if st.session_state.phase in ("test", "done"):
            run_chat()
        else:
            st.info("Complete memory review before starting chat mode.")
        return
    {"consent": run_consent, "intake": run_intake, "review": run_review,
     "test": run_test, "done": run_done}.get(st.session_state.phase, run_consent)()


if __name__ == "__main__":
    page = st.navigation([
        st.Page(main, title="Pilot", default=True),
        st.Page(render_results, title="Results", url_path="results"),
    ])
    page.run()