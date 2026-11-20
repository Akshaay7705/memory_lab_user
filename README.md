# Memory Lab Streamlit Pilot

An informal pilot app built with Streamlit, Groq, and MongoDB. It evaluates four memory modes for each participant question: `no_memory`, `rag`, `rolling_summary`, and `graph_memory`. Participants can review and rate each answer, choose which mode answered best, switch to chat to add or correct facts, view a personal summary, and export their processed data.

Run with `streamlit run pilot.py`. Configure `.streamlit/secrets.toml` (never commit this file):

```toml
MONGODB_URI = "mongodb+srv://..."
MONGODB_DATABASE = "memory_lab"
GROQ_API_KEY = "..."
GROQ_MODEL = "openai/gpt-oss-120b"
# Configure actual rates for the selected Groq model to estimate dollar cost:
COST_PER_1K_INPUT_USD = 0.0
COST_PER_1K_OUTPUT_USD = 0.0
```

## Mongo collections

- `graph_facts`: confirmed participant facts and their history. Fields include `participant_id`, `entity`, `attribute`, `value`, `category`, `session_id`, `superseded`, and `created_at`. Chat updates mark the prior value superseded and insert the current value.
- `pilot_results`: one row for each question and memory mode. Stores the question, mode, answer, facts supplied to that mode, latency, token counts, estimated cost, participant's best-mode selection, and accuracy/feedback fields. Accuracy is calculated from participant ratings (Correct = 1, Partly correct = 0.5, Incorrect = 0; Not sure is excluded from the average).
- `pilot_feedback`: one row per rated answer, including the participant's correctness label, usefulness score, intrusive flag, comment, timing, token counts, and cost estimate.

The participant Summary page shows the mode-level averages and best-mode selections. It provides CSV/JSON downloads for answer results, plus CSV downloads for confirmed memory facts and feedback. Cost estimates use the per-token rates configured above; set them to the rates for the selected model. Participant IDs are random UUIDs; the app does not request names, emails, or IP addresses.

The app reads `GROQ_API_KEY`, `GROQ_MODEL`, and the MongoDB URI from Streamlit secrets. Do not enter sensitive personal information. This is an informal pilot, not a controlled study.
