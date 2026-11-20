"""Prompts used by Memory Lab Pilot."""

EXTRACTION_PROMPT = """Extract at most one personal fact from the participant's latest message.
Return ONLY valid JSON with keys entity, attribute, value, category. Use nulls if no
fact is stated. Categories must be one of Location, Work/School, People, Interests,
Preferences, Other. Never infer sensitive personal information. Do not include prose.

Topic: {topic}
Message: {message}
"""

ANSWER_PROMPT = """Answer the participant's question using only the supplied memory facts.
Treat current facts as current and superseded facts as history. If the facts do not
support an answer, say you do not know. Be concise and friendly.
Question: {question}
Facts: {facts}
"""
