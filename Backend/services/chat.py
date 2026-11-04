"""Symptom question -> retrieved paper excerpts -> grounded answer from Gemini."""
import logging
import time

from Backend.config import ANSWER_MODEL
from Backend.services import embeddings
from Backend.services.embeddings import EmbeddingError

log = logging.getLogger('chat')

MAX_EXCERPT_CHARS = 1200
GENERATE_ATTEMPTS = 3  # waits 3s, then 6s, on rate limits / temporary outages

SYSTEM_PROMPT = """You are Clinical Crew, a clinical decision-support assistant. A user describes symptoms;
you receive numbered excerpts from peer-reviewed research papers retrieved for that description.

Rules:
- Base your reasoning ONLY on the numbered excerpts. Cite them inline as [1], [2], etc.
- Give a short list of the most likely conditions the excerpts support, most likely first. For each,
  explain which of the user's symptoms match the evidence and which do not.
- If the excerpts do not support any conclusion, say so plainly instead of guessing.
- Mention symptoms that would require urgent or emergency care if they appear in the evidence or are
  obviously serious (for example chest pain, difficulty breathing, confusion, fainting).
- Suggest reasonable next steps, such as which clinician to see or which tests the evidence mentions.
- Never present the answer as a definitive diagnosis. End with one sentence reminding the user to
  consult a qualified healthcare professional.
- Format the answer in concise Markdown with short headings or bullet points."""


class ChatError(ValueError):
    """Safe message for the browser; never includes credentials or request URLs."""


class AnswerError(RuntimeError):
    """Safe answer-model failure message (HTTP 503 to the browser)."""


def retrieve(question, k=6, disease=None, collection=None, client=None):
    collection = collection if collection is not None else embeddings.get_collection()
    if collection.count() == 0:
        raise ChatError('The paper library is empty. Upload XML papers before asking questions.')
    owns_client = client is None
    client = client or embeddings.make_client()
    try:
        return embeddings.search(collection, client, question, k=k, disease=disease)
    finally:
        if owns_client:
            client.close()


def build_prompt(question, history, sources):
    excerpts = '\n\n'.join(
        f"[{i}] {s.get('title', 'Untitled')} ({s.get('year', 'n.d.')})"
        f"{' - ' + s['heading'] if s.get('heading') else ''}\n{s['text'][:MAX_EXCERPT_CHARS]}"
        for i, s in enumerate(sources, 1))
    turns = '\n'.join(f"{t['role'].upper()}: {t['content']}" for t in history)
    parts = [f'Research excerpts:\n{excerpts}']
    if turns:
        parts.append(f'Earlier conversation:\n{turns}')
    parts.append(f'USER: {question}')
    return '\n\n'.join(parts)


def answer_error(error):
    code = getattr(error, 'code', None)
    if code == 429:
        message = (f'The answering model ({ANSWER_MODEL}) is rate-limited or out of quota (HTTP 429). '
                   'Wait a minute and retry, or check the Gemini quota for this key in Google AI Studio.')
    elif code in (401, 403):
        message = f'Google rejected the answering request (HTTP {code}). Check the Gemini API key and its access.'
    elif code == 404:
        message = f'The answering model {ANSWER_MODEL!r} was not found. Set ANSWER_MODEL in .env to an available model.'
    else:
        label = f'HTTP {code}' if isinstance(code, int) else 'provider error'
        message = f'Answer generation failed ({label}). Check Gemini access and network connectivity, then retry.'
    return AnswerError(message)


def generate(prompt, client=None):
    from google.genai import types
    owns_client = client is None
    client = client or embeddings.make_client()
    try:
        for attempt in range(GENERATE_ATTEMPTS):
            try:
                result = client.models.generate_content(
                    model=ANSWER_MODEL, contents=prompt,
                    config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.2))
                break
            except Exception as error:
                code = getattr(error, 'code', None)
                if code not in (429, 500, 503) or attempt == GENERATE_ATTEMPTS - 1:
                    raise answer_error(error) from None
                wait = 3 * 2 ** attempt
                log.warning('Temporary answer-model failure (HTTP %s); retrying in %ds', code, wait)
                time.sleep(wait)
    finally:
        if owns_client:
            client.close()
    text = (result.text or '').strip()
    if not text:
        raise ChatError('The answering model returned an empty response. Try rephrasing the question.')
    return text


def public_source(source):
    keep = ('chunk_id', 'paper_id', 'disease', 'title', 'authors', 'year', 'journal', 'doi', 'url', 'heading')
    item = {k: source[k] for k in keep if source.get(k) not in (None, '')}
    item['relevance'] = round(max(0.0, 1.0 - float(source['distance'])), 3)
    item['excerpt'] = source['text'][:MAX_EXCERPT_CHARS]
    return item


def answer(question, history=(), k=6, disease=None):
    """Return the grounded answer plus the excerpts it was based on."""
    sources = retrieve(question, k=k, disease=disease)
    if not sources:
        raise ChatError('No matching research excerpts were found. Try another category or more detail.')
    text = generate(build_prompt(question, list(history), sources))
    return {'answer': text, 'model': ANSWER_MODEL, 'sources': [public_source(s) for s in sources]}


def categories():
    collection = embeddings.get_collection()
    metadatas = collection.get(include=['metadatas'])['metadatas']
    return sorted({m['disease'] for m in metadatas if m.get('disease')}, key=str.lower)


__all__ = ['AnswerError', 'ChatError', 'EmbeddingError', 'answer', 'categories']
