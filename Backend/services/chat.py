"""Symptom question -> retrieved paper excerpts -> grounded answer from Gemini.

When a new symptom description fits more than one indexed disease, a triage step asks targeted follow-up
questions to narrow the possibilities down before the grounded answer is written.
"""
import json
import logging
import math
import time

from Backend.config import ANSWER_MODEL
from Backend.services import embeddings
from Backend.services.embeddings import EmbeddingError

log = logging.getLogger('chat')

MAX_EXCERPT_CHARS = 1200
GENERATE_ATTEMPTS = 3  # waits 3s, then 6s, on rate limits / temporary outages

# Differential clarification
MAX_CLARIFY_ROUNDS = 2   # follow-up rounds before answering with whatever candidates remain
MAX_QUESTIONS = 3
POOL_SIZE = 24           # nearest chunks inspected to find which diseases are in play
PER_DISEASE = 3          # excerpts per disease shown to the triage step
MAX_TRIAGE_SOURCES = 12
MAX_CASE_CHARS = 1800    # retrieval query built from the symptoms plus the user's answers

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
- Only discuss conditions from the indexed categories named in the excerpt labels. If the user answered
  follow-up questions, use those answers to explain why the remaining condition(s) fit best.
- Format the answer in concise Markdown with short headings or bullet points."""

TRIAGE_PROMPT = """You are the triage step of Clinical Crew, a clinical decision-support assistant. You receive the
user's symptom description (and any answers they gave to earlier follow-up questions) plus numbered excerpts from
research papers. Each excerpt is labelled with the condition category it belongs to. Only the listed candidate
conditions may be considered; never introduce any other condition.

Decide which candidate conditions the user's symptoms are genuinely consistent with, according to the excerpts.
Leave out a candidate whose excerpts were retrieved but whose features do not fit what the user described, or that
the user's answers have ruled out.

Fill the JSON fields:
- matching: the consistent candidates, most likely first, each with a one-sentence reason citing excerpts as [1], [2]. Address the user
  as "you" (for example "Your fatigue and joint aches match..."), never as "the user".
- questions: only when two or more candidates match, 1 to 3 short, plain-language questions about the user's own
  symptoms or history whose answers would best tell those candidates apart, based on distinguishing features the
  excerpts describe (onset and timing, specific accompanying signs, exposures, progression). Give each question 2 to
  4 short answer options, including an "unsure" style option when sensible. Never ask about lab results, never ask
  something the user already answered, and return an empty list when fewer than two candidates match.
- urgent: true if the description contains symptoms that need urgent or emergency care."""


class ChatError(ValueError):
    """Safe message for the browser; never includes credentials or request URLs."""


class AnswerError(RuntimeError):
    """Safe answer-model failure message (HTTP 503 to the browser)."""


def disease_filter(disease=None, diseases=None):
    if disease:
        return {'disease': disease}
    if diseases:
        return {'disease': diseases[0]} if len(diseases) == 1 else {'disease': {'$in': list(diseases)}}
    return None


def retrieve(question, k=6, disease=None, diseases=None, collection=None, client=None):
    collection = collection if collection is not None else embeddings.get_collection()
    if collection.count() == 0:
        raise ChatError('The paper library is empty. Upload XML papers before asking questions.')
    owns_client = client is None
    client = client or embeddings.make_client()
    try:
        return embeddings.search(collection, client, question, k=k, where=disease_filter(disease, diseases))
    finally:
        if owns_client:
            client.close()


def balance(sources, per_disease, limit):
    """Nearest excerpts first, but at most `per_disease` from each disease so every candidate is represented."""
    counts, picked = {}, []
    for source in sorted(sources, key=lambda s: float(s['distance'])):
        name = source.get('disease')
        if counts.get(name, 0) < per_disease:
            counts[name] = counts.get(name, 0) + 1
            picked.append(source)
            if len(picked) == limit:
                break
    return picked


def diseases_of(sources):
    return list(dict.fromkeys(s['disease'] for s in sources if s.get('disease')))


def case_query(question, history):
    """Symptoms plus the user's answers so far, so retrieval for a short reply like "yes" keeps its context."""
    text = ' '.join([t['content'] for t in history if t['role'] == 'user'] + [question])
    return text[-MAX_CASE_CHARS:]


def label(name):
    return name.replace('_', ' ')


def build_prompt(question, history, sources):
    excerpts = '\n\n'.join(
        f"[{i}] {s.get('title', 'Untitled')} ({s.get('year', 'n.d.')})"
        f"{' - ' + s['heading'] if s.get('heading') else ''}"
        f"{' [category: ' + s['disease'] + ']' if s.get('disease') else ''}\n{s['text'][:MAX_EXCERPT_CHARS]}"
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


def generate(prompt, client=None, system=SYSTEM_PROMPT, schema=None):
    from google.genai import types
    owns_client = client is None
    client = client or embeddings.make_client()
    extra = {'response_mime_type': 'application/json', 'response_json_schema': schema} if schema else {}
    try:
        for attempt in range(GENERATE_ATTEMPTS):
            try:
                result = client.models.generate_content(
                    model=ANSWER_MODEL, contents=prompt,
                    config=types.GenerateContentConfig(system_instruction=system, temperature=0.2, **extra))
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


def triage_schema(allowed):
    return {
        'type': 'object',
        'properties': {
            'matching': {'type': 'array', 'items': {
                'type': 'object',
                'properties': {'condition': {'type': 'string', 'enum': list(allowed)},
                               'reason': {'type': 'string'}},
                'required': ['condition', 'reason']}},
            'questions': {'type': 'array', 'maxItems': MAX_QUESTIONS, 'items': {
                'type': 'object',
                'properties': {'question': {'type': 'string'},
                               'options': {'type': 'array', 'items': {'type': 'string'}, 'maxItems': 4}},
                'required': ['question', 'options']}},
            'urgent': {'type': 'boolean'},
        },
        'required': ['matching', 'questions', 'urgent'],
    }


def triage(question, history, sources, allowed):
    """Ask the model which of the allowed (retrieved) diseases fit, and what to ask to tell them apart.

    Returns None when the reply cannot be used, so the caller falls back to a normal answer.
    """
    prompt = f"Candidate conditions: {', '.join(allowed)}\n\n" + build_prompt(question, history, sources)
    raw = generate(prompt, system=TRIAGE_PROMPT, schema=triage_schema(allowed))
    try:
        data = json.loads(raw)
    except ValueError:
        log.warning('Triage reply was not JSON; answering directly')
        return None
    if not isinstance(data, dict):
        return None
    lookup = {name.lower(): name for name in allowed}
    matching, seen = [], set()
    for item in data.get('matching') or []:
        if not isinstance(item, dict):
            continue
        name = lookup.get(str(item.get('condition', '')).strip().lower())
        if name and name not in seen:  # only diseases the project actually retrieved
            seen.add(name)
            matching.append({'condition': name, 'reason': str(item.get('reason', '')).strip()})
    questions = []
    for item in (data.get('questions') or [])[:MAX_QUESTIONS]:
        if not isinstance(item, dict):
            continue
        text = str(item.get('question', '')).strip()
        options = [str(o).strip() for o in (item.get('options') or []) if str(o).strip()][:4]
        if text:
            questions.append({'question': text, 'options': options})
    return {'matching': matching, 'questions': questions, 'urgent': bool(data.get('urgent'))}


def clarification_text(matching):
    lines = ['Your symptoms are consistent with more than one condition covered by the indexed research:', '']
    lines += [f"- **{label(m['condition']).capitalize()}**" + (f" - {m['reason']}" if m['reason'] else '')
              for m in matching]
    lines += ['', 'To narrow this down, please answer a few more questions about your symptoms.']
    return '\n'.join(lines)


def response(text, sources, **extra):
    return {'answer': text, 'model': ANSWER_MODEL, 'sources': [public_source(s) for s in sources],
            'kind': 'answer', 'candidates': [], 'questions': [], 'clarify_round': 0, **extra}


def answer(question, history=(), k=6, disease=None, candidates=None, clarify_round=0, clarify=True):
    """Return the grounded answer plus the excerpts it was based on.

    On a new case (first message, or answers to earlier follow-up questions) whose symptoms fit more than one
    indexed disease, return follow-up questions instead (`kind == 'clarification'`) together with the diseases
    still in play. The client sends those back as `candidates` along with the user's answers.
    """
    history = list(history)
    candidates = [] if disease else list(dict.fromkeys(c for c in candidates or () if c))
    query = case_query(question, history) if candidates else question
    new_case = not history or bool(candidates)

    if clarify and new_case and not disease and clarify_round < MAX_CLARIFY_ROUNDS:
        pool = retrieve(query, k=POOL_SIZE, diseases=candidates or None)
        in_play = diseases_of(pool)
        if len(in_play) > 1:
            sources = balance(pool, PER_DISEASE, MAX_TRIAGE_SOURCES)
            result = triage(question, history, sources, in_play)
            if result and result['matching']:
                matching = result['matching']
                if len(matching) > 1 and result['questions'] and not result['urgent']:
                    names = [m['condition'] for m in matching]
                    # Keep every excerpt so the [n] citations in the reasons still line up.
                    return response(clarification_text(matching), sources, kind='clarification',
                                    candidates=names, questions=result['questions'],
                                    clarify_round=clarify_round + 1)
                candidates = [m['condition'] for m in matching]
        elif in_play:
            candidates = in_play

    if len(candidates) > 1:
        pool = retrieve(query, k=max(k, POOL_SIZE), diseases=candidates)
        sources = balance(pool, math.ceil(k / len(candidates)), k)
    else:
        sources = retrieve(query, k=k, disease=disease or (candidates[0] if candidates else None))
    if not sources:
        raise ChatError('No matching research excerpts were found. Try another category or more detail.')
    text = generate(build_prompt(question, history, sources))
    return response(text, sources, candidates=candidates)


def categories():
    collection = embeddings.get_collection()
    metadatas = collection.get(include=['metadatas'])['metadatas']
    return sorted({m['disease'] for m in metadatas if m.get('disease')}, key=str.lower)


__all__ = ['AnswerError', 'ChatError', 'EmbeddingError', 'answer', 'categories']
