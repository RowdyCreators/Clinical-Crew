import { useEffect, useRef, useState } from 'react'
import { fetchCategories, sendMessage, type ClarifyingQuestion, type Source, type Turn } from './api'
import Composer from './components/Composer'
import MessageBubble from './components/MessageBubble'

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
  error?: boolean
  /** Set when the assistant asked follow-up questions to tell several possible diseases apart. */
  clarification?: {
    candidates: string[]
    questions: ClarifyingQuestion[]
    round: number
  }
}

const SUGGESTIONS = [
  'I have been very thirsty, urinating often, and feel tired all the time.',
  'Rash shaped like a bullseye after a hike, with fever and joint pain.',
  'My father keeps forgetting recent conversations and gets lost in familiar places.',
]

const MAX_HISTORY = 12
const MAX_TURN_CHARS = 8000

let nextId = 0
const newId = () => `m${Date.now()}-${nextId++}`

const SKIP_MESSAGE = 'Please skip the questions and show me the possible conditions now.'

// The questions are shown as buttons, so add them to the text the model sees in later turns.
function turnContent(m: Message) {
  const questions = m.clarification?.questions ?? []
  const text = questions.length
    ? `${m.content}\n\n${questions
        .map((q, i) => `${i + 1}. ${q.question}${q.options.length ? ` (${q.options.join(' / ')})` : ''}`)
        .join('\n')}`
    : m.content
  return text.slice(0, MAX_TURN_CHARS)
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [pending, setPending] = useState(false)
  const [categories, setCategories] = useState<string[]>([])
  const [category, setCategory] = useState('')
  const abortRef = useRef<AbortController | null>(null)
  const endRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    fetchCategories().then(setCategories).catch(() => setCategories([]))
  }, [])

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages, pending])

  // The open follow-up questions, if the latest assistant reply asked any.
  const lastReply = [...messages].reverse().find((m) => m.role === 'assistant' && !m.error)
  const openClarification = lastReply?.clarification ? lastReply : undefined

  async function submit(text: string, { clarify = true }: { clarify?: boolean } = {}) {
    const message = text.trim()
    if (!message || pending) return

    // Earlier successful turns give the model context for follow-up questions.
    const history: Turn[] = messages
      .filter((m) => !m.error)
      .slice(-MAX_HISTORY)
      .map((m) => ({ role: m.role, content: turnContent(m) }))
    const pendingCase = openClarification?.clarification

    setMessages((prev) => [...prev, { id: newId(), role: 'user', content: message }])
    setPending(true)
    const controller = new AbortController()
    abortRef.current = controller

    try {
      const result = await sendMessage(
        {
          message,
          history,
          disease: category || null,
          candidates: pendingCase?.candidates ?? [],
          clarify_round: pendingCase?.round ?? 0,
          clarify,
        },
        controller.signal,
      )
      setMessages((prev) => [
        ...prev,
        {
          id: newId(),
          role: 'assistant',
          content: result.answer,
          sources: result.sources,
          clarification:
            result.kind === 'clarification'
              ? {
                  candidates: result.candidates,
                  questions: result.questions,
                  round: result.clarify_round,
                }
              : undefined,
        },
      ])
    } catch (error) {
      if (controller.signal.aborted) return
      setMessages((prev) => [
        ...prev,
        {
          id: newId(),
          role: 'assistant',
          content: error instanceof Error ? error.message : 'Something went wrong.',
          error: true,
        },
      ])
    } finally {
      if (abortRef.current === controller) {
        abortRef.current = null
        setPending(false)
      }
    }
  }

  function resetChat() {
    abortRef.current?.abort()
    abortRef.current = null
    setPending(false)
    setMessages([])
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo" aria-hidden="true">+</span>
          <div>
            <h1>Clinical Crew</h1>
            <p>Symptom analysis grounded in indexed research papers</p>
          </div>
        </div>
        <div className="controls">
          <label className="select">
            <span>Category</span>
            <select value={category} onChange={(e) => setCategory(e.target.value)}>
              <option value="">All categories</option>
              {categories.map((c) => (
                <option key={c} value={c}>
                  {c.replace(/_/g, ' ')}
                </option>
              ))}
            </select>
          </label>
          <button className="ghost" onClick={resetChat} disabled={!messages.length && !pending}>
            New chat
          </button>
        </div>
      </header>

      <main className="chat" aria-live="polite">
        <div className="chat-inner">
          {messages.length === 0 && (
            <section className="welcome">
              <h2>Describe what you're experiencing</h2>
              <p>
                Include the symptoms, when they started, how severe they are, and anything that makes
                them better or worse. Answers cite the research excerpts they rely on.
              </p>
              <div className="suggestions">
                {SUGGESTIONS.map((s) => (
                  <button key={s} onClick={() => submit(s)}>
                    {s}
                  </button>
                ))}
              </div>
            </section>
          )}

          {messages.map((m) => (
            <MessageBubble
              key={m.id}
              message={m}
              active={m === openClarification && m === messages[messages.length - 1] && !pending}
              onAnswer={(text) => submit(text)}
              onSkip={() => submit(SKIP_MESSAGE, { clarify: false })}
            />
          ))}

          {pending && (
            <div className="row assistant">
              <div className="bubble typing" aria-label="Analyzing symptoms">
                <span />
                <span />
                <span />
                <em>Searching papers and analyzing…</em>
              </div>
            </div>
          )}
          <div ref={endRef} />
        </div>
      </main>

      <footer className="footer">
        <Composer disabled={pending} onSubmit={submit} />
        <p className="disclaimer">
          For informational and research purposes only. Not a medical diagnosis. If you have severe
          symptoms, contact emergency services.
        </p>
      </footer>
    </div>
  )
}
