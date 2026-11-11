import { useState } from 'react'
import type { ClarifyingQuestion } from '../api'

interface Props {
  questions: ClarifyingQuestion[]
  /** Only the latest, unanswered set of questions accepts input. */
  active: boolean
  onAnswer: (text: string) => void
  onSkip: () => void
}

export default function ClarifyingQuestions({ questions, active, onAnswer, onSkip }: Props) {
  const [picked, setPicked] = useState<Record<number, string>>({})
  const answered = questions.filter((_, i) => picked[i])

  function send() {
    const text = questions
      .map((q, i) => (picked[i] ? `${q.question} ${picked[i]}` : ''))
      .filter(Boolean)
      .join('\n')
    if (text) onAnswer(text)
  }

  return (
    <div className="clarify">
      <ol>
        {questions.map((q, i) => (
          <li key={i}>
            <p>{q.question}</p>
            {q.options.length > 0 && (
              <div className="options" role="group" aria-label={q.question}>
                {q.options.map((option) => (
                  <button
                    key={option}
                    type="button"
                    className={picked[i] === option ? 'selected' : undefined}
                    aria-pressed={picked[i] === option}
                    disabled={!active}
                    onClick={() =>
                      setPicked((prev) => ({ ...prev, [i]: prev[i] === option ? '' : option }))
                    }
                  >
                    {option}
                  </button>
                ))}
              </div>
            )}
          </li>
        ))}
      </ol>
      {active && (
        <div className="clarify-actions">
          <button type="button" className="primary" disabled={!answered.length} onClick={send}>
            Send answers
          </button>
          <button type="button" className="ghost" onClick={onSkip}>
            Skip questions
          </button>
          <span>Or type your answers below.</span>
        </div>
      )}
    </div>
  )
}
