import { useState } from 'react'
import Markdown from 'react-markdown'
import type { Message } from '../App'
import ClarifyingQuestions from './ClarifyingQuestions'
import SourceList from './SourceList'

// Turn "[2]" or "[1, 3]" citations into links that jump to the matching source cards.
function linkCitations(text: string, messageId: string, count: number) {
  return text.replace(/\[(\d+(?:\s*,\s*\d+)*)\](?!\()/g, (match, list: string) => {
    const nums = list.split(',').map((n) => Number(n.trim()))
    if (!nums.every((n) => n >= 1 && n <= count)) return match
    return nums.map((n) => `[\\[${n}\\]](#${messageId}-src-${n})`).join('')
  })
}

interface Props {
  message: Message
  /** True for the latest reply while its follow-up questions are still open. */
  active?: boolean
  onAnswer?: (text: string) => void
  onSkip?: () => void
}

export default function MessageBubble({ message, active = false, onAnswer, onSkip }: Props) {
  const { id, role, content, sources = [], error, clarification } = message
  const [sourcesOpen, setSourcesOpen] = useState(false)

  function showSource(anchor: string) {
    setSourcesOpen(true)
    // Wait one frame so the list is expanded before scrolling to the card.
    requestAnimationFrame(() => {
      const target = document.getElementById(anchor)
      if (!target) return
      target.scrollIntoView({ behavior: 'smooth', block: 'center' })
      target.classList.add('flash')
      setTimeout(() => target.classList.remove('flash'), 1200)
    })
  }

  if (role === 'user') {
    return (
      <div className="row user">
        <div className="bubble">{content}</div>
      </div>
    )
  }

  return (
    <div className="row assistant">
      <div className={`bubble${error ? ' error' : ''}`}>
        {error ? (
          <p>{content}</p>
        ) : (
          <div className="markdown">
            <Markdown
              components={{
                a: ({ href, children }) =>
                  href?.startsWith('#') ? (
                    <a
                      href={href}
                      className="cite"
                      onClick={(e) => {
                        e.preventDefault()
                        showSource(href.slice(1))
                      }}
                    >
                      {children}
                    </a>
                  ) : (
                    <a href={href} target="_blank" rel="noreferrer">
                      {children}
                    </a>
                  ),
              }}
            >
              {linkCitations(content, id, sources.length)}
            </Markdown>
          </div>
        )}
        {clarification && clarification.questions.length > 0 && (
          <ClarifyingQuestions
            questions={clarification.questions}
            active={active}
            onAnswer={(text) => onAnswer?.(text)}
            onSkip={() => onSkip?.()}
          />
        )}
        {sources.length > 0 && (
          <SourceList
            messageId={id}
            sources={sources}
            open={sourcesOpen}
            onToggle={() => setSourcesOpen((o) => !o)}
          />
        )}
      </div>
    </div>
  )
}
