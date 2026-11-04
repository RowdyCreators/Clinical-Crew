import { useEffect, useRef, useState, type FormEvent, type KeyboardEvent } from 'react'

const MAX_CHARS = 2000

interface Props {
  disabled: boolean
  onSubmit: (text: string) => void
}

export default function Composer({ disabled, onSubmit }: Props) {
  const [text, setText] = useState('')
  const ref = useRef<HTMLTextAreaElement>(null)

  // Grow the textarea with its content, up to the CSS max-height.
  useEffect(() => {
    const el = ref.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${el.scrollHeight}px`
  }, [text])

  function send(e?: FormEvent) {
    e?.preventDefault()
    if (disabled || !text.trim()) return
    onSubmit(text)
    setText('')
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) send(e)
  }

  return (
    <form className="composer" onSubmit={send}>
      <textarea
        ref={ref}
        rows={1}
        value={text}
        maxLength={MAX_CHARS}
        placeholder="Describe your symptoms…"
        aria-label="Describe your symptoms"
        onChange={(e) => setText(e.target.value)}
        onKeyDown={onKeyDown}
        autoFocus
      />
      <button type="submit" className="send" disabled={disabled || !text.trim()} aria-label="Send">
        <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
          <path d="M3.4 20.4 21 12 3.4 3.6 3.4 10l12.6 2-12.6 2z" fill="currentColor" />
        </svg>
      </button>
    </form>
  )
}
