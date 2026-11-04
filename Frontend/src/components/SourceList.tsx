import type { Source } from '../api'

function sourceLink(s: Source) {
  if (s.url) return s.url
  if (s.doi) return `https://doi.org/${s.doi}`
  return undefined
}

interface Props {
  messageId: string
  sources: Source[]
  open: boolean
  onToggle: () => void
}

export default function SourceList({ messageId, sources, open, onToggle }: Props) {

  return (
    <div className="sources">
      <button className="sources-toggle" onClick={onToggle} aria-expanded={open}>
        {open ? 'Hide' : 'Show'} {sources.length} research excerpt{sources.length === 1 ? '' : 's'}
      </button>
      <ol className={`source-list${open ? ' open' : ''}`}>
        {sources.map((s, i) => {
          const link = sourceLink(s)
          return (
            <li key={s.chunk_id} id={`${messageId}-src-${i + 1}`} className="source">
              <div className="source-head">
                <span className="num">{i + 1}</span>
                <div>
                  {link ? (
                    <a href={link} target="_blank" rel="noreferrer" className="source-title">
                      {s.title ?? 'Untitled paper'}
                    </a>
                  ) : (
                    <span className="source-title">{s.title ?? 'Untitled paper'}</span>
                  )}
                  <div className="meta">
                    {[s.journal, s.year, s.heading].filter(Boolean).join(' · ')}
                  </div>
                </div>
                <span className="score" title="Similarity between your description and this excerpt">
                  {Math.round(s.relevance * 100)}%
                </span>
              </div>
              <p className="excerpt">{s.excerpt}</p>
              {s.disease && <span className="tag">{s.disease.replace(/_/g, ' ')}</span>}
            </li>
          )
        })}
      </ol>
    </div>
  )
}
