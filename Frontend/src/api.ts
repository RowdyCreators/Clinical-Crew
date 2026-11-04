export type Role = 'user' | 'assistant'

export interface Source {
  chunk_id: string
  paper_id?: string
  disease?: string
  title?: string
  authors?: string
  year?: number
  journal?: string
  doi?: string
  url?: string
  heading?: string
  relevance: number
  excerpt: string
}

export interface ChatResponse {
  answer: string
  model: string
  sources: Source[]
}

export interface Turn {
  role: Role
  content: string
}

export interface ChatRequest {
  message: string
  history: Turn[]
  disease?: string | null
  top_k?: number
}

async function errorMessage(response: Response): Promise<string> {
  try {
    const body = await response.json()
    if (typeof body.detail === 'string') return body.detail
    if (Array.isArray(body.detail) && body.detail[0]?.msg) return body.detail[0].msg
  } catch {
    /* not JSON */
  }
  if (response.status === 502 || response.status === 504) {
    return 'Cannot reach the Clinical Crew backend. Is the API server running on port 8010?'
  }
  return `Request failed (HTTP ${response.status}).`
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(path, init)
  } catch {
    throw new Error('Cannot reach the Clinical Crew backend. Is the API server running?')
  }
  if (!response.ok) throw new Error(await errorMessage(response))
  return response.json() as Promise<T>
}

export function sendMessage(body: ChatRequest, signal?: AbortSignal): Promise<ChatResponse> {
  return request<ChatResponse>('/api/v1/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  })
}

export async function fetchCategories(): Promise<string[]> {
  const data = await request<{ categories: string[] }>('/api/v1/chat/categories')
  return data.categories
}
