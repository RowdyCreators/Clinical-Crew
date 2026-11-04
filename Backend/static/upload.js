const form = document.querySelector('#upload-form')
const submit = document.querySelector('#submit')
const panel = document.querySelector('#progress-panel')
const status = document.querySelector('#status')
const progress = document.querySelector('#progress')
const detail = document.querySelector('#detail')
const result = document.querySelector('#result')
const error = document.querySelector('#error')
const resume = document.querySelector('#resume')
const checkConnection = document.querySelector('#check-connection')
const connectionStatus = document.querySelector('#connection-status')
const labels = { queued: 'Waiting to start', parsing: 'Reading the article', chunking: 'Creating chunks', embedding: 'Generating embeddings', storing: 'Saving to the paper library', completed: 'Completed', failed: 'Processing stopped' }
let timer
let jobId = localStorage.getItem('clinical_ingestion_job')

function busy(value) {
  submit.disabled = value
  form.querySelectorAll('input').forEach(input => { input.disabled = value })
}

function showError(message) { error.textContent = message; error.hidden = false }

async function readResponse(response) {
  const data = await response.json()
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'The upload could not be processed. Check the file and category.')
  return data
}

async function poll() {
  clearTimeout(timer)
  resume.hidden = true
  try {
    const job = await readResponse(await fetch(`/api/v1/ingestions/${jobId}`))
    panel.hidden = false
    status.textContent = labels[job.status] || job.status
    detail.textContent = job.total_chunks ? `${job.completed_chunks} of ${job.total_chunks} chunks embedded.` : 'Preparing your article.'
    if (job.total_chunks) progress.value = Math.round(job.completed_chunks / job.total_chunks * 100)
    else progress.removeAttribute('value')
    if (job.status === 'completed' || job.status === 'failed') {
      busy(false)
      localStorage.removeItem('clinical_ingestion_job')
      if (job.error) showError(job.error)
      if (job.result) {
        const r = job.result
        result.textContent = r.outcome === 'already_indexed'
          ? `“${r.title}” is already indexed: ${r.chunks} chunks. No additional embedding requests were needed.`
          : `“${r.title}” is ready: ${r.chunks} chunks stored in ${r.collection}.`
      }
    } else timer = setTimeout(poll, 1000)
  } catch (err) {
    showError('Could not check progress. Processing may still be running. ' + err.message)
    busy(false)
    resume.hidden = false
  }
}

form.addEventListener('submit', async event => {
  event.preventDefault()
  error.hidden = true
  result.textContent = ''
  const file = form.querySelector('#file').files[0]
  if (!file || !file.name.toLowerCase().endsWith('.xml')) return showError('Choose a PMC/JATS XML file.')
  if (!file.size || file.size > 10 * 1024 * 1024) return showError('Choose a nonempty XML file up to 10 MB.')
  const body = new FormData(form)
  busy(true)
  try {
    const job = await readResponse(await fetch('/api/v1/ingestions', { method: 'POST', body }))
    jobId = job.id
    localStorage.setItem('clinical_ingestion_job', jobId)
    await poll()
  } catch (err) { busy(false); showError(err.message) }
})
resume.addEventListener('click', () => { error.hidden = true; busy(true); poll() })
checkConnection.addEventListener('click', async () => {
  checkConnection.disabled = true
  connectionStatus.textContent = 'Checking Google authentication and model access…'
  try {
    const checked = await readResponse(await fetch('/api/v1/embedding/check', { method: 'POST' }))
    connectionStatus.textContent = `Connected to ${checked.model}. Embeddings use ${checked.dimensions} dimensions.`
  } catch (err) { connectionStatus.textContent = err.message }
  finally { checkConnection.disabled = false }
})
if (jobId) { busy(true); poll() }
