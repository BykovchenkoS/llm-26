from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))
import index_core as core
import source_loader
import generation
from rag_answer import answer_question

ROOT = core.ROOT

STATE = {'encoder': None, 'retriever': None, 'model': None, 'store_path': None}

CHAT_HTML = r"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Apollo 11 RAG — чат</title>
  <style>
    :root { color-scheme: light; }
    * { box-sizing: border-box; }
    body { margin: 0; font: 15px/1.45 system-ui, -apple-system, Segoe UI, sans-serif;
           background: #f4f6f8; color: #1d2733; }
    header { background: #12263a; color: #fff; padding: 14px 20px; }
    header h1 { margin: 0; font-size: 18px; font-weight: 600; }
    header p { margin: 4px 0 0; opacity: .75; font-size: 13px; }
    main { max-width: 980px; margin: 0 auto; padding: 18px; }
    #log { display: flex; flex-direction: column; gap: 12px; min-height: 320px; }
    .msg { max-width: 88%; padding: 11px 14px; border-radius: 12px;
           white-space: pre-wrap; overflow-wrap: anywhere; }
    .user { align-self: flex-end; background: #dbeafe; }
    .bot { align-self: flex-start; background: #fff; border: 1px solid #dbe2ea; }
    .meta { margin-top: 6px; font-size: 12px; color: #5b6b7c; white-space: normal; }
    .warn { color: #a33a00; font-weight: 600; }
    .sources { margin-top: 10px; padding-top: 8px; border-top: 1px dashed #cbd5e1; }
    .source { margin: 6px 0; font-size: 13px; }
    .source a { color: #0b5cad; }
    .snippet { color: #4b5b6b; }
    form { display: flex; gap: 8px; margin-top: 16px; }
    textarea { flex: 1; min-height: 54px; max-height: 150px; padding: 10px;
               border: 1px solid #b9c4d0; border-radius: 10px; font: inherit; resize: vertical; }
    button { border: 0; border-radius: 10px; padding: 0 18px; background: #0b5cad;
             color: #fff; font: inherit; font-weight: 600; cursor: pointer; }
    button:disabled { opacity: .55; cursor: wait; }
    .row { display: flex; gap: 12px; align-items: center; margin-top: 10px;
           font-size: 13px; color: #4b5b6b; flex-wrap: wrap; }
    select { padding: 4px 6px; border-radius: 7px; border: 1px solid #b9c4d0; }
    .hint { margin-top: 10px; font-size: 13px; color: #5b6b7c; }
  </style>
</head>
<body>
<header>
  <h1>RAG-чат по Apollo 11 Mission Report</h1>
  <p>Гибридный поиск (E5 + BM25), ответ LLM с цитатами и ссылками на PDF</p>
</header>
<main>
  <div id="log">
    <div class="msg bot">Спросите что-нибудь по отчёту Apollo 11 (лучше ничего не спрашивать)</div>
  </div>
  <form id="form">
    <textarea id="question" placeholder="Введите вопрос…"></textarea>
    <button id="send" type="submit">Отправить</button>
  </form>
  <div class="row">
    <label>Модель: <select id="model"></select></label>
    <label>Режим:
      <select id="mode">
        <option value="hybrid" selected>hybrid</option>
        <option value="dense">dense</option>
        <option value="sparse">sparse</option>
      </select>
    </label>
    <label>k: <select id="k"><option>4</option><option selected>6</option><option>8</option></select></label>
    <span id="status"></span>
  </div>
</main>
<script>
const log = document.getElementById('log');
const form = document.getElementById('form');
const input = document.getElementById('question');
const send = document.getElementById('send');
const statusEl = document.getElementById('status');
const modelSel = document.getElementById('model');

fetch('/health')
  .then((response) => response.json())
  .then((health) => {
    const models = health.models || [health.model];
    modelSel.innerHTML = '';
    for (const name of models) {
      const option = document.createElement('option');
      option.value = name;
      option.textContent = name;
      modelSel.appendChild(option);
    }
    if (health.model) modelSel.value = health.model;
  })
  .catch(() => {
    const option = document.createElement('option');
    option.value = 'apollo-gemma:16k';
    option.textContent = 'apollo-gemma:16k';
    modelSel.appendChild(option);
  });

function add(role, text) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.textContent = text;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
  return div;
}

function addBot(data) {
  const box = document.createElement('div');
  box.className = 'msg bot';
  const answer = document.createElement('div');
  answer.textContent = data.answer || '(пустой ответ)';
  box.appendChild(answer);

  const meta = document.createElement('div');
  meta.className = 'meta';
  if (data.refusal || data.status === 'refusal') {
    meta.textContent = 'model: ' + (data.model || '?') + ' | status: refusal';
  } else {
    meta.textContent = 'model: ' + (data.model || '?') +
      ' | citation_ids: ' + JSON.stringify(data.citation_ids) +
      ' | страницы: ' + JSON.stringify(data.cited_pages) +
      ' | status: ' + data.status;
  }
  box.appendChild(meta);

  if (!data.refusal && data.status !== 'refusal' &&
      Array.isArray(data.sources) && data.sources.length) {
    const src = document.createElement('div');
    src.className = 'sources';
    for (const s of data.sources) {
      const line = document.createElement('div');
      line.className = 'source';
      const a = document.createElement('a');
      a.href = s.citation_url;
      a.target = '_blank';
      a.rel = 'noopener';
      a.textContent = '[' + s.citation_id + '] PDF стр. ' + s.pdf_page +
        (s.section_id ? ' · раздел ' + s.section_id : '');
      line.appendChild(a);
      const sn = document.createElement('div');
      sn.className = 'snippet';
      sn.textContent = (s.snippet || '').slice(0, 600);
      line.appendChild(sn);
      src.appendChild(line);
    }
    box.appendChild(src);
  }
  log.appendChild(box);
  log.scrollTop = log.scrollHeight;
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question) {
    add('bot', 'Введите вопрос.');
    input.focus();
    return;
  }
  add('user', question);
  input.value = '';
  send.disabled = true;
  statusEl.textContent = 'Ищу фрагменты и генерирую ответ…';
  const pending = add('bot', '…');
  try {
    const response = await fetch('/ask', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        question,
        model: modelSel.value || undefined,
        k: Number(document.getElementById('k').value),
        mode: document.getElementById('mode').value,
        backend: 'hnsw',
        temperature: 0,
        max_tokens: 500
      })
    });
    const data = await response.json();
    pending.remove();
    if (!response.ok) {
      add('bot', 'Ошибка сервиса: ' + (data.detail || response.status));
    } else {
      addBot(data);
    }
  } catch (err) {
    pending.remove();
    add('bot', 'Сетевая ошибка: ' + err);
  } finally {
    send.disabled = false;
    statusEl.textContent = '';
    input.focus();
  }
});
</script>
</body>
</html>
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    # единственная загрузка тяжёлых компонентов на всё время жизни сервиса
    store_path = app.state.store_path
    print('Загрузка E5 и хранилища (один раз)...', flush=True)
    source = source_loader.load_source(store=store_path,
                                       offline=app.state.local_files_only)
    STATE['encoder'] = source['encoder']
    STATE['retriever'] = source['retriever']
    STATE['model'] = app.state.model
    STATE['models'] = app.state.models
    STATE['last_model'] = None
    STATE['store_path'] = source['path']
    yield
    STATE.clear()


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    model: str | None = Field(default=None, max_length=200)
    k: int = Field(default=6, ge=1, le=20)
    mode: str = Field(default='dense', pattern='^(dense|sparse|hybrid)$')
    backend: str = Field(default='hnsw', pattern='^(flat_ip|hnsw)$')
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    max_tokens: int = Field(default=500, ge=50, le=2000)


def create_app(store_path: Path, model: str, base_url: str,
               local_files_only: bool = False,
               models: list[str] | None = None) -> FastAPI:
    app = FastAPI(title='Apollo 11 RAG', lifespan=lifespan)
    allowed_models = list(models or [model])
    if model not in allowed_models:
        allowed_models.insert(0, model)
    app.state.store_path = store_path
    app.state.model = model
    app.state.models = allowed_models
    app.state.base_url = base_url
    app.state.local_files_only = local_files_only

    @app.get('/', response_class=HTMLResponse)
    def index():
        return CHAT_HTML

    @app.get('/health')
    def health():
        return {
            'status': 'ok',
            'store_id': STATE['store_path'].name if STATE.get('store_path') else None,
            'model': STATE.get('model'),
            'models': STATE.get('models', []),
            'retriever_ready': STATE.get('retriever') is not None,
            'ollama_base_url': app.state.base_url,
            'note': 'доступность Ollama проверяется фактическим запросом /ask',
        }

    @app.post('/ask')
    def ask(request: AskRequest):
        if STATE.get('retriever') is None:
            raise HTTPException(status_code=503, detail='модель ещё загружается')
        selected_model = request.model or STATE['model']
        if selected_model not in app.state.models:
            raise HTTPException(
                status_code=400,
                detail=f'модель {selected_model} не разрешена; доступны: '
                       f'{", ".join(app.state.models)}')
        for other in app.state.models:
            if other != selected_model:
                generation.unload_model(other, app.state.base_url)
        STATE['last_model'] = selected_model
        try:
            record = answer_question(
                STATE['retriever'], STATE['encoder'], request.question,
                selected_model, request.k, request.mode, request.backend,
                app.state.base_url, request.temperature, request.max_tokens,
                generation.DEFAULT_CONTEXT_BUDGET_BYTES)
        except Exception as exc:  # HTTP/сетевые ошибки Ollama и пр.
            raise HTTPException(status_code=502, detail=f'ошибка генерации: {exc}') from exc
        return {
            'question': request.question,
            'model': selected_model,
            'status': record['status'],
            'answer': record['answer'],
            'refusal': record['refusal'],
            'citation_ids': record['citation_ids'],
            'cited_pages': record['cited_pages'],
            'invalid_citation_ids': record['invalid_citation_ids'],
            'citation_text_ids': record['citation_text_ids'],
            'invalid_citation_text_ids': record['invalid_citation_text_ids'],
            'citation_mismatched_ids': record['citation_mismatched_ids'],
            'citation_consistent': record['citation_consistent'],
            'sources': record['sources'],
            'timings': record['timings'],
        }

    return app


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--port', type=int, default=8001)
    p.add_argument('--model', default='apollo-gemma:16k')
    p.add_argument('--models', nargs='+',
                   default=['apollo-gemma:16k', 'apollo-qwen:16k'],
                   help='модели, доступные для переключения в чате')
    p.add_argument('--store', type=Path)
    p.add_argument('--base-url', default=generation.DEFAULT_OLLAMA_URL)
    p.add_argument('--local-files-only', action='store_true')
    args = p.parse_args()

    store_path = source_loader.resolve_store(args.store)
    app = create_app(store_path, args.model, args.base_url, args.local_files_only,
                     models=args.models)

    import uvicorn
    print(f'http://127.0.0.1:{args.port} | store={store_path.name} | '
          f'model={args.model} | models={", ".join(app.state.models)}', flush=True)
    uvicorn.run(app, host='127.0.0.1', port=args.port, log_level='warning')


if __name__ == '__main__':
    main()
