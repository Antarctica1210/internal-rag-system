# Frontend Migration: Vanilla HTML → React + TypeScript

**Scope:** Convert `app/import_process/pages/import.html` and `app/query_process/page/chat.html` into a React + TypeScript single-page application.

**Date:** 2026-04-30

---

## Summary

| Page | Lines | Complexity | Estimate |
|---|---|---|---|
| `import.html` | ~320 | Low | 1–2 days |
| `chat.html` | ~895 | Medium-High | 4–6 days |
| Setup + shared code | — | — | 1 day |
| **Total** | | | **6–9 days** |

---

## `import.html` — Low Complexity (1–2 days)

The logic is straightforward and maps cleanly to React patterns.

### Current behaviour
- Drag-and-drop / click-to-select file upload
- Per-file status badge, progress bar, and expandable log list
- `setInterval` polling of `GET /status/{task_id}` after upload

### React decomposition

| Component | Responsibility |
|---|---|
| `DropZone` | Drag-over, drag-leave, drop, click-to-open file input |
| `FileItem` | Status badge, progress bar, log accordion |
| `useFileUpload` hook | `POST /upload` → receive `task_id` → start polling → update state |

### State shape
```ts
interface FileUploadState {
  id: string;
  name: string;
  sizeKb: number;
  status: 'uploading' | 'processing' | 'completed' | 'failed';
  progressPct: number;
  doneList: string[];
  runningList: string[];
}
```

### Notes
- DOM mutation via `insertAdjacentHTML` becomes a `useState<FileUploadState[]>` array.
- `setInterval` polling moves into `useEffect` with a cleanup return to avoid memory leaks.
- No tricky logic — this page is a direct 1:1 port.

---

## `chat.html` — Medium-High Complexity (4–6 days)

Significantly larger. Key challenges are SSE streaming and image URL rendering.

### Current behaviour
- Session persistence via `localStorage`
- History loaded on mount from `GET /history/{session_id}`
- Health check pill polling every 5 s
- Stream toggle (SSE vs blocking)
- SSE events: `progress`, `delta`, `final`, `final_answer`, `error`
- In-place DOM updates during streaming (typing indicator → accumulated text → final answer)
- Image URL extraction from answer text + backend `image_urls` list
- Clear history (DELETE + DOM cleanup)

### React decomposition

| Component | Responsibility |
|---|---|
| `TopBar` | Brand, stream toggle, API status pill, clear button |
| `ChatWindow` | Scrollable message list |
| `MessageBubble` | Renders user or bot bubble; accepts `role`, `text`, `imageUrls`, `ts` |
| `TypingIndicator` | Three-dot bounce animation (shown during streaming skeleton) |
| `ProgressDetails` | Expandable `<details>` with done/running node lists |
| `AnswerImages` | Image grid with lazy-load and fallback link |
| `Composer` | Textarea + Send button, Enter/Shift+Enter handling |

### Custom hooks

| Hook | Responsibility |
|---|---|
| `useSession` | Read/write `sessionId` from `localStorage` |
| `useHistory` | Fetch conversation history on mount |
| `useSSE` | Open `EventSource`, dispatch typed events, close and cleanup on unmount |
| `useQuerySubmit` | `POST /query`, branch to SSE or blocking path |

### State shape
```ts
type MessageRole = 'user' | 'bot';

interface ChatMessage {
  id: string;
  role: MessageRole;
  text: string;
  imageUrls: string[];
  ts?: number;
  // bot-only
  doneList?: string[];
  runningList?: string[];
  pipelineStatus?: 'pending' | 'processing' | 'completed' | 'failed';
  isStreaming?: boolean;
}
```

### Area-by-area breakdown

#### SSE handling (~1 day)
The five event listeners (`progress`, `delta`, `final`, `final_answer`, `error`) become a `useSSE` hook.
The main risk: React state updates during streaming must be batched to avoid excessive re-renders.
Use a single `rawText` ref for accumulation and only call `setState` on meaningful deltas or event boundaries.

```ts
// sketch
function useSSE(sessionId: string, onDelta, onFinal, onProgress, onError) {
  useEffect(() => {
    const es = new EventSource(`${API_BASE}/stream/${sessionId}`);
    es.addEventListener('delta', ...);
    es.addEventListener('final', ...);
    // ...
    return () => es.close(); // cleanup on unmount
  }, [sessionId]);
}
```

#### Image URL parsing (~0.5 days)
`parseAnswerAndImages`, `extractUrlsLoose`, `isImageUrl`, `normalizeUrl`, `dedupeKeepOrder` translate 1:1 to pure TypeScript utility functions in `src/utils/imageUtils.ts`. No logic changes required, just type annotations.

#### Message list state (~1 day)
Currently all state is implicit in the DOM (elements found by random ID / CSS class).
In React this becomes an explicit `ChatMessage[]` array — the biggest structural change, and a direct improvement in correctness.
The streaming skeleton pattern becomes: append a `isStreaming: true` message, then mutate it in-place via `setMessages(msgs => msgs.map(...))`.

#### History + clear (~0.5 days)
`loadHistory` becomes a `useEffect` on mount. Clear calls `DELETE /history/{session_id}`, then filters the message array.

---

## Project setup (1 day, one-time)

### Recommended stack
| Tool | Choice |
|---|---|
| Bundler | Vite |
| Framework | React 18 |
| Language | TypeScript (strict) |
| Styling | CSS Modules or Tailwind (existing CSS variables map cleanly to Tailwind tokens) |
| HTTP | Native `fetch` (no extra library needed) |
| SSE | Native `EventSource` |

### Serving options
Two approaches for integrating with the FastAPI backend:

**Option A — Static build served by FastAPI (simpler)**
Build the React app (`vite build`) and mount the `dist/` folder as `StaticFiles` in `main_service.py`.
No separate container. Single port (8000) serves both API and UI.

**Option B — Separate frontend container (cleaner separation)**
Add a `frontend` service to `docker-compose.yml` running `nginx` to serve the built assets.
CORS is already configured on the backend (`allow_origins=["*"]`).

Option A is recommended for an internal tool at this scale.

### Shared API client
```ts
// src/api/client.ts
const API_BASE = import.meta.env.VITE_API_BASE ?? '';

export const api = {
  health: () => fetch(`${API_BASE}/health`),
  query: (body: QueryRequest) => fetch(`${API_BASE}/query`, { method: 'POST', ... }),
  stream: (sessionId: string) => new EventSource(`${API_BASE}/stream/${sessionId}`),
  history: (sessionId: string, limit = 50) => fetch(`${API_BASE}/history/${sessionId}?limit=${limit}`),
  clearHistory: (sessionId: string) => fetch(`${API_BASE}/history/${sessionId}`, { method: 'DELETE' }),
  upload: (files: File[]) => { /* FormData */ },
  status: (taskId: string) => fetch(`${API_BASE}/status/${taskId}`),
};
```

---

## Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| SSE + React state causing excessive re-renders during streaming | Medium | High | Accumulate delta in a `useRef`, flush to state only on `final` or at throttled intervals |
| Image rendering logic regression | Low | Medium | Port `imageUtils.ts` with unit tests before wiring into components |
| Session ID mismatch after hot reload in dev | Low | Low | `useSession` hook reads from `localStorage` on every mount |

---

## Suggested implementation order

1. Project scaffold (Vite + React + TS)
2. Shared API client + TypeScript types for all API responses
3. `import.html` migration (simpler — good warm-up)
4. Chat: static message rendering (no streaming yet)
5. Chat: SSE `useSSE` hook + streaming skeleton
6. Chat: image URL utilities + `AnswerImages` component
7. Chat: history load, clear, health check
8. Integration test end-to-end with running backend
9. (Optional) `vite build` → mount as `StaticFiles` in `main_service.py`
