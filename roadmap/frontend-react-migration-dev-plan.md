# Frontend React Migration — Development Plan

**Created:** 2026-04-30
**Source:** [frontend-react-migration.md](./frontend-react-migration.md)
**Estimated Total:** 6–9 days

---

## Phase 0: Project Scaffold (Day 1)

**Outcome:** Vite + React 18 + TypeScript (strict) project bootstrapped and integrated with the FastAPI backend.

### Tasks

| # | Task | Details |
|---|---|---|
| 0.1 | Init Vite project | `npm create vite@latest frontend -- --template react-ts` under the repo root |
| 0.2 | Install dependencies | React 18, React DOM, and dev tooling (ESLint, Prettier) |
| 0.3 | Styling setup | Decide between CSS Modules and Tailwind; configure accordingly. Existing CSS variables should map cleanly to either choice |
| 0.4 | Directory structure | `src/api/`, `src/components/`, `src/hooks/`, `src/utils/`, `src/pages/`, `src/types/` |
| 0.5 | Backend integration | Two approaches (choose one before starting Phase 1):<br>• **Option A (recommended):** Serve `dist/` as `StaticFiles` in `main_service.py` — single port, no extra container<br>• **Option B:** Add `nginx` service in `docker-compose.yml` — cleaner separation, needs CORS (already configured) |
| 0.6 | Verify dev loop | `vite dev` running, proxy to backend API works, hot reload confirmed |

---

## Phase 1: API Client + Shared Types (Day 1)

**Outcome:** Typed API client and TypeScript interfaces shared between both pages.

### Tasks

| # | Task | Details |
|---|---|---|
| 1.1 | Define types | `src/types/api.ts` — interfaces for `QueryRequest`, `QueryResponse`, `StreamEvent`, `HistoryMessage`, `FileUploadState`, `HealthStatus` |
| 1.2 | Build API client | `src/api/client.ts` — typed wrapper around native `fetch` and `EventSource` with all endpoints (health, query, stream, history, clearHistory, upload, status) |
| 1.3 | Env configuration | `VITE_API_BASE` for the backend URL, defaulting to `''` (same-origin) for Option A |

---

## Phase 2: Import Page Migration (Days 2–3)

**Outcome:** `import.html` fully replaced by a React page. Low complexity, good warm-up before the chat page.

### Components

| Component | Responsibility |
|---|---|
| `DropZone` | Drag-over, drag-leave, drop events, click-to-open file input dialog |
| `FileItem` | Status badge, progress bar, expandable log accordion |
| `ImportPage` | Page layout, composes `DropZone` + `FileItem` list |

### Hook

| Hook | Responsibility |
|---|---|
| `useFileUpload` | `POST /upload` → receive `task_id` → `setInterval` polling of `GET /status/{task_id}` → updates `FileUploadState[]` |

### Tasks

| # | Task | Details |
|---|---|---|
| 2.1 | `useFileUpload` hook | Encapsulate upload + polling logic. `useEffect` cleanup clears the interval to avoid memory leaks. Return `{ files, addFiles, clearFiles }` |
| 2.2 | `DropZone` component | Drag-and-drop area with visual feedback on drag-over / drag-leave. Falls back to hidden `<input type="file">` for click-to-select |
| 2.3 | `FileItem` component | Renders status badge (color-coded), progress bar (animated width), and collapsible log list from `doneList` / `runningList` |
| 2.4 | `ImportPage` component | Composes the above. Replaces the single `import.html` page |
| 2.5 | Remove old code | Delete `import.html` and any orphaned vanilla JS once verified |

### State Shape

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

---

## Phase 3: Chat Page — Static Rendering (Days 3–4)

**Outcome:** Chat UI renders correctly for non-streaming paths (history load, blocking queries). No SSE yet.

### Components

| Component | Responsibility |
|---|---|
| `TopBar` | Brand/logo, stream toggle switch, API status pill (health check), clear history button |
| `ChatWindow` | Scrollable container for the message list, auto-scrolls to bottom on new messages |
| `MessageBubble` | Single message — user (right-aligned) or bot (left-aligned). Renders text, images, progress details |
| `ProgressDetails` | Expandable `<details>` element showing done/running node lists |
| `AnswerImages` | Image grid with lazy loading and fallback links |
| `Composer` | Textarea + Send button. Enter to send, Shift+Enter for newline |
| `TypingIndicator` | Three-dot bounce animation (placeholder during streaming) |

### Hooks

| Hook | Responsibility |
|---|---|
| `useSession` | Read/write `sessionId` from `localStorage`, generate new UUID on first visit |
| `useHistory` | `GET /history/{session_id}` on mount, return `ChatMessage[]` |

### Tasks

| # | Task | Details |
|---|---|---|
| 3.1 | `useSession` hook | Persist session ID in `localStorage`. Generate UUID if none exists |
| 3.2 | `useHistory` hook | Fetch history on mount, transform API response into `ChatMessage[]` |
| 3.3 | `TopBar` component | Stream toggle (saved to `localStorage`), health check pill (polling every 5s via `useEffect` + `setInterval`), clear history button |
| 3.4 | `MessageBubble` component | Role-based alignment and styling. Renders text content. Bot messages show `ProgressDetails` and `AnswerImages` when present |
| 3.5 | `ProgressDetails` component | Collapsible `<details>`/`<summary>` with done/running lists |
| 3.6 | `AnswerImages` component | Lazy-loaded `<img>` grid. Fallback to `<a>` link if image fails to load |
| 3.7 | `Composer` component | Controlled textarea. Enter submits, Shift+Enter inserts newline. Disabled while streaming |
| 3.8 | `ChatWindow` component | Maps `messages[]` to `MessageBubble` components. Auto-scroll on new message via `useRef` + `scrollIntoView` |
| 3.9 | `ChatPage` component | Page-level composition of `TopBar` + `ChatWindow` + `Composer` |
| 3.10 | Blocking query flow | `POST /query` without stream — append user message, wait for response, append bot message |

### State Shape

```ts
type MessageRole = 'user' | 'bot';

interface ChatMessage {
  id: string;
  role: MessageRole;
  text: string;
  imageUrls: string[];
  ts?: number;
  // bot-only fields
  doneList?: string[];
  runningList?: string[];
  pipelineStatus?: 'pending' | 'processing' | 'completed' | 'failed';
  isStreaming?: boolean;
}
```

---

## Phase 4: Chat Page — SSE Streaming (Days 5–6)

**Outcome:** Real-time streaming responses with live text accumulation and progress updates.

### Hook

| Hook | Responsibility |
|---|---|
| `useSSE` | Open `EventSource` on `/stream/{session_id}`, dispatch typed events (`progress`, `delta`, `final`, `final_answer`, `error`), close and cleanup on unmount |

### Tasks

| # | Task | Details |
|---|---|---|
| 4.1 | `useSSE` hook | Wraps `EventSource` in `useEffect`. Listens for 5 event types. Cleans up on unmount / session change. Exposes callbacks or returns latest event state |
| 4.2 | Streaming skeleton pattern | On query submit: append a `ChatMessage` with `isStreaming: true`. Accumulate deltas into `rawText` via `useRef`. Flush to state at event boundaries (`final`, `final_answer`) or throttled intervals. When complete, set `isStreaming: false` |
| 4.3 | `TypingIndicator` integration | Show `TypingIndicator` when the streaming bot message has no text yet. Replace with accumulated text on first delta |
| 4.4 | Progress event handling | During streaming, update `doneList` / `runningList` on the in-flight message. `ProgressDetails` re-renders reactively |
| 4.5 | `useQuerySubmit` hook | `POST /query` → branches to SSE path or blocking path based on stream toggle. Manages the submit → streaming → complete lifecycle |
| 4.6 | Render optimization | Use `React.memo` on `MessageBubble` for completed messages. State updates only mutate the in-flight streaming message via `setMessages(msgs => msgs.map(...))` |

### SSE Event Map

| Event | Action |
|---|---|
| `progress` | Update `doneList` / `runningList` on the in-flight bot message |
| `delta` | Accumulate to `rawText` ref, flush to message `text` periodically |
| `final` | Flush any remaining delta, mark `pipelineStatus: 'completed'` |
| `final_answer` | Replace or append the final answer text |
| `error` | Mark `pipelineStatus: 'failed'`, display error text |

---

## Phase 5: Image Utilities + AnswerImages (Day 6)

**Outcome:** Image URL extraction from answer text and rendering, ported 1:1 from vanilla JS.

### Tasks

| # | Task | Details |
|---|---|---|
| 5.1 | Port utility functions | `src/utils/imageUtils.ts` — `parseAnswerAndImages`, `extractUrlsLoose`, `isImageUrl`, `normalizeUrl`, `dedupeKeepOrder`. Add TypeScript types. No logic changes |
| 5.2 | Unit tests | Test `imageUtils.ts` in isolation before wiring into components. Cover: markdown image syntax, bare URLs, malformed URLs, duplicates |
| 5.3 | Wire into `MessageBubble` | Pass extracted image URLs to `AnswerImages`. Handle both markdown image syntax and the backend `image_urls` list |
| 5.4 | `AnswerImages` component | Lazy load with `loading="lazy"`. `onerror` fallback to a clickable link. Responsive grid layout |

---

## Phase 6: History + Clear + Health Check (Day 6–7)

**Outcome:** History loading and clearing work correctly with the React state model.

### Tasks

| # | Task | Details |
|---|---|---|
| 6.1 | History loading | `useEffect` on mount calls `GET /history/{session_id}`, populates `messages[]` state. Handles empty state gracefully |
| 6.2 | Clear history | `DELETE /history/{session_id}` → filter `messages[]` to remove bot messages (or clear entirely, match current behavior) |
| 6.3 | Health check polling | `setInterval` every 5s inside `useEffect` in `TopBar`. Updates status pill (green / red). Cleanup on unmount |

---

## Phase 7: Integration Testing + Cleanup (Day 8)

**Outcome:** End-to-end verification against a running backend, old code removed.

### Tasks

| # | Task | Details |
|---|---|---|
| 7.1 | End-to-end test | Full flow: upload file → query → SSE stream → image rendering → history reload → clear. Test against running backend |
| 7.2 | Error states | Test: network down, 500 responses, malformed SSE events, empty history, invalid session ID |
| 7.3 | Remove old code | Delete `app/import_process/pages/import.html` and `app/query_process/page/chat.html` once verified |
| 7.4 | Update routes | If using Option A, add `StaticFiles` mount for `dist/` in `main_service.py`. Remove old HTML-serving routes |

---

## Phase 8: Production Build + Deploy (Day 9, Optional)

**Outcome:** Optimized production build served by the existing backend.

### Tasks

| # | Task | Details |
|---|---|---|
| 8.1 | Production build | `vite build` — verify output size, code splitting, asset hashing |
| 8.2 | Backend integration | Mount `dist/` as `StaticFiles` in FastAPI. Serve `index.html` for SPA fallback routes |
| 8.3 | Docker update | If using Docker, update `Dockerfile` to include the `vite build` step and copy `dist/` |
| 8.4 | Smoke test | Deploy, verify both pages load and function correctly in production mode |

---

## Risk Mitigation Checklist

| Risk | Action |
|---|---|
| SSE + React state causing excessive re-renders | Use `useRef` for delta accumulation, flush to state only at event boundaries or throttled intervals |
| Image rendering regression | Unit-test `imageUtils.ts` before wiring into components |
| Session ID mismatch after HMR in dev | `useSession` reads from `localStorage` on every mount — no in-memory caching |

---

## Dependency Graph

```
Phase 0 (Scaffold)
  └─> Phase 1 (API Client + Types)
        ├─> Phase 2 (Import Page) ─────────────────────────┐
        └─> Phase 3 (Chat — Static)                         │
              └─> Phase 4 (Chat — SSE)                      │
                    └─> Phase 5 (Image Utils)               │
                          └─> Phase 6 (History + Clear)     │
                                └─> Phase 7 (Integration)   │
                                      └─> Phase 8 (Deploy)  │
                                                             │
        Phase 2 (Import Page) ──────────────────────────────┘
        (can run in parallel with Phases 3–6)
```
