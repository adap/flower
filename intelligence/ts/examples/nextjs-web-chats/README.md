# Chat App

A Next.js application (using the App Router, TypeScript, and TailwindCSS) demonstrating three distinct web chat approaches with the [@flwr/flwr](https://www.npmjs.com/package/@flwr/flwr) package.

## Minimum Requirements

- **Node.js** (v14+ recommended)
- **npm** or **yarn**
- **Next.js** (using the App Router)
- **TypeScript**
- **TailwindCSS**
- **@flwr/flwr** package installed

## Setup & Installation

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

## Chat Pages Specificities

### 1. **API Chat (/api-chat)**

- **Approach:** Client component that sends questions to an API route.
- **Functionality:**
  - Uses local state to maintain the chat log.
  - Calls the `/api/chat` endpoint which internally uses shared history (via `lib/chat.ts`) and the @flwr/flwr client.
  - Renders messages with Markdown formatting.
- **Use Case:** Ideal for separating client and server logic with API calls.

### 2. **Server-side Chat (/server-side-chat)**

- **Approach:** Server-rendered page using server actions.
- **Functionality:**
  - Renders chat history on the server.
  - Uses a client component (`ChatForm.tsx`) to handle input submission.
  - The send button is disabled and a loading indicator is displayed (using `useTransition`) while waiting for @flwr/flwr to process the response.
  - Maintains global conversation history.
- **Note:** Server components do not support client-side interactivity (like transitions), so the form is split into a client component.
- **Use Case:** Perfect when you need server-side rendering for SEO or performance while still retaining interactive elements.

### 3. **Client-side Chat (/client-side-chat)**

- **Approach:** Fully client-side component that directly consumes the @flwr/flwr client.
- **Functionality:**
  - Uses React hooks (state and effects) to handle messages.
  - Directly calls the FlowerIntelligence client from `@flwr/flwr` using the shared conversation history from `lib/chat.ts`.
  - Renders a consistent chat UI with Markdown formatting.
- **Use Case:** Ideal when you want to handle all interactions on the client for a more dynamic experience.

### 4. **Client-side No History Chat (/client-side-no-history-chat)**

- **Approach:** Fully client-side component that directly consumes the @flwr/flwr client without utilizing a shared or persistent conversation history.
- **Functionality:**
  - Uses React hooks (state and effects) to manage messages independently for each session.
  - Calls the FlowerIntelligence client from `@flwr/flwr` on each request without appending messages to a global history.
  - Renders a transient chat UI where each new interaction is processed without prior context.
- **Use Case:** Ideal for scenarios where a stateless chat experience is desired, such as quick tests or single-turn interactions without historical context.
