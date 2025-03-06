import { FlowerIntelligence } from '@flwr/flwr';
import { isProbablyReaderable, Readability } from '@mozilla/readability';
import { storage, history, tabs, runtime, Runtime } from 'webextension-polyfill';

const fi = FlowerIntelligence.instance;
fi.remoteHandoff = true;
fi.apiKey = (import.meta.env.VITE_API_KEY as string) || '';

const CONTEXT_SIZE = 10;

async function processHistoryUrlsOnStartup(): Promise<void> {
  const { sessionHistory = [] } = (await storage.local.get({
    sessionHistory: [],
  })) as { sessionHistory: string[] };
  const additionalUrlsNeeded = CONTEXT_SIZE - sessionHistory.length;

  if (additionalUrlsNeeded > 0) {
    const additionalUrls = await getAdditionalUrls(sessionHistory.length);

    for (const url of additionalUrls) {
      // Check if we have already enough URLs in context
      if (sessionHistory.length >= CONTEXT_SIZE) break;

      await handlePageVisit(url); // Summarize and add each URL to context
    }
  }
}

// Get additional URLs from browsing history
async function getAdditionalUrls(sessionLength: number): Promise<string[]> {
  const historyUrlsNeeded = CONTEXT_SIZE - sessionLength;
  if (historyUrlsNeeded > 0) {
    const historyItems = await history.search({
      text: '',
      maxResults: historyUrlsNeeded,
    });
    return historyItems.reduce<string[]>((urls, item) => {
      if (item.url !== undefined) {
        urls.push(item.url);
      }
      return urls;
    }, []);
  }
  return [];
}

async function getPageContentByUrl(url?: string): Promise<string> {
  if (!url || url.startsWith('about:')) return '';
  try {
    // Fetch the page HTML
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Failed to fetch URL: ${response.statusText}`);

    const html = await response.text();

    // Parse the HTML into a document
    const doc = new DOMParser().parseFromString(html, 'text/html');

    if (isProbablyReaderable(doc)) {
      // Use Readability to extract the main content
      const reader = new Readability(doc);
      const article = reader.parse();

      return article
        ? article.textContent
            .replace(/\s+/g, ' ')
            .replace(/\n{3,}/g, '\n\n')
            .trim()
        : '';
    } else {
      return '';
    }
  } catch {
    console.error(`Error fetching or processing URL: ${url}`);
    return '';
  }
}

// Incrementally update the session summary and cache it
async function updateSessionSummary(url: string): Promise<void> {
  const content = await getPageContentByUrl(url);
  let summary = `No summary for ${url}.\n`;
  if (content !== '') {
    const summaryResponse = await fi.chat({
      messages: [
        {
          role: 'user',
          content: `Can you write a short summary of the following webpage:\n${content}`,
        },
      ],
      model: 'meta/llama3.2-1b/fp16',
      forceRemote: true,
    });
    if (summaryResponse.ok && summaryResponse.message.content) {
      summary = `Summary of ${url}:\n${summaryResponse.message.content}\n`;
    } else if (!summaryResponse.ok) {
      console.error(
        `Summary for ${url} failed with error code ${summaryResponse.failure.code.toString()}: ${summaryResponse.failure.description}`
      );
    }
  }

  // Retrieve existing cached summary
  const { fullContent = '' } = (await storage.local.get({
    fullContent: '',
  })) as { fullContent: string };
  const updatedFullContent = `${fullContent}\n${summary}`;

  // Update the cache with the new summary
  await storage.local.set({ fullContent: updatedFullContent });
}

// Call this function every time a page is visited to update the session context
async function handlePageVisit(url: string): Promise<void> {
  const { sessionHistory = [] } = (await storage.local.get({
    sessionHistory: [],
  })) as { sessionHistory: string[] };

  if (!sessionHistory.includes(url)) {
    sessionHistory.push(url);
    await storage.local.set({ sessionHistory });

    await updateSessionSummary(url); // Update the summary incrementally
    await cacheContext(url);
  }
}

// Store the full context in cache for quick access by popup.ts
async function cacheContext(currentUrl?: string): Promise<void> {
  const { fullContent } = (await storage.local.get({
    fullContent: '',
  })) as { fullContent: string };
  const { sessionHistory } = (await storage.local.get({
    sessionHistory: [],
  })) as { sessionHistory: string[] };
  const additionalUrls = await getAdditionalUrls(sessionHistory.length);
  const contextUrls = [...sessionHistory, ...additionalUrls].slice(0, CONTEXT_SIZE);

  await storage.local.set({
    cachedContext: {
      currentContent: await getPageContentByUrl(currentUrl),
      summary: fullContent,
      contextUrls,
    },
  });
}

async function initializeContextOnStartup(): Promise<void> {
  await processHistoryUrlsOnStartup(); // Process history URLs to prefill context
  await cacheContext(); // Cache the initial context
}

tabs.onUpdated.addListener((_tabId, changeInfo, tab) => {
  if (changeInfo.status === 'complete' && tab.url) {
    void handlePageVisit(tab.url);
  }
});

runtime.onMessage.addListener(
  (
    message: {
      type: string;
      currentUrl: string;
    },
    _sender: Runtime.MessageSender,
    _: (response?: unknown) => void
  ) => {
    if (message.type === 'updateCurrentPage') {
      return cacheContext(message.currentUrl);
    }
  }
);

void initializeContextOnStartup();
