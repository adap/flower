import { FlowerIntelligence, type StreamEvent } from '@flwr/flwr';
import { storage } from 'webextension-polyfill';
import { FLWR_ICON } from './icons';

const fi = FlowerIntelligence.instance;
fi.remoteHandoff = true;
fi.apiKey = (import.meta.env.VITE_API_KEY as string) || '';

interface Message {
  role: string;
  content: string;
}

const SYSTEM_PROMPT: Message = {
  role: 'system',
  content: 'You are a helpful browser assistant.',
};

function getElement(selector: string): HTMLElement {
  const element = document.querySelector(selector);
  if (!element) {
    throw new Error(`Element with selector "${selector}" not found.`);
  }
  return element as HTMLElement;
}

function getPrompts(): HTMLDivElement[] {
  const nodeList = document.querySelectorAll('.prompt');
  const prompts: HTMLDivElement[] = [];

  nodeList.forEach((node) => {
    if (node instanceof HTMLDivElement) {
      prompts.push(node);
    }
  });

  return prompts;
}

const elements = {
  messageHistoryDiv: getElement('#chat-history') as HTMLDivElement,
  sendButton: getElement('#send-button') as HTMLButtonElement,
  messageInput: getElement('#message-input') as HTMLTextAreaElement,
  promptSuggestions: getElement('#prompt-suggestions') as HTMLDivElement,
  optionsButton: getElement('.options-menu') as HTMLDivElement,
  dropdownMenu: getElement('.dropdown-menu') as HTMLDivElement,
  clearHistoryButton: getElement('#clear-history-btn') as HTMLButtonElement,
};

let messageHistory: Message[] = [];

// Load message history from storage on page load
async function initializeHistory() {
  const { messageHistory: storedHistory } = (await storage.local.get('messageHistory')) as {
    messageHistory: Message[] | undefined;
  };
  messageHistory = storedHistory ?? [];
  renderHistory();
}

// Render history from memory to UI
function renderHistory() {
  const historyDiv = elements.messageHistoryDiv;

  // Filter out system messages and specific context messages
  const splitMarker = '\n\nQuestion: ';
  const filteredMessages = messageHistory.map((msg) => {
    if (msg.content.includes(splitMarker)) {
      return { role: msg.role, content: msg.content.split(splitMarker)[1] };
    }
    return { role: msg.role, content: msg.content };
  });

  // Populate the history div with filtered messages, separating user and assistant messages
  historyDiv.innerHTML = filteredMessages
    .map((msg) => {
      const bubbleClass = msg.role === 'user' ? 'user-message' : 'assistant-message';
      const containerClass =
        msg.role === 'user'
          ? ''
          : `<div class="assistant-container"><div class="icon-container">${FLWR_ICON}</div>`;
      return `<div class="message-row">${containerClass}<div class="message-bubble ${bubbleClass}"><div class="bubbleText">${msg.content}</div></div></div>${msg.role === 'user' ? '' : '</div>'}`;
    })
    .join('');

  historyDiv.scrollTop = historyDiv.scrollHeight; // Auto-scroll to the latest message
}

// Append a message to the in-memory history and re-render
function addMessageToHistory(message: Message) {
  messageHistory.push(message);
  renderHistory();
}

// Save the complete message history to storage
async function saveHistoryToStorage() {
  await storage.local.set({ messageHistory });
}

// Clear history both in-memory and in storage
async function clearHistory() {
  messageHistory = [];
  await storage.local.set({ messageHistory: [] });
  await storage.local.set({ fullContent: '' });
  await storage.local.set({ sessionHistory: [] });
  await storage.local.set({
    cachedContext: { currentContent: '', summary: '', contextUrls: [] },
  });
  renderHistory();
}

async function sendMessage() {
  const messageText = elements.messageInput.value.trim();
  if (!messageText) return;

  elements.messageInput.value = '';
  const initialButtonContent = elements.sendButton.innerHTML; // Store initial button content
  elements.sendButton.innerHTML = '<div class="loader"></div>'; // Show loader in the button

  addMessageToHistory({
    role: 'user',
    content: messageText,
  });

  // Make a deep copy of the context before streaming response
  const context = structuredClone(messageHistory);

  const assistantResponse: Message = { role: 'assistant', content: '' };
  addMessageToHistory(assistantResponse); // Add assistant response container for streaming

  // Update the assistant response directly in the UI without caching chunks
  const updateAssistantResponse = (event: StreamEvent) => {
    assistantResponse.content += event.chunk;
    renderHistory();
  };

  fi.remoteHandoff = false;

  if (!fi.remoteHandoff) {
    console.log('Running locally...');
  }

  await fi.chat({
    messages: [SYSTEM_PROMPT, ...context],
    model: 'meta/llama3.2-1b/instruct-fp16',
    stream: true,
    onStreamEvent: updateAssistantResponse,
  });

  // Save the full conversation history to storage after streaming is complete
  elements.sendButton.innerHTML = initialButtonContent;
  await saveHistoryToStorage();
}

// Attach click listeners to SVG icons within the send button
const sendButtonIcons = elements.sendButton.querySelectorAll('svg');
sendButtonIcons.forEach((icon) => {
  icon.addEventListener('click', () => void sendMessage());
});

// Allow pressing Enter to send messages without Shift or modifying the text
elements.messageInput.addEventListener('keydown', (event) => {
  void (async () => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      await sendMessage();
    }
  })();
});

// Show prompt suggestions when the input gains focus
elements.messageInput.addEventListener('focus', () => {
  if (!elements.messageInput.value.trim()) {
    elements.promptSuggestions.classList.add('show');
  }
});

// Hide prompt suggestions on blur with delay to allow click on prompt
elements.messageInput.addEventListener('blur', () => {
  setTimeout(() => {
    elements.promptSuggestions.classList.remove('show');
  }, 150);
});

// Filter prompts based on input as the user types
elements.messageInput.addEventListener('input', () => {
  elements.promptSuggestions.classList.add('show');
  const inputText = elements.messageInput.value.toLowerCase();
  filterPrompts(inputText);
});

// Display only prompts containing the input text
function filterPrompts(input: string) {
  const hasVisiblePrompts = getPrompts().some((prompt) => {
    const promptText = (prompt.textContent ?? '').toLowerCase();
    if (promptText.includes(input)) {
      prompt.style.display = 'inline-flex';
      return true;
    } else {
      prompt.style.display = 'none';
      return false;
    }
  });

  if (!hasVisiblePrompts) {
    elements.promptSuggestions.classList.remove('show');
  }
}

// Insert selected prompt into the message input on click
getPrompts().forEach((prompt) => {
  prompt.addEventListener('click', (event) => {
    elements.messageInput.value = (event.target as HTMLDivElement).textContent ?? '';
    elements.promptSuggestions.classList.remove('show');
    elements.messageInput.focus();
  });
});

// Initialize history on page load
window.addEventListener(
  'load',
  () =>
    void (async () => {
      await initializeHistory();
    })()
);

// Options menu toggle
elements.optionsButton.addEventListener('click', () => {
  elements.dropdownMenu.style.display =
    elements.dropdownMenu.style.display === 'flex' ? 'none' : 'flex';
});

// Close dropdown menu if clicked outside
document.addEventListener('click', (event) => {
  if (
    elements.dropdownMenu.style.display === 'flex' &&
    !elements.optionsButton.contains(event.target as Node) &&
    !elements.dropdownMenu.contains(event.target as Node)
  ) {
    elements.dropdownMenu.style.display = 'none';
  }
});

// Attach event to clear history button in dropdown menu
elements.clearHistoryButton.addEventListener('click', () => {
  void (async () => {
    await clearHistory();
    elements.dropdownMenu.style.display = 'none';
  })();
});
