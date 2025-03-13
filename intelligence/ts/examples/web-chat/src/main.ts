import { FlowerIntelligence, Progress, StreamEvent } from '@flwr/flwr';

const MODEL = 'meta/llama3.2-1b/instruct-fp16';
// const MODEL= 'meta/llama3.2-3b/instruct-q4';
// const MODEL= 'meta/llama3.1-8b/instruct-q4';
// const MODEL= 'deepseek/r1';

const fi = FlowerIntelligence.instance;

async function sendChat(message: string, func: (text: string) => void): Promise<string> {
  const response = await fi.chat({
    messages: [
      { role: 'system', content: 'You are a helpful assistant' },
      { role: 'user', content: message },
    ],
    stream: true,
    onStreamEvent: (event: StreamEvent) => func(event.chunk),
    model: MODEL,
  });
  if (!response.ok) {
    console.error(`${response.failure.code}: ${response.failure.description}`);
    return 'Error';
  }
  return response.message.content;
}

const chatInput = document.getElementById('chatInput') as HTMLInputElement;
const sendButton = document.getElementById('sendButton') as HTMLButtonElement;
const loadButton = document.getElementById('loadButton') as HTMLButtonElement;
const chatLog = document.getElementById('chatLog') as HTMLDivElement;
const loading = document.getElementById('loading') as HTMLDivElement;
const title = document.getElementById('title') as HTMLHeadingElement;
title.textContent = title.textContent + ` (${MODEL})`;

sendButton.addEventListener('click', async () => {
  const message = chatInput.value.trim();
  if (!message) return;

  appendToChatLog(`You: ${message}`);
  chatInput.value = '';

  appendToChatLog('Bot: ');
  await sendChat(message, (input: string) => appendToChatLog(input, true));
  chatLog.scrollTop = chatLog.scrollHeight;
});

loadButton.addEventListener('click', async () => {
  fi.fetchModel(MODEL, (progress: Progress) => {
    loading.textContent = progress.description ?? '';
  });
});

function appendToChatLog(text: string, stream: boolean = false) {
  if (stream && chatLog.lastChild) {
    chatLog.lastChild.textContent = chatLog.lastChild.textContent + text;
  } else {
    const p = document.createElement('p');
    p.textContent = text;
    chatLog.appendChild(p);
    chatLog.scrollTop = chatLog.scrollHeight;
  }
}
