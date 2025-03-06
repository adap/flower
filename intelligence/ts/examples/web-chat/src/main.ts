import { FlowerIntelligence, Progress } from '@flwr/flwr';

const fi = FlowerIntelligence.instance;

async function sendChat(message: string): Promise<string> {
  const response = await fi.chat({
    messages: [
      { role: 'system', content: 'You are a helpful assistant' },
      { role: 'user', content: message },
    ],
    // model: 'meta/llama3.2-1b/instruct-fp16',
    // model: 'meta/llama3.2-3b/instruct-q4',
    // model: 'meta/llama3.1-8b/instruct-q4',
    // model: 'deepseek/r1',
    forceLocal: true,
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

sendButton.addEventListener('click', async () => {
  const message = chatInput.value.trim();
  if (!message) return;

  appendToChatLog(`You: ${message}`);
  chatInput.value = '';

  const response = await sendChat(message);
  appendToChatLog(`Bot: ${response}`);
});

loadButton.addEventListener('click', async () => {
  fi.fetchModel('meta/llama3.2-1b/instruct-fp16', (progress: Progress) => {
    // To change later
    console.log(progress.description);
    console.log(progress.loadedBytes);
    console.log(progress.totalBytes);
    console.log(progress.percentage);
  });
});

function appendToChatLog(text: string) {
  const p = document.createElement('p');
  p.textContent = text;
  chatLog.appendChild(p);
  chatLog.scrollTop = chatLog.scrollHeight;
}
