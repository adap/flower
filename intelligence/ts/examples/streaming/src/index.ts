import { FlowerIntelligence, type StreamEvent } from '@flwr/flwr';

const fi = FlowerIntelligence.instance;

async function main() {
  const response = await fi.chat({
    messages: [
      { role: 'system', content: 'You are a helpful assistant' },
      { role: 'user', content: 'How are you?' },
    ],
    stream: true,
    onStreamEvent: (event: StreamEvent) => {
      process.stdout.write(event.chunk);
    },
  });

  if (!response.ok) {
    console.error(response.failure.description);
  } else {
    console.log(`\n\nComplete reply: ${response.message.content}`);
  }
}

await main().then().catch();
