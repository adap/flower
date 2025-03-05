import { FlowerIntelligence, type StreamEvent } from '@flwr/flwr';

const fi = FlowerIntelligence.instance;
fi.remoteHandoff = true;
fi.apiKey = process.env.FI_API_KEY ?? 'REPLACE_HERE';

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
    encrypt: true,
    forceRemote: true,
  });

  if (!response.ok) {
    console.error(`${response.failure.code}: ${response.failure.description}`);
  } else {
    console.log(`\n\nComplete reply: ${response.message.content ?? 'None'}`);
  }
}

await main().then().catch();
