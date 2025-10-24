import { FlowerIntelligence } from '@flwr/flwr';

const fi = FlowerIntelligence.instance;
fi.remoteHandoff = true;

// MODIFY VALUES BELOW
fi.apiKey = process.env.FI_API_KEY ?? '';

async function main() {
  const response = await fi.chat({
    messages: [{ role: 'user', content: 'How are you?' }],
    forceRemote: true,
    encrypt: true,
  });

  if (!response.ok) {
    console.error(`${response.failure.code}: ${response.failure.description}`);
  } else {
    console.log(response.message.content);
  }
}

await main().then().catch();
