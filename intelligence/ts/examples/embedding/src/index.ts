import { FlowerIntelligence } from '@flwr/flwr';

const fi = FlowerIntelligence.instance;
fi.remoteHandoff = true;
fi.apiKey = process.env.FI_API_KEY ?? '';

async function main() {
  const response = await fi.embed({
    model: 'qwen/qwen3-embedding',
    input: 'Hello, world!',
  });

  if (!response.ok) {
    console.error(response.failure.description);
    process.exit(1);
  } else {
    console.log(response.value);
  }
}

await main().then().catch();
