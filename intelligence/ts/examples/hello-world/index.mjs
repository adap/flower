import { FlowerIntelligence } from '../../dist/flowerintelligence.es.js';

const fi = FlowerIntelligence.instance;

async function main() {
  const response = await fi.chat({
    messages: [
      { role: 'system', content: 'You are a helpful assistant' },
      { role: 'user', content: 'How are you?' },
    ],
  });
  if (!response.ok) {
    console.error(`${response.failure.code}: ${response.failure.description}`);
    process.exit(1);
  } else {
    console.log(response.message.content);
  }
}

await main().then().catch();
