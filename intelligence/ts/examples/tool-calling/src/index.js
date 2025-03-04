import { FlowerIntelligence } from '@flwr/flwr';

const fi = FlowerIntelligence.instance;
fi.remoteHandoff = true;
fi.apiKey = process.env.FI_API_KEY ?? 'REPLACE_HERE';

function draftEmail({ receiver, content }) {
  console.log(`Email to ${receiver}:\n${content}`);
}

const functionsMap = {
  draftEmail,
};

async function main() {
  const response = await fi.chat({
    messages: [
      { role: 'system', content: 'You are a helpful assistant' },
      {
        role: 'user',
        content: 'Can you draft an email about my football game to my friend Tom?',
      },
    ],
    forceRemote: true,
    tools: [
      {
        type: 'function',
        function: {
          name: 'draftEmail',
          description: 'Draft an email for a given receiver',
          parameters: {
            type: 'object',
            properties: {
              receiver: {
                type: 'string',
                description: 'The name of the person the email should be sent to.',
              },
              content: {
                type: 'string',
                description: 'The content of the email to send.',
              },
            },
            required: ['receiver', 'content'],
          },
        },
      },
    ],
  });

  if (!response.ok) {
    console.error(`${response.failure.code}: ${response.failure.description}`);
  } else {
    if (response.message.toolCalls) {
      const tool = response.message.toolCalls.pop();
      if (tool) {
        functionsMap[tool.function.name](tool.function.arguments);
      }
    }
  }
}

await main().then().catch();
