import { FlowerIntelligence } from '@flwr/flwr';

const fi = FlowerIntelligence.instance;

async function sendChatCompletion() {
  fi.remoteHandoff = true;
  fi.apiKey = process.env.FI_API_KEY ?? '';

  const response = await fi.chat({
    model: 'mistralai/mistral-small-3.1-24b',
    forceRemote: true,
    onStreamEvent: (event) => {
      console.log(event);
    },
    stream: true,
    messages: [
      {
        role: 'system',
        content: `You are a helpful executive assistant.
The current date and time is 2025-06-30T21:09:42.631Z.
❖ You MAY have access to tools that give you access to real-time or external data.
❖ Whenever the user asks for information that depends on real-time or external data, you MUST attempt to call an appropriate tool.
❖ If the user asks for information that you do not have access to, be honest and say so.
❖ If you have access to a tool that can provide the information, but you don't have the enough information to use it, ask the user for the missing information.
❖ Under no circumstances should you fabricate the missing data.
Before sending your final reply, silently ask yourself:
"Did I *successfully* call a tool to obtain every live fact I'm about to state?"
If the answer is "no", refuse as instructed above.
Is the message that I'm about to send to the user actually useful for a human or do I need to call more tools to make it useful?
Respond in Markdown that is pleasant, concise, and helpful. Use subheaders, bullet points, and bold / italics to help structure the response. Use emojis where appropriate.
Never invent information unless the user explicitly requests creative fiction.`,
      },
      {
        role: 'user',
        content: 'What are the top news stories in the United States today? (Use the search tool.)',
      },
    ],
    tools: [
      {
        type: 'function',
        function: {
          name: 'addTasks',
          description: "Add a task to the user's task (to do) list.",
          parameters: {
            type: 'object',
            properties: {
              tasks: {
                type: 'array',
                description: "The tasks to add to the user's task (to do) list.",
              },
            },
            required: ['tasks'],
          },
        },
      },
      {
        type: 'function',
        function: {
          name: 'deleteTasks',
          description: "Delete a task from the user's task (to do) list.",
          parameters: {
            type: 'object',
            properties: {
              taskIds: {
                type: 'array',
                description: "The IDs of the tasks to delete from the user's task (to do) list.",
              },
            },
            required: ['taskIds'],
          },
        },
      },
      {
        type: 'function',
        function: {
          name: 'getTasks',
          description: "Get the user's task (to do) list.",
          parameters: {
            type: 'object',
            properties: {},
            required: [],
          },
        },
      },
      {
        type: 'function',
        function: {
          name: 'search',
          description: 'Search DuckDuckGo and return formatted results',
          parameters: {
            type: 'object',
            properties: {
              query: {
                type: 'string',
                description: 'The search query string',
              },
              max_results: {
                type: 'number',
                description: 'Maximum number of results to return (default: 10)',
              },
            },
            required: ['query'],
          },
        },
      },
      {
        type: 'function',
        function: {
          name: 'fetch_content',
          description: 'Fetch and parse content from a webpage URL',
          parameters: {
            type: 'object',
            properties: {
              url: {
                type: 'string',
                description: 'The webpage URL to fetch content from',
              },
            },
            required: ['url'],
          },
        },
      },
      {
        type: 'function',
        function: {
          name: 'search_locations',
          description: 'Search for locations by name to get coordinates for weather data',
          parameters: {
            type: 'object',
            properties: {
              query: {
                type: 'string',
                description: 'The location name to search for',
              },
            },
            required: ['query'],
          },
        },
      },
      {
        type: 'function',
        function: {
          name: 'get_current_weather',
          description:
            'Get the current weather for specified coordinates. Call search_locations first to find coordinates if you only have a location name.',
          parameters: {
            type: 'object',
            properties: {
              lat: { type: 'number', description: 'Latitude coordinate' },
              lng: { type: 'number', description: 'Longitude coordinate' },
              days: {
                type: 'number',
                description:
                  'Number of days to forecast (1-16, default: 3) - only used for forecast',
              },
            },
            required: ['lat', 'lng'],
          },
        },
      },
      {
        type: 'function',
        function: {
          name: 'get_weather_forecast',
          description:
            'Get the weather forecast for specified coordinates. Call search_locations first to find coordinates if you only have a location name.',
          parameters: {
            type: 'object',
            properties: {
              lat: { type: 'number', description: 'Latitude coordinate' },
              lng: { type: 'number', description: 'Longitude coordinate' },
              days: {
                type: 'number',
                description:
                  'Number of days to forecast (1-16, default: 3) - only used for forecast',
              },
            },
            required: ['lat', 'lng'],
          },
        },
      },
    ],
    toolChoice: 'auto',
  });

  if (!response.ok) {
    console.error(response.failure.description);
    process.exit(1);
  } else {
    console.log(response.message.content);
    console.log(response.message.toolCalls);
    console.log(response.usage);
  }
}

sendChatCompletion().catch(console.error);
