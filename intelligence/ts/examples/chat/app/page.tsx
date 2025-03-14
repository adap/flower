'use client';

import React, { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { FlowerIntelligence, ChatResponseResult, Message } from '@flwr/flwr';

const fi: FlowerIntelligence = FlowerIntelligence.instance;

const history: Message[] = [
  { role: "system", content: "You are a friendly assistant that loves using emojis." }
];

interface ChatEntry {
  role: 'user' | 'bot';
  content: string;
}

// List of available models
const availableModels = [
  "meta/llama3.2-1b/instruct-fp16",
  "meta/llama3.2-3b/instruct-q4",
  "meta/llama3.1-8b/instruct-q4",
  "deepseek/r1-distill-llama-8b/q4"
];

export default function ClientSideChatPage() {
  // Initialize local state using the current global history (excluding the system message)
  const [chatLog, setChatLog] = useState<ChatEntry[]>(
    history.filter((msg) => msg.role !== 'system').map((msg) => ({
      role: msg.role as 'user' | 'bot',
      content: msg.content,
    }))
  );
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [model, setModel] = useState(availableModels[0]); // Default to first model

  const sendQuestion = async () => {
    if (!input.trim()) return;
    setLoading(true);

    // Append the user's question to both local and global history
    setChatLog((prev) => [...prev, { role: 'user', content: input }]);
    history.push({ role: 'user', content: input });
    setInput('');

    // Append an empty bot message as a placeholder for streamed content.
    setChatLog((prev) => [...prev, { role: 'bot', content: '' }]);

    try {
      const response: ChatResponseResult = await fi.chat({
        messages: history,
        model: model,
        stream: true,
        onStreamEvent: (event) => {
          // Append each chunk to the last bot message.
          setChatLog((prev) => {
            const updated = [...prev];
            const lastIndex = updated.length - 1;
            updated[lastIndex] = {
              role: 'bot',
              content: updated[lastIndex].content + event.chunk,
            };
            return updated;
          });
        },
      });
      if (!response.ok) {
        // If the response fails, update the last bot message with an error.
        setChatLog((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = {
            role: 'bot',
            content: 'Failed to get a valid response.',
          };
          return updated;
        });
      } else {
        // If successful, append the final bot message to the global history.
        history.push(response.message);
      }
    } catch (error) {
      setChatLog(
        history
          .filter((msg) => msg.role !== 'system')
          .map((msg) => ({
            role: msg.role as 'user' | 'bot',
            content: msg.content,
          }))
      );
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      sendQuestion();
    }
  };

  return (
    <div className="flex flex-col h-[calc(80vh)] border rounded shadow bg-white m-20">
      {/* Chat Messages */}
      <div className="flex-grow p-4 overflow-auto">
        {chatLog.map((entry, index) => (
          <div
            key={index}
            className={`mb-4 flex ${entry.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`p-3 rounded-lg ${entry.role === 'user'
                  ? 'max-w-[75%] bg-gray-300 text-gray-900 rounded-tr-none'
                  : 'text-gray-800 rounded-tl-none'
                }`}
            >
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {entry.role === 'bot' && entry.content === '' ? "Thinking..." : entry.content}
              </ReactMarkdown>
            </div>
          </div>
        ))}
      </div>

      {/* Input Area with Model Select */}
      <div className="border-t p-4 bg-gray-50 flex items-center">
        {/* Model select on the left */}
        <select
          value={model}
          onChange={(e) => setModel(e.target.value)}
          className="mr-4 p-2 border border-gray-300 rounded bg-white text-gray-800"
        >
          {availableModels.map((modelName) => (
            <option key={modelName} value={modelName}>
              {modelName}
            </option>
          ))}
        </select>

        {/* Input field and Send button */}
        <div className="flex flex-grow space-x-2">
          <input
            type="text"
            placeholder="Type your question..."
            className="flex-grow p-2 border border-gray-300 rounded focus:outline-none bg-white text-gray-800"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
          />
          <button
            onClick={sendQuestion}
            disabled={loading || !input.trim()}
            className="bg-blue-500 text-white px-4 py-2 rounded disabled:bg-gray-400"
          >
            {loading ? 'Sending...' : 'Send'}
          </button>
        </div>
      </div>
    </div>
  );
}
