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
  // Store the model used to generate this message (for bot messages)
  modelUsed?: string;
}

const availableModels = [
  "meta/llama3.2-1b/instruct-fp16",
  "meta/llama3.2-3b/instruct-q4",
  "meta/llama3.1-8b/instruct-q4",
  "deepseek/r1-distill-llama-8b/q4"
];

// A simple collapsible component to show/hide internal reasoning.
const Collapsible: React.FC<{ content: string }> = ({ content }) => {
  const [isOpen, setIsOpen] = useState(false);
  return (
    <div className="mb-2">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="text-sm text-blue-500 underline mb-1"
      >
        {isOpen ? 'Hide internal reasoning' : 'Show internal reasoning'}
      </button>
      {isOpen && (
        <div className="p-2 border-l-4 border-blue-500 bg-blue-50 text-sm italic">
          {content}
        </div>
      )}
    </div>
  );
};

export default function ClientSideChatPage() {
  const [chatLog, setChatLog] = useState<ChatEntry[]>(
    history
      .filter((msg) => msg.role !== 'system')
      .map((msg) => ({ role: msg.role as 'user' | 'bot', content: msg.content }))
  );
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [model, setModel] = useState(availableModels[0]);
  const [allowRemote, setAllowRemote] = useState(false);

  // Render the assistant's response based on the model used for that message.
  const renderAssistantContent = (entry: ChatEntry) => {
    const content = entry.content;
    if (!content) return "Thinking...";
    // Use the model used when the message was generated
    const usedModel = entry.modelUsed || model;
    if (usedModel === "deepseek/r1-distill-llama-8b/q4") {
      // Use regex to extract content inside <think> tags and the main content.
      const regex = /<think>([\s\S]*?)<\/think>([\s\S]*)/;
      const match = content.match(regex);
      if (match) {
        const internalReasoning = match[1].trim();
        const mainContent = match[2].trim();
        return (
          <>
            <Collapsible content={internalReasoning} />
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {mainContent}
            </ReactMarkdown>
          </>
        );
      }
    }
    return <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>;
  };

  const sendQuestion = async () => {
    if (!input.trim()) return;
    setLoading(true);

    // Set remote handoff based on the toggle.
    if (allowRemote) {
      fi.remoteHandoff = true;
      fi.apiKey = process.env.NEXT_PUBLIC_API_KEY || "";
    } else {
      fi.remoteHandoff = false;
      fi.apiKey = "";
    }

    // Append user's message.
    setChatLog((prev) => [...prev, { role: 'user', content: input }]);
    history.push({ role: 'user', content: input });
    setInput('');

    // Append placeholder bot message with the current model stored.
    setChatLog((prev) => [
      ...prev,
      { role: 'bot', content: '', modelUsed: model }
    ]);

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
              ...updated[lastIndex],
              content: updated[lastIndex].content + event.chunk,
            };
            return updated;
          });
        },
      });
      if (!response.ok) {
        setChatLog((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = {
            ...updated[updated.length - 1],
            content: 'Failed to get a valid response.',
          };
          return updated;
        });
      } else {
        history.push(response.message);
      }
    } catch (error) {
      setChatLog(
        history
          .filter((msg) => msg.role !== 'system')
          .map((msg) => ({ role: msg.role as 'user' | 'bot', content: msg.content }))
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
              {entry.role === 'bot'
                ? renderAssistantContent(entry)
                : <ReactMarkdown remarkPlugins={[remarkGfm]}>{entry.content}</ReactMarkdown>}
            </div>
          </div>
        ))}
      </div>

      {/* Input Area with Model Select and Remote Handoff Toggle */}
      <div className="border-t p-4 bg-gray-50 flex items-center">
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
        <label className="mr-4 flex items-center space-x-2">
          <input
            type="checkbox"
            checked={allowRemote}
            onChange={(e) => setAllowRemote(e.target.checked)}
          />
          <span className="text-gray-800">Allow remote handoff</span>
        </label>
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
