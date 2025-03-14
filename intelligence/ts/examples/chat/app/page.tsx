'use client';

import React, { useState, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { FlowerIntelligence, ChatResponseResult, Message, Progress } from '@flwr/flwr';

const fi: FlowerIntelligence = FlowerIntelligence.instance;

const history: Message[] = [
  { role: 'system', content: 'You are a friendly assistant that loves using emojis.' },
];

interface ChatEntry {
  role: 'user' | 'bot';
  content: string;
  // For bot messages, store the model used when it was generated
  modelUsed?: string;
}

const availableModels = [
  'meta/llama3.2-1b/instruct-fp16',
  'meta/llama3.2-3b/instruct-q4',
  'meta/llama3.1-8b/instruct-q4',
  'deepseek/r1-distill-llama-8b/q4',
];

const isProduction = process.env.NODE_ENV === 'production';

// A simple collapsible component to show/hide internal reasoning.
const Collapsible: React.FC<{ content: string }> = ({ content }) => {
  const [isOpen, setIsOpen] = useState(false);
  return (
    <div className="mb-2">
      <button onClick={() => setIsOpen(!isOpen)} className="text-sm text-blue-500 underline mb-1">
        {isOpen ? 'Hide internal reasoning' : 'Show internal reasoning'}
      </button>
      {isOpen && (
        <div className="p-2 border-l-4 border-blue-500 bg-blue-50 text-sm italic">{content}</div>
      )}
    </div>
  );
};

// Helper function using guard clauses to determine remote handoff settings.
const getRemoteHandoffSettings = (allowRemote: boolean) => {
  if (isProduction || !allowRemote) return { remoteHandoff: false, apiKey: '' };
  return { remoteHandoff: true, apiKey: process.env.NEXT_PUBLIC_API_KEY || '' };
};

export default function ClientSideChatPage() {
  const [chatLog, setChatLog] = useState<ChatEntry[]>(
    history
      .filter((msg) => msg.role !== 'system')
      .map((msg) => ({ role: msg.role as 'user' | 'bot', content: msg.content }))
  );
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  // Initialize model state as null; load the stored model on mount.
  const [model, setModel] = useState<string | null>(null);
  useEffect(() => {
    const stored = localStorage.getItem('selectedModel');
    setModel(stored || availableModels[0]);
  }, []);

  // Update localStorage whenever the model changes (and is not null).
  useEffect(() => {
    if (model !== null) {
      localStorage.setItem('selectedModel', model);
    }
  }, [model]);

  const [allowRemote, setAllowRemote] = useState(false);
  // In production, force allowRemote to be false.
  useEffect(() => {
    if (isProduction) {
      setAllowRemote(false);
    }
  }, []);

  // Keep track of which models have been loaded.
  const [loadedModels, setLoadedModels] = useState<string[]>([]);
  // Store the current model loading description (null when not loading).
  const [modelLoadingDescription, setModelLoadingDescription] = useState<string | null>(null);

  // Helper to render bot responses based on the model used.
  const renderAssistantContent = (entry: ChatEntry) => {
    const content = entry.content;
    if (!content) return 'Thinking...';
    // Use stored model if available, otherwise fall back to current model.
    const usedModel = entry.modelUsed || model || availableModels[0];
    if (usedModel === 'deepseek/r1-distill-llama-8b/q4') {
      // Extract <think> internal reasoning and main content.
      const regex = /<think>([\s\S]*?)<\/think>([\s\S]*)/;
      const match = content.match(regex);
      if (match) {
        const internalReasoning = match[1].trim();
        const mainContent = match[2].trim();
        return (
          <>
            <Collapsible content={internalReasoning} />
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{mainContent}</ReactMarkdown>
          </>
        );
      }
    }
    return <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>;
  };

  const sendQuestion = async () => {
    if (!input.trim()) return;
    setLoading(true);

    // Use guard clause helper to set remote handoff settings.
    const { remoteHandoff, apiKey } = getRemoteHandoffSettings(allowRemote);
    fi.remoteHandoff = remoteHandoff;
    fi.apiKey = apiKey;

    // Check if the selected model is loaded; if not, fetch it.
    const currentModel = model || availableModels[0];
    if (!loadedModels.includes(currentModel)) {
      setModelLoadingDescription('Start to fetch params');
      await fi.fetchModel(currentModel, (progress: Progress) => {
        setModelLoadingDescription(progress.description ?? null);
      });
      setLoadedModels((prev) => [...prev, currentModel]);
      setModelLoadingDescription(null);
    }

    // Append the user's message.
    setChatLog((prev) => [...prev, { role: 'user', content: input }]);
    history.push({ role: 'user', content: input });
    setInput('');

    // Append a placeholder bot message with the current model stored.
    setChatLog((prev) => [...prev, { role: 'bot', content: '', modelUsed: currentModel }]);

    try {
      const response: ChatResponseResult = await fi.chat({
        messages: history,
        model: currentModel,
        stream: true,
        onStreamEvent: (event) => {
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
        return; // Guard clause: exit early on error.
      }
      history.push(response.message);
    } catch {
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
    <div className="flex flex-col h-[calc(80vh)] border rounded shadow bg-white mt-16 mx-auto mb-8 max-w-7xl">
      {/* Display model loading description if model is being loaded */}
      {modelLoadingDescription !== null && (
        <div className="p-2 text-center bg-yellow-100 text-yellow-800">
          {modelLoadingDescription}
        </div>
      )}

      {/* Chat Messages */}
      <div className="flex-grow p-4 overflow-auto">
        {chatLog.map((entry, index) => (
          <div
            key={index}
            className={`mb-4 flex ${entry.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`p-3 rounded-lg ${
                entry.role === 'user'
                  ? 'max-w-[75%] bg-gray-300 text-gray-900 rounded-tr-none'
                  : 'text-gray-800 rounded-tl-none'
              }`}
            >
              {entry.role === 'bot' ? (
                renderAssistantContent(entry)
              ) : (
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{entry.content}</ReactMarkdown>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Input Area with Model Select, Remote Handoff Toggle, and Text Input */}
      <div className="border-t p-4 bg-gray-50 flex flex-col md:flex-row items-center md:space-x-4 space-y-4 md:space-y-0">
        {/* Model Select */}
        <div className="w-full md:w-1/4 p-2 border border-gray-300 rounded bg-white text-gray-800">
          <select
            className="w-full"
            value={model || availableModels[0]}
            onChange={(e) => setModel(e.target.value)}
          >
            {availableModels.map((modelName) => (
              <option key={modelName} value={modelName}>
                {modelName}
              </option>
            ))}
          </select>
        </div>
        {/* Remote Handoff Toggle (only in non-production) */}
        {!isProduction && (
          <div className="w-full md:w-1/4">
            <label className="flex items-center space-x-2 text-lg">
              <input
                type="checkbox"
                checked={allowRemote}
                onChange={(e) => setAllowRemote(e.target.checked)}
              />
              <span className="text-gray-800">Allow remote handoff</span>
            </label>
          </div>
        )}
        {/* Text Input and Send Button */}
        <div className="w-full md:w-1/2 flex flex-col md:flex-row space-y-2 md:space-y-0 md:space-x-2">
          <input
            type="text"
            placeholder="Type your question..."
            className="w-full p-2 border border-gray-300 rounded focus:outline-none bg-white text-gray-800"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
          />
          <button
            onClick={sendQuestion}
            disabled={loading || !input.trim() || modelLoadingDescription !== null}
            className="w-full md:w-auto bg-blue-500 text-white px-4 py-2 rounded disabled:bg-gray-400"
          >
            {loading ? 'Sending...' : 'Send'}
          </button>
        </div>
      </div>
    </div>
  );
}
