import { history } from '@/lib/chat';
import { submitChat } from '@/lib/serverActions';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import ChatForm from './ChatForm';

// Ensure the page is always rendered dynamically.
export const dynamic = 'force-dynamic';

export default async function ServerSideChatPage() {
  return (
    <div className="flex flex-col h-[calc(100vh-60px)] border rounded shadow bg-white dark:bg-gray-800">
      {/* Chat Messages */}
      <div className="flex-grow p-4 overflow-auto">
        {history
          .filter((entry) => entry.role !== 'system')
          .map((entry, i) => (
            <div
              key={i}
              className={`mb-4 flex ${entry.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-[75%] p-3 rounded-lg ${
                  entry.role === 'user'
                    ? 'bg-blue-500 text-white rounded-tr-none'
                    : 'bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-100 rounded-tl-none'
                }`}
              >
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{entry.content}</ReactMarkdown>
              </div>
            </div>
          ))}
      </div>

      {/* Explanation for no visual feedback */}
      <div className="p-4 text-sm text-gray-500 dark:text-gray-400">
        Note: There is no visual feedback while the server processes your question. Please wait for
        the response to appear.
      </div>

      {/* Chat Form */}
      <ChatForm action={submitChat} />
    </div>
  );
}
