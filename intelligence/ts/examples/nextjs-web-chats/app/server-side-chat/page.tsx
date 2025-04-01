import { chatWithHistory, history } from '@/lib/chat';
import { revalidatePath } from 'next/cache';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import ChatForm from './ChatForm';

// Ensure the page is always rendered dynamically.
export const dynamic = 'force-dynamic';

// Server Action to handle form submission.
export async function submitChat(formData: FormData) {
  'use server';
  const question = formData.get('question') as string;
  if (!question?.trim()) return;
  try {
    await chatWithHistory(question);
  } catch (error) {
    console.error(error);
  }
  revalidatePath('/server-side-chat');
}

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

      {/* Chat Form */}
      <ChatForm action={submitChat} />
    </div>
  );
}
