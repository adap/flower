'use client';

import { useTransition, useRef } from 'react';

export default function ChatForm({ action }: { action: (formData: FormData) => Promise<void> }) {
  const [isPending, startTransition] = useTransition();
  const inputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const formData = new FormData(e.currentTarget);
    // Clear the input field after submission.
    if (inputRef.current) {
      inputRef.current.value = '';
    }
    startTransition(() => {
      action(formData);
    });
  };

  return (
    <form
      onSubmit={handleSubmit}
      className="border-t p-4 bg-gray-50 flex space-x-2 pl-20 dark:bg-gray-900"
    >
      <input
        type="text"
        name="question"
        placeholder="Type your question..."
        ref={inputRef}
        className="flex-grow p-2 border border-gray-300 dark:border-gray-600 rounded focus:outline-none bg-white dark:bg-gray-800 text-gray-800 dark:text-gray-100"
        disabled={isPending}
      />
      <button
        type="submit"
        className="bg-blue-500 text-white px-4 py-2 rounded disabled:bg-gray-400"
        disabled={isPending}
      >
        {isPending ? 'Sending...' : 'Send'}
      </button>
    </form>
  );
}
