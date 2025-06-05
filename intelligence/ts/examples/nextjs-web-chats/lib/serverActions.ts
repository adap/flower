import { chatWithHistory } from '@/lib/chat';
import { revalidatePath } from 'next/cache';

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
