import { chatWithHistory } from '@/lib/chat';

export async function POST(req: Request) {
  try {
    const { question } = await req.json();
    const message = await chatWithHistory(question);
    return new Response(JSON.stringify({ message }), { status: 200 });
  } catch (error) {
    console.error('Error in /api/chat:', error);
    return new Response(
      JSON.stringify({ error: 'Failed to process the request. Please try again later.' }),
      { status: 500 }
    );
  }
}
