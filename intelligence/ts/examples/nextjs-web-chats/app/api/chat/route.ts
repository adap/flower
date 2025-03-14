import { chatWithHistory } from '@/lib/chat';

export async function POST(req: Request) {
    try {
        const { question } = await req.json();
        const message = await chatWithHistory(question);
        return new Response(JSON.stringify({ message }), { status: 200 });
    } catch (error) {
        return new Response(
            JSON.stringify({ error: (error as Error).message }),
            { status: 500 }
        );
    }
}
