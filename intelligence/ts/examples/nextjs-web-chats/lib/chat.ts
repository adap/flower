import { ChatResponseResult, FlowerIntelligence, Message, ChatOptions } from '@flwr/flwr';

const fi: FlowerIntelligence = FlowerIntelligence.instance;

// Global chat history with an initial system message.
export const history: Message[] = [
    { role: "system", content: "You are a friendly assistant that loves using emojis." }
];

export async function chatWithHistory(question: string): Promise<string> {
    history.push({ role: "user", content: question });
    const response: ChatResponseResult = await fi.chat(question, {
        messages: history,
    } as ChatOptions);
    if (response.ok) {
        history.push(response.message);
        return response.message.content;
    } else {
        throw new Error('Failed to get a valid response.');
    }
}
