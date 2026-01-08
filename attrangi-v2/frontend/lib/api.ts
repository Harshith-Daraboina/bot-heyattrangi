export const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface ChatResponse {
    reply: string;
    expression: string;
}

interface SummaryResponse {
    status: string;
    summary: string;
}

export const sendChatMessage = async (sessionId: string, message: string): Promise<ChatResponse> => {
    const res = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            session_id: sessionId,
            message: message,
        }),
    });
    if (!res.ok) throw new Error('Failed to send message');
    return res.json();
};

export const generateSummary = async (sessionId: string): Promise<SummaryResponse> => {
    const res = await fetch(`${API_URL}/summary`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            session_id: sessionId,
        }),
    });
    if (!res.ok) throw new Error('Failed to generate summary');
    return res.json();
};

export const resetSession = async (sessionId: string) => {
    const res = await fetch(`${API_URL}/reset`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            session_id: sessionId,
        }),
    });
    if (!res.ok) throw new Error('Failed to reset session');
    return res.json();
};
