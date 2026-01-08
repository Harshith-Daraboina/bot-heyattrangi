"use client";

import React, { useState, useEffect, useRef } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, RotateCcw, FileText, Sparkles } from 'lucide-react';
import { sendChatMessage, generateSummary, resetSession } from '@/lib/api';
import { cn } from '@/lib/utils';
import Image from 'next/image';

interface Message {
    id: string;
    role: 'user' | 'assistant';
    content: string;
    expression?: string;
}

const EXPRESSION_MAP: Record<string, string> = {
    EMPATHETIC: "/bot_expressions/EMPATHETIC.jpg",
    STRESSED: "/bot_expressions/STRESSED.jpg",
    TIRED: "/bot_expressions/TIRED.jpg",
    REFLECTIVE: "/bot_expressions/REFLECTIVE.jpg",
    SAFETY: "/bot_expressions/SAFETY.jpg",
    NEUTRAL: "/bot_expressions/NEUTRAL.jpg",
};

const getExpressionImage = (expression: string): string => {
    return EXPRESSION_MAP[expression] || EXPRESSION_MAP["NEUTRAL"];
};

export default function ChatInterface() {
    const [sessionId, setSessionId] = useState<string>('');
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [currentExpression, setCurrentExpression] = useState('NEUTRAL');
    const [summary, setSummary] = useState<string | null>(null);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        // Initialize Session ID
        const sid = localStorage.getItem('attrangi_session_id') || uuidv4();
        localStorage.setItem('attrangi_session_id', sid);
        setSessionId(sid);

        // Initial greeting could be handled differently, but let's leave it blank or add one
        setMessages([{
            id: 'init',
            role: 'assistant',
            content: "Hi, I'm Attrangi. I'm here to listen. How are you feeling today?",
            expression: "NEUTRAL"
        }]);
    }, []);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages]);

    const handleSend = async () => {
        if (!input.trim() || isLoading) return;

        const userMsg: Message = { id: uuidv4(), role: 'user', content: input };
        setMessages(prev => [...prev, userMsg]);
        setInput('');
        setIsLoading(true);

        try {
            // Optimistic update - keep current expression while thinking
            const response = await sendChatMessage(sessionId, userMsg.content);

            const botMsg: Message = {
                id: uuidv4(),
                role: 'assistant',
                content: response.reply,
                expression: response.expression
            };

            setMessages(prev => [...prev, botMsg]);
            setCurrentExpression(response.expression || 'NEUTRAL');

        } catch (error) {
            console.error(error);
            // Handle error gracefully
        } finally {
            setIsLoading(false);
        }
    };

    const handleSummary = async () => {
        if (isLoading) return;
        setIsLoading(true);
        try {
            const res = await generateSummary(sessionId);
            setSummary(res.summary);
        } catch (e) {
            console.error(e);
        } finally {
            setIsLoading(false);
        }
    };

    const handleReset = async () => {
        if (!sessionId) return;
        try {
            await resetSession(sessionId);
            setMessages([{
                id: uuidv4(),
                role: 'assistant',
                content: "Session cleared. I'm ready to start fresh.",
                expression: "NEUTRAL"
            }]);
            setSummary(null);
            setCurrentExpression('NEUTRAL');
        } catch (e) {
            console.error(e);
        }
    };

    return (
        <div className="flex h-screen font-sans">
            {/* Sidebar - Bot Avatar */}
            <aside className="w-[450px] bg-slate-950 border-r border-slate-800 flex flex-col justify-between p-6">
                <div className="flex flex-col items-center gap-4">
                    <motion.div
                        key={currentExpression}
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ duration: 0.3 }}
                        className="w-full rounded-2xl shadow-lg border-4 border-slate-800 overflow-hidden"
                    >
                        <Image
                            src={getExpressionImage(currentExpression)}
                            alt={currentExpression}
                            width={400}
                            height={400}
                            className="w-full h-auto object-cover"
                            priority
                        />
                    </motion.div>
                    <div className="w-full flex justify-center mt-2">
                        <span className="bg-blue-900 text-blue-100 px-3 py-1 text-sm rounded-full uppercase tracking-wider font-medium">
                            {currentExpression} MODE
                        </span>
                    </div>
                </div>
                
                <div className="flex flex-col gap-2">
                    <button
                        onClick={handleSummary}
                        disabled={isLoading}
                        className="w-full h-14 text-lg bg-slate-800 border border-slate-700 hover:bg-slate-700 text-slate-200 rounded-xl transition-colors flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        <FileText size={18} />
                        Generate Summary
                    </button>
                    <button
                        onClick={handleReset}
                        disabled={isLoading}
                        className="w-full h-14 text-lg bg-slate-800 border border-slate-700 hover:bg-slate-700 text-slate-200 rounded-xl transition-colors flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        <RotateCcw size={18} />
                        Reset
                    </button>
                </div>
            </aside>

            {/* Main Chat Area */}
            <div className="flex-1 flex flex-col">
                {/* Header */}
                <header className="glass-card p-4 border-b border-slate-800">
                    <h1 className="text-3xl font-bold text-slate-100">Hey Attrangi</h1>
                    <p className="text-slate-400 mt-1">A safe place to talk.</p>
                </header>

                {/* Chat Messages */}
                <main className="flex-1 overflow-y-auto p-4 space-y-4 scrollbar-thin scrollbar-thumb-gray-700 scrollbar-track-transparent">
                <AnimatePresence initial={false}>
                    {messages.map((msg) => (
                        <motion.div
                            key={msg.id}
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0 }}
                            className={cn(
                                "flex w-full mb-4",
                                msg.role === 'user' ? "justify-end" : "justify-start"
                            )}
                        >
                            <div className={cn(
                                "max-w-[80%] md:max-w-[70%] p-4 rounded-2xl shadow-sm",
                                msg.role === 'user'
                                    ? "bg-blue-900 text-white rounded-tr-sm border border-blue-600"
                                    : "bg-slate-800 text-slate-100 rounded-tl-sm border border-slate-700"
                            )}>
                                <p className="leading-relaxed whitespace-pre-wrap text-base">{msg.content}</p>
                            </div>
                        </motion.div>
                    ))}
                    {isLoading && messages[messages.length - 1]?.role === 'user' && (
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            className="flex justify-start w-full"
                        >
                            <div className="bg-slate-800 text-slate-100 p-4 rounded-2xl rounded-tl-sm border border-slate-700 flex items-center">
                                <div className="flex gap-1">
                                    <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce [animation-delay:-0.3s]"></span>
                                    <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce [animation-delay:-0.15s]"></span>
                                    <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce"></span>
                                </div>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>
                <div ref={messagesEndRef} />
            </main>

            {/* Summary Modal / Section */}
            <AnimatePresence>
                {summary && (
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: 20 }}
                        className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm"
                        onClick={() => setSummary(null)}
                    >
                        <div
                            className="bg-card w-full max-w-2xl max-h-[80vh] overflow-y-auto rounded-3xl p-6 shadow-2xl border border-white/10"
                            onClick={(e) => e.stopPropagation()}
                        >
                            <div className="flex justify-between items-center mb-4">
                                <h2 className="text-2xl font-bold flex items-center gap-2">
                                    <Sparkles className="text-yellow-400" />
                                    Clinical Summary
                                </h2>
                                <button onClick={() => setSummary(null)} className="text-muted-foreground hover:text-white">Close</button>
                            </div>
                            <div className="prose prose-invert max-w-none">
                                <p className="whitespace-pre-wrap text-gray-300 leading-relaxed">{summary}</p>
                            </div>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>

                {/* Input Area */}
                <footer className="bg-slate-900 border-t border-slate-800 p-4">
                    <form
                        onSubmit={(e) => { e.preventDefault(); handleSend(); }}
                        className="flex items-center gap-2 max-w-5xl mx-auto"
                    >
                        <textarea
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            placeholder="What's been on your mind?"
                            rows={1}
                            className="flex-1 bg-slate-800 border border-slate-700 text-slate-100 placeholder-slate-400 px-4 py-3 rounded-xl resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                            disabled={isLoading}
                            onKeyDown={(e) => {
                                if (e.key === 'Enter' && !e.shiftKey) {
                                    e.preventDefault();
                                    handleSend();
                                }
                            }}
                        />
                        <button
                            type="submit"
                            disabled={isLoading || !input.trim()}
                            className="p-3 bg-blue-600 hover:bg-blue-500 text-white rounded-lg transition-transform hover:scale-105 disabled:opacity-50 disabled:hover:scale-100"
                        >
                            <Send size={20} />
                        </button>
                    </form>
                </footer>
            </div>
        </div>
    );
}
