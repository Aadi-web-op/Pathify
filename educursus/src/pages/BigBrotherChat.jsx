// src/pages/BigBrotherChat.jsx
import React, { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { callGeminiAPI } from '../api/gemini';
import { GeminiLoader } from '../components/GeminiLoader';
import { AITypingEffect } from '../components/AITypingEffect';
import { playFuturisticSound } from '../hooks/useSound';

export default function BigBrotherChat() {
    const [messages, setMessages] = useState([{ text: "Hey there! I'm your Big Brother AI. Ask me anything about careers, skills, or tech. What's on your mind? 🤔", sender: 'ai' }]);
    const [input, setInput] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(scrollToBottom, [messages]);

    const handleSend = async () => {
        if (!input.trim()) return;
        playFuturisticSound();
        const userMessage = { text: input, sender: 'user' };
        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setIsLoading(true);

        const prompt = `You are 'Big Brother,' a friendly and knowledgeable older sibling AI. You give career advice to students in an encouraging, cool, and slightly informal way. Use emojis where appropriate. Keep your answers concise and helpful. The user's question is: "${input}"`;
        try {
            const aiResponse = await callGeminiAPI(prompt);
            setMessages(prev => [...prev, { text: aiResponse, sender: 'ai' }]);
        } catch (error) {
            setMessages(prev => [...prev, { text: "Oops, my circuits are a bit fried right now. Try asking me again in a moment.", sender: 'ai' }]);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="flex flex-col h-full">
            <h1 className="text-4xl font-bold text-black dark:text-text-primary mb-2">Talk with Big Brother 🤖</h1>
            <p className="text-text-secondary mb-4">Your personal AI mentor for career advice, skill questions, and tech talk.</p>
            <div className="holographic-card flex-1 flex flex-col p-4 rounded-2xl">
                <div className="flex-1 overflow-y-auto pr-2 space-y-4">
                    {messages.map((msg, index) => (
                        <motion.div key={index} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className={`flex gap-3 ${msg.sender === 'user' ? 'flex-row-reverse' : ''}`}>
                            <div className={`w-8 h-8 rounded-full flex items-center justify-center ${msg.sender === 'ai' ? 'bg-sky-500 text-white' : 'bg-slate-200 dark:bg-slate-600 text-black dark:text-text-primary'}`}>{msg.sender === 'ai' ? '🤖' : '🧑'}</div>
                            <div className={`max-w-md p-3 rounded-xl ${msg.sender === 'ai' ? 'chat-bubble-ai' : 'bg-slate-200 dark:bg-slate-700'}`}>
                                {msg.sender === 'ai' && index === messages.length - 1 && isLoading === false ? <AITypingEffect text={msg.text} /> : <p className="text-black dark:text-text-primary">{msg.text}</p>}
                            </div>
                        </motion.div>
                    ))}
                    <div ref={messagesEndRef} />
                </div>
                <div className="mt-4 flex gap-4">
                    <input type="text" value={input} onChange={e => setInput(e.target.value)} onKeyPress={e => e.key === 'Enter' && handleSend()} placeholder="Ask your 'Big Bro' anything..." className="flex-1 bg-input-bg border border-input-border rounded-lg p-3 focus:outline-none focus:ring-2 focus:ring-sky-500 text-black dark:text-text-primary" />
                    <motion.button whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }} onClick={handleSend} disabled={isLoading} className="primary-btn px-6 py-3 disabled:opacity-50">
                        {isLoading ? <GeminiLoader /> : 'Send'}
                    </motion.button>
                </div>
            </div>
        </div>
    );
};