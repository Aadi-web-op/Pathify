import React from 'react';

export const GlobalStyles = () => (
    <style>{`
        :root { 
            --bg: #F0F4F8; 
            --text-primary: #000000; 
            --text-secondary: #475569; 
            --card-bg: rgba(255, 255, 255, 0.6); 
            --card-border: rgba(0, 191, 255, 0.3); 
            --card-hover-border: rgba(0, 191, 255, 0.8); 
            --card-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1); 
            --card-hover-shadow: 0 16px 40px rgba(0, 191, 255, 0.2); 
            --sidebar-bg: rgba(255, 255, 255, 0.5); 
            --sidebar-border: #E2E8F0; 
            --accent: #00BFFF; 
            --accent-dark: #1E90FF; 
            --accent-text: #0ea5e9; 
            --accent-text-dark: #0284c7; 
            --input-bg: rgba(255, 255, 255, 0.8); 
            --input-border: #CBD5E1; 
        } 
        .dark { 
            --bg: #02040A; 
            --text-primary: #E2E8F0; 
            --text-secondary: #94A3B8; 
            --card-bg: rgba(16, 23, 59, 0.6); 
            --card-border: rgba(0, 191, 255, 0.2); 
            --card-hover-border: rgba(0, 191, 255, 0.5); 
            --card-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2); 
            --card-hover-shadow: 0 16px 40px rgba(0, 191, 255, 0.15); 
            --sidebar-bg: rgba(16, 23, 59, 0.5); 
            --sidebar-border: rgba(0, 191, 255, 0.1); 
            --input-bg: rgba(30, 41, 59, 0.5); 
            --input-border: #475569; 
        } 
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'); 
        body { 
            font-family: 'Inter', sans-serif; 
            background-color: var(--bg); 
            color: var(--text-primary); 
            transition: background-color 0.3s ease, color 0.3s ease; 
            overflow-x: hidden; 
        } 
        .holographic-card { 
            background: var(--card-bg); 
            backdrop-filter: blur(16px); 
            -webkit-backdrop-filter: blur(16px); 
            border: 1px solid var(--card-border); 
            box-shadow: var(--card-shadow); 
            transition: all 0.3s ease; 
        } 
        .holographic-card:hover { 
            border-color: var(--card-hover-border); 
            transform: translateY(-8px) scale(1.02); 
            box-shadow: var(--card-hover-shadow); 
        } 
        .stepper-item::before { 
            content: ''; 
            position: absolute; 
            left: 1.25rem; 
            top: 2.5rem; 
            bottom: -1rem; 
            width: 2px; 
            background-color: var(--input-border); 
        } 
        .dark .stepper-item::before { 
            background-color: #334155; 
        } 
        .stepper-item:last-child::before { 
            display: none; 
        } 
        .checkbox-custom:checked { 
            background-image: url("data:image/svg+xml,%3csvg viewBox='0 0 16 16' fill='white' xmlns='http://www.w3.org/2000/svg'%3e%3cpath d='M12.207 4.793a1 1 0 010 1.414l-5 5a1 1 0 01-1.414 0l-2-2a1 1 0 011.414-1.414L6.5 9.086l4.293-4.293a1 1 0 011.414 0z'/%3e%3c/svg%3e"); 
        } 
        .primary-btn { 
            background: linear-gradient(45deg, var(--accent), var(--accent-dark)); 
            color: #FFFFFF; 
            font-weight: 700; 
            border-radius: 8px; 
            transition: all 0.3s ease; 
            box-shadow: 0 4px 15px rgba(0, 191, 255, 0.2); 
            animation: pulse-glow 2s infinite alternate; 
        } 
        .primary-btn:hover { 
            transform: translateY(-3px); 
            box-shadow: 0 8px 25px rgba(0, 191, 255, 0.4); 
            animation-play-state: paused; 
        } 
        @keyframes pulse-glow { 
            from { box-shadow: 0 0 10px rgba(0, 191, 255, 0.2), 0 0 20px rgba(0, 191, 255, 0.2); } 
            to { box-shadow: 0 0 20px rgba(0, 191, 255, 0.5), 0 0 30px rgba(0, 191, 255, 0.5); } 
        } 
        .gemini-loader-dot { 
            animation: gemini-loader 1.4s infinite ease-in-out both; 
        } 
        .gemini-loader-dot:nth-child(1) { 
            animation-delay: -0.32s; 
        } 
        .gemini-loader-dot:nth-child(2) { 
            animation-delay: -0.16s; 
        } 
        @keyframes gemini-loader { 
            0%, 80%, 100% { transform: scale(0); } 
            40% { transform: scale(1.0); } 
        } 
        #aurora-background { 
            position: fixed; 
            top: 0; 
            left: 0; 
            width: 100%; 
            height: 100%; 
            z-index: -1; 
            background: radial-gradient(ellipse at top, #FFFFFF, transparent), radial-gradient(ellipse at bottom, #D1E5F0, transparent); 
            background-size: 200% 200%; 
            animation: aurora-pan 30s linear infinite; 
        } 
        .dark #aurora-background { 
            background: radial-gradient(ellipse at top, #02040A, transparent), radial-gradient(ellipse at bottom, #1E90FF, transparent); 
        } 
        @keyframes aurora-pan { 
            from { background-position: 0% 0%; } 
            to { background-position: -200% 0%; } 
        } 
        .chat-bubble-ai { 
            background: linear-gradient(45deg, rgba(30, 144, 255, 0.3), rgba(0, 191, 255, 0.3)); 
            animation: pulse-glow-bubble 3s infinite alternate; 
        } 
        @keyframes pulse-glow-bubble { 
            from { box-shadow: 0 0 5px rgba(0, 191, 255, 0.1); } 
            to { box-shadow: 0 0 15px rgba(0, 191, 255, 0.4); } 
        }
    `}
    </style>
);

export default GlobalStyles