import React from 'react';
import { playFuturisticSound } from '../hooks/useSound';

export default function LoginPage({ onLogin }) {
    return (
        <div className="w-full h-screen flex flex-col items-center justify-center p-4">
            <div className="w-full max-w-md text-center">
                <h1 className="text-5xl font-extrabold mb-2 text-black dark:text-text-primary">Welcome Back</h1>
                <p className="text-text-secondary mb-8">Access your personalized career console.</p>
                <div className="holographic-card p-8 rounded-2xl">
                    <h2 className="text-2xl font-semibold mb-6 text-text-secondary">Secure Login</h2>
                    <form onSubmit={(e) => { e.preventDefault(); onLogin(); playFuturisticSound('success'); }}>
                        <input type="email" placeholder="Email (demo)" required className="w-full mb-4 bg-input-bg border border-input-border rounded-lg p-3 focus:outline-none focus:ring-2 focus:ring-sky-500 text-black dark:text-text-primary" />
                        <input type="password" placeholder="Password (demo)" required className="w-full mb-4 bg-input-bg border border-input-border rounded-lg p-3 focus:outline-none focus:ring-2 focus:ring-sky-500 text-black dark:text-text-primary" />
                        <button type="submit" className="primary-btn w-full px-8 py-3">Access Dashboard</button>
                    </form>
                </div>
            </div>
        </div>
    );
}
