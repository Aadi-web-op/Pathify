import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';

// --- Mock Implementations (for demonstration) ---
const callGeminiAPI = async (prompt) => {
    console.log("Calling Gemini with prompt:", prompt);
    await new Promise(resolve => setTimeout(resolve, 1500)); // Simulate API delay
    return "Based on your profile, you have a strong analytical mindset suited for tech roles. Your interest in both Computer Science and team collaboration suggests a great potential in fields like software engineering or product management. Keep exploring your project ideas!";
};

const GeminiLoader = () => <div className="text-accent-text">Analyzing your profile...</div>;

const AITypingEffect = ({ text }) => {
    // A simple component to display text, a real one would have a typing effect.
    return <p className="text-center text-lg text-slate-300">{text}</p>;
};

const playFuturisticSound = (sound) => console.log(`Playing sound: ${sound || 'click'}`);

const LogOutIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"></path>
        <polyline points="16 17 21 12 16 7"></polyline>
        <line x1="21" y1="12" x2="9" y2="12"></line>
    </svg>
);
// --- End Mock Implementations ---


export default function Dashboard({ profile, progress, onLogout }) {
    const [analysis, setAnalysis] = useState('');
    const [isLoadingAnalysis, setIsLoadingAnalysis] = useState(false);
    const [level, setLevel] = useState('Explorer');

    // If profile data is not yet available, show a loading state.
    // This prevents errors from trying to access properties of `undefined`.
    if (!profile || !progress) {
        return (
            <div className="w-full h-full flex items-center justify-center p-8"><GeminiLoader /></div>
        );
    }

    useEffect(() => {
        const completedCount = Object.values(progress).filter(p => p).length;
        if (completedCount > 15) setLevel('Industry Ready');
        else if (completedCount > 8) setLevel('Skilled');
        else setLevel('Explorer');
    }, [progress]);

    const handleAnalysis = async () => {
        playFuturisticSound();
        setIsLoadingAnalysis(true);
        setAnalysis('');
        const prompt = `Analyze this student profile and provide a short, encouraging summary (2-3 sentences) of their potential strengths and direction. Profile: Age: ${profile.age}, Stream: ${profile.stream}, Interests: ${profile.interests.join(', ')}, Career Interests: ${profile.careerInterests.join(', ')}, Project Experience: ${profile.projectExperience}, Team Preference: ${profile.teamPreference}, Problem Solving Style: ${profile.problemSolvingStyle}, 5-year Goal: ${profile.longTermGoal}.`;
        try {
            const result = await callGeminiAPI(prompt);
            setAnalysis(result);
        } catch (error) {
            setAnalysis("Sorry, I couldn't generate an analysis at this moment.");
        } finally {
            setIsLoadingAnalysis(false);
        }
    };

    const handleLogout = () => {
        playFuturisticSound('logout');
        // The `onLogout` prop is a function passed down from the parent component (e.g., App.jsx).
        // When this button is clicked, we call this function to tell the parent component
        // to handle the logout logic, which should include switching the view back to the LandingPage.
        if (onLogout) {
            onLogout();
        }
    };

    return (
        <div className="relative p-4 md:p-8">
            <motion.button
                onClick={handleLogout}
                whileHover={{ scale: 1.05, rotate: -2 }}
                whileTap={{ scale: 0.95 }}
                className="absolute top-4 right-4 z-10 flex items-center gap-2 bg-slate-700/50 hover:bg-slate-600/70 text-slate-300 font-semibold py-2 px-4 rounded-lg border border-slate-600 transition-colors"
                aria-label="Logout"
            >
                <LogOutIcon />
                Logout
            </motion.button>

            <h1 className="text-4xl font-bold text-black dark:text-text-primary mb-2">Welcome back, <span className="text-accent-text">{profile.name}</span>!</h1>
            <p className="text-text-secondary mb-8">Your personalized career console is ready. Let's explore your future.</p>
            <div className="grid md:grid-cols-3 gap-8">
                <div className="holographic-card p-6 rounded-2xl md:col-span-2">
                    <h2 className="text-2xl font-bold text-accent-text-dark mb-4">Student Profile</h2>
                    <div className="grid grid-cols-2 gap-4 text-black dark:text-text-primary">
                        <p><strong>Age:</strong> {profile.age}</p>
                        <p><strong>Stream:</strong> {profile.stream}</p>
                        <p><strong>Learning Style:</strong> {profile.learningStyle || 'N/A'}</p>
                        <p><strong>Work Env:</strong> {profile.workEnv || 'N/A'}</p>
                    </div>
                </div>
                <div className="holographic-card p-6 rounded-2xl flex flex-col justify-center items-center">
                    <h2 className="text-xl font-bold text-accent-text-dark mb-2">Your Level</h2>
                    <p className="text-3xl font-extrabold text-accent-text">{level}</p>
                    <p className="text-sm text-text-secondary mt-1">{Object.values(progress).filter(p => p).length} skills completed</p>
                </div>
                <div className="holographic-card p-8 rounded-2xl flex flex-col justify-center items-center min-h-[150px] md:col-span-3">
                    <h2 className="text-2xl font-bold text-accent-text-dark mb-4">AI-Powered Profile Analysis ✨</h2>
                    {isLoadingAnalysis ? <GeminiLoader /> : analysis ? ( <AITypingEffect text={analysis} /> ) : ( <button onClick={handleAnalysis} className="primary-btn px-6 py-3">Generate My Analysis</button> )}
                </div>
            </div>
        </div>
    );
}

