// src/pages/Dashboard.jsx
import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { callGeminiAPI } from '../api/gemini';
import { GeminiLoader } from '../components/GeminiLoader';
import { AITypingEffect } from '../components/AITypingEffect';
import { playFuturisticSound } from '../hooks/useSound';

export default function Dashboard({ profile, progress }) {
    const [analysis, setAnalysis] = useState('');
    const [isLoadingAnalysis, setIsLoadingAnalysis] = useState(false);
    const [level, setLevel] = useState('Explorer');

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

    return (
        <div>
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