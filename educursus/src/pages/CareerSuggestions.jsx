// src/pages/CareerSuggestions.jsx
import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { callGeminiAPI } from '../api/gemini';
import { mockCareers } from '../api/mockData';
import { GeminiLoader } from '../components/GeminiLoader';
import { SparklesIcon } from '../assets/icons';
import { playFuturisticSound } from '../hooks/useSound';

export default function CareerSuggestions({ profile, navigate, setSelectedCareer, setGeneratedRoadmaps }) {
    const [suggestions, setSuggestions] = useState([]);
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        const fetchSuggestions = async () => {
            setIsLoading(true);
            setGeneratedRoadmaps({});
            const prompt = `Based on this student profile (Age: ${profile.age}, Stream: ${profile.stream}, Interests: ${profile.interests.join(', ')}, Learning Style: ${profile.learningStyle}, Preferred Work Environment: ${profile.workEnv}, Project Experience: ${profile.projectExperience}, Team Preference: ${profile.teamPreference}, Problem Solving Style: ${profile.problemSolvingStyle}, 5-year Goal: ${profile.longTermGoal}), suggest 3 highly relevant career paths for the tech industry. For each, provide an 'id', 'title', and short 'description'.`;
            try {
                const resultString = await callGeminiAPI(prompt, true);
                const parsedResult = JSON.parse(resultString);
                setSuggestions(parsedResult);
            } catch (error) {
                setSuggestions(mockCareers); // Fallback to mock data on error
            } finally {
                setIsLoading(false);
            }
        };
        fetchSuggestions();
    }, [profile, setGeneratedRoadmaps]);

    const renderedSuggestions = suggestions.length > 0 ? suggestions : mockCareers;

    return (
        <div>
            <h1 className="text-4xl font-bold text-black dark:text-text-primary mb-2">Career Suggestions ✨</h1>
            <p className="text-text-secondary mb-8">Based on your profile, here are AI-curated paths that align with your interests.</p>
            {isLoading ? (
                <div className="flex justify-center items-center h-64"><GeminiLoader /></div>
            ) : (
                <motion.div
                    className="grid md:grid-cols-2 lg:grid-cols-3 gap-6"
                    variants={{ hidden: { opacity: 0 }, show: { opacity: 1, transition: { staggerChildren: 0.1 } } }}
                    initial="hidden"
                    animate="show"
                >
                    {renderedSuggestions.map(career => (
                        <motion.div
                            key={career.id}
                            className="holographic-card p-6 rounded-2xl cursor-pointer"
                            onClick={() => {
                                setSelectedCareer(career);
                                navigate('roadmap');
                                playFuturisticSound('navigate');
                            }}
                            variants={{ hidden: { y: 20, opacity: 0 }, show: { y: 0, opacity: 1 } }}
                        >
                            <div className="text-accent-text mb-4"><SparklesIcon /></div>
                            <h3 className="text-xl font-bold text-black dark:text-text-primary">{career.title}</h3>
                            <p className="text-text-secondary text-sm mt-2">{career.description}</p>
                        </motion.div>
                    ))}
                </motion.div>
            )}
        </div>
    );
};