import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { playFuturisticSound } from '../hooks/useSound';
import { mockCareers } from '../api/mockData';

export default function OnboardingSequence({ onComplete }) {
    const [step, setStep] = useState(0);
    const [profile, setProfile] = useState({ name: '', age: '', gender: '', stream: '', interests: [], careerInterests: [], learningStyle: '', workEnv: '', projectExperience: '', teamPreference: '', problemSolvingStyle: '', longTermGoal: '' });
    
    const questions = [
        { key: 'name', type: 'text', q: "First, what should we call you?" },
        { key: 'age', type: 'number', q: "How old are you?" },
        { key: 'stream', type: 'radio', q: "What's your current academic stream?", options: ['Science', 'Commerce', 'Arts', 'Engineering', 'Other'] },
        { key: 'interests', type: 'checkbox', q: "Which subjects spark your interest? (Pick a few)", options: ['Mathematics', 'Physics', 'Computer Science', 'Design', 'Business', 'Biology'] },
        { key: 'projectExperience', type: 'radio', q: "Have you worked on any personal or academic projects?", options: ['Yes, a few', 'Just getting started', 'No, not yet'] },
        { key: 'teamPreference', type: 'radio', q: "Do you prefer working in a team or independently?", options: ['In a Team', 'Independently', 'A bit of both'] },
        { key: 'problemSolvingStyle', type: 'radio', q: "How do you approach solving a difficult problem?", options: ['Logically & Step-by-step', 'Creatively & Brainstorming', 'By Researching Solutions'] },
        { key: 'learningStyle', type: 'radio', q: "How do you learn best?", options: ['By Doing (Practical)', 'By Reading (Theoretical)', 'By Watching (Visual)', 'By Collaborating'] },
        { key: 'workEnv', type: 'radio', q: "What kind of work environment excites you?", options: ['Fast-paced Startup', 'Large Tech Company', 'Freelance / Own Business', 'Research & Academia'] },
        { key: 'careerInterests', type: 'checkbox', q: "Any early career thoughts? (It's okay to guess!)", options: mockCareers.map(c => c.title) },
        { key: 'longTermGoal', type: 'text', q: "What's your ultimate 5-year career goal?" },
    ];

    const currentQ = questions[step];

    const handleNext = (e) => {
        e.preventDefault();
        playFuturisticSound();
        if (step < questions.length - 1) {
            setStep(step + 1);
        } else {
            onComplete(profile);
            playFuturisticSound('success');
        }
    };

    const handleChange = (e) => {
        const { name, value, type, checked } = e.target;
        if (type === 'checkbox') {
            setProfile(p => ({ ...p, [name]: checked ? [...p[name], value] : p[name].filter(v => v !== value) }));
        } else {
            setProfile(p => ({ ...p, [name]: value }));
        }
    };

    const progress = ((step + 1) / questions.length) * 100;

    return (
        <div className="w-full h-screen flex flex-col items-center justify-center p-4">
            <div className="w-full max-w-2xl">
                <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-2 mb-8">
                    <div className="bg-sky-500 h-2 rounded-full transition-all duration-500" style={{ width: `${progress}%` }}></div>
                </div>
                <AnimatePresence mode="wait">
                    <motion.div key={step} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }} transition={{ duration: 0.3 }}>
                        <form onSubmit={handleNext}>
                            <h2 className="text-3xl font-bold mb-6 text-black dark:text-text-primary">{currentQ.q}</h2>
                            {currentQ.type === 'text' && <input type="text" name={currentQ.key} value={profile[currentQ.key]} onChange={handleChange} required className="w-full bg-input-bg border border-input-border rounded-lg p-3 text-lg focus:outline-none focus:ring-2 focus:ring-sky-500 text-black dark:text-text-primary" />}
                            {currentQ.type === 'number' && <input type="number" name={currentQ.key} value={profile[currentQ.key]} onChange={handleChange} required className="w-full bg-input-bg border border-input-border rounded-lg p-3 text-lg focus:outline-none focus:ring-2 focus:ring-sky-500 text-black dark:text-text-primary" />}
                            {currentQ.type === 'radio' && <div className="grid grid-cols-2 gap-4">{currentQ.options.map(opt => <label key={opt} className={`p-4 border rounded-lg cursor-pointer transition-colors ${profile[currentQ.key] === opt ? 'bg-sky-500/20 border-sky-500' : 'border-input-border hover:border-sky-400'}`}><input type="radio" name={currentQ.key} value={opt} onChange={handleChange} checked={profile[currentQ.key] === opt} className="hidden" /> {opt} </label>)}</div>}
                            {currentQ.type === 'checkbox' && <div className="grid grid-cols-2 gap-4">{currentQ.options.map(opt => <label key={opt} className={`p-4 border rounded-lg cursor-pointer transition-colors ${profile[currentQ.key].includes(opt) ? 'bg-sky-500/20 border-sky-500' : 'border-input-border hover:border-sky-400'}`}><input type="checkbox" name={currentQ.key} value={opt} onChange={handleChange} checked={profile[currentQ.key].includes(opt)} className="hidden" /> {opt} </label>)}</div>}
                            <button type="submit" className="primary-btn mt-8 px-8 py-3">{step < questions.length - 1 ? 'Next' : 'Create My Profile'}</button>
                        </form>
                    </motion.div>
                </AnimatePresence>
            </div>
        </div>
    );
}
