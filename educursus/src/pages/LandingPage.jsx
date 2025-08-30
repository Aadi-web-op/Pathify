import React from 'react';
import { motion } from 'framer-motion';
import { AnimatedText } from '../components/AnimatedText';
import { playFuturisticSound } from '../hooks/useSound';
import { EducursusLogo } from '../assets/icons';

export default function LandingPage({ startOnboarding }) {
    const featureSections = [
        { title: "Personalized Onboarding.", description: "Answer a few quick questions about your skills and interests. Our AI uses this information to build a profile that understands your unique starting point.", visual: () => <motion.div whileHover={{ scale: 1.1, rotate: 5 }} transition={{ type: "spring", stiffness: 300 }}><svg width="200" height="200" viewBox="0 0 200 200"><path d="M50 150 Q100 50 150 150" stroke="var(--accent)" strokeWidth="4" fill="none" strokeLinecap="round"/><circle cx="50" cy="150" r="8" fill="var(--accent-dark)"/><circle cx="150" cy="150" r="8" fill="var(--accent-dark)"/><circle cx="100" cy="93" r="12" fill="var(--bg)" stroke="var(--accent-dark)" strokeWidth="4"/></svg></motion.div> },
        { title: "AI-Curated Career Paths.", description: "Receive career suggestions perfectly aligned with your profile. Discover roles you might not have considered and see how your skills map to real-world jobs.", visual: () => <motion.div whileHover={{ scale: 1.1, rotate: -5 }} transition={{ type: "spring", stiffness: 300 }}><svg width="200" height="200" viewBox="0 0 200 200"><rect x="40" y="40" width="120" height="120" rx="10" fill="var(--bg)" stroke="var(--accent)" strokeWidth="4"/><path d="M70 70h60M70 100h60M70 130h40" stroke="var(--accent-dark)" strokeWidth="4" strokeLinecap="round"/></svg></motion.div> },
        { title: "Dynamic Learning Roadmaps.", description: "Once you choose a path, our AI generates a step-by-step learning roadmap. It's a clear plan from foundation to expert, broken down into manageable phases.", visual: () => <motion.div whileHover={{ scale: 1.1, rotate: 5 }} transition={{ type: "spring", stiffness: 300 }}><svg width="200" height="200" viewBox="0 0 200 200"><path d="M40 60h120M40 100h120M40 140h120" stroke="#CBD5E1" strokeWidth="4" strokeLinecap="round"/><circle cx="40" cy="60" r="8" fill="var(--accent-dark)"/><circle cx="80" cy="100" r="8" fill="var(--accent-dark)"/><circle cx="40" cy="140" r="8" fill="var(--accent-dark)"/></svg></motion.div> },
        { title: "Track Your Progress.", description: "Check off skills as you learn them and visualize your growth with our skill radar. See how close you are to your goal and stay motivated on your journey.", visual: () => <motion.div whileHover={{ scale: 1.1, rotate: -5 }} transition={{ type: "spring", stiffness: 300 }}><svg width="200" height="200" viewBox="0 0 200 200"><circle cx="100" cy="100" r="60" fill="none" stroke="#CBD5E1" strokeWidth="4"/><path d="M100 100 L 160 100 A 60 60 0 0 1 100 40 Z" fill="var(--accent)" fillOpacity="0.7"/></svg></motion.div> }
    ];

    return (
        <div className="w-full">
            <header className="fixed top-0 left-0 right-0 z-10 bg-white/50 dark:bg-slate-900/50 backdrop-blur-lg">
                <nav className="container mx-auto px-6 py-4 flex justify-between items-center">
                    <div className="flex items-center gap-3">
                        <EducursusLogo />
                        <span className="font-bold text-xl text-black dark:text-text-primary">Educursus</span>
                    </div>
                    <div className="hidden md:flex items-center gap-8 text-text-secondary font-semibold">
                        <a href="#features" className="hover:text-accent-text transition-colors">Features</a>
                        <a href="#" className="hover:text-accent-text transition-colors">FAQ</a>
                        <a href="#" className="hover:text-accent-text transition-colors">Contact</a>
                    </div>
                </nav>
            </header>
            <main>
                <section className="min-h-screen flex flex-col items-center justify-center text-center p-4 pt-20">
                    <AnimatedText text="Architect Your Future." el="h1" className="text-5xl md:text-7xl font-extrabold mb-4 text-black dark:text-text-primary" />
                    <AnimatedText text="Educursus is your personal AI co-pilot, transforming your interests into a clear, actionable career roadmap. Stop guessing, start building." el="p" className="text-lg md:text-xl text-text-secondary max-w-2xl mx-auto mb-8" />
                    <motion.div initial={{opacity:0, y:20}} animate={{opacity:1, y:0}} transition={{duration: 0.5, delay: 1.5}}>
                        <button onClick={() => { startOnboarding(); playFuturisticSound(); }} className="primary-btn px-10 py-4 text-lg">
                            Calibrate My Career Path
                        </button>
                    </motion.div>
                </section>
                <section id="features" className="py-20">
                    {featureSections.map((feature, index) => (
                        <div key={index} className={`container mx-auto px-6 py-16 flex flex-col md:flex-row items-center gap-12 ${index % 2 !== 0 ? 'md:flex-row-reverse' : ''}`}>
                            <motion.div initial={{opacity: 0, x: -50}} whileInView={{opacity: 1, x: 0}} viewport={{ once: true, amount: 0.5 }} transition={{duration: 0.6}} className="md:w-1/2 flex justify-center">
                                <feature.visual />
                            </motion.div>
                            <div className="md:w-1/2">
                                <AnimatedText text={feature.title} el="h2" className="text-3xl md:text-4xl font-extrabold mb-4 text-black dark:text-text-primary" />
                                <AnimatedText text={feature.description} el="p" className="text-text-secondary text-lg" />
                            </div>
                        </div>
                    ))}
                </section>
                <section className="text-center py-20 px-4">
                    <AnimatedText text="Ready to Build Your Future?" el="h2" className="text-4xl md:text-5xl font-extrabold mb-4 text-black dark:text-text-primary" />
                    <AnimatedText text="Your personalized career plan is just a few clicks away." el="p" className="text-lg text-text-secondary max-w-xl mx-auto mb-8" />
                    <motion.div initial={{opacity: 0, y: 20}} animate={{opacity: 1, y: 0}} transition={{duration: 0.5, delay: 0.8}}>
                        <button onClick={() => { startOnboarding(); playFuturisticSound(); }} className="primary-btn px-12 py-4 text-xl">
                            Start Now For Free
                        </button>
                    </motion.div>
                </section>
            </main>
        </div>
    );
}
