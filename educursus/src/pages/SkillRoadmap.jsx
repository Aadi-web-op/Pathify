// src/pages/SkillRoadmap.jsx
import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { callGeminiAPI } from '../api/gemini';
import { mockRoadmaps } from '../api/mockData';
import { GeminiLoader } from '../components/GeminiLoader';
import { SkillGapChart } from '../components/SkillGapChart';
import { playFuturisticSound } from '../hooks/useSound';

export default function SkillRoadmap({ selectedCareer, progress, toggleSkill, generatedRoadmaps, setGeneratedRoadmaps, addToken }) {
    const [isLoading, setIsLoading] = useState(false);
    const [miniProject, setMiniProject] = useState(null);
    const [isLoadingProject, setIsLoadingProject] = useState(false);

    const handleGenerateProject = async (phaseName, skillsInPhase) => {
        setIsLoadingProject(true);
        setMiniProject(null);
        playFuturisticSound();
        const prompt = `Create a small, actionable mini-project for a student learning about '${selectedCareer.title}'. The project should test these specific skills: ${skillsInPhase.join(', ')}. The phase is called '${phaseName}'. Provide a 'title', a short 'description', and an array of 3 'tasks'. Format as JSON.`;
        try {
            const resultString = await callGeminiAPI(prompt, true);
            const parsedResult = JSON.parse(resultString);
            setMiniProject({ ...parsedResult, phase: phaseName });
        } catch (error) {
            setMiniProject({ title: "Project Generation Error", description: "Could not generate a project. Please try again.", tasks: [], phase: phaseName });
        } finally {
            setIsLoadingProject(false);
        }
    };

    useEffect(() => {
        if (selectedCareer && !generatedRoadmaps[selectedCareer.id]) {
            const fetchRoadmap = async () => {
                setIsLoading(true);
                const prompt = `Generate a 5-phase learning roadmap for a student aspiring to become a '${selectedCareer.title}'. For each phase, provide a 'name' and an array of 3-4 key 'skills'.`;
                try {
                    const resultString = await callGeminiAPI(prompt, true);
                    const parsedResult = JSON.parse(resultString);
                    setGeneratedRoadmaps(prev => ({ ...prev, [selectedCareer.id]: parsedResult }));
                } catch (error) {
                    setGeneratedRoadmaps(prev => ({ ...prev, [selectedCareer.id]: mockRoadmaps['data-scientist'] }));
                } finally {
                    setIsLoading(false);
                }
            };
            fetchRoadmap();
        }
    }, [selectedCareer, generatedRoadmaps, setGeneratedRoadmaps]);

    const roadmap = generatedRoadmaps[selectedCareer?.id] || [];
    const allSkills = roadmap.flatMap(phase => phase.skills);

    if (!selectedCareer) {
        return <div className="text-center p-8"><h2 className="text-2xl font-bold text-text-secondary">Please select a career suggestion first.</h2></div>
    }

    return (
        <div>
            <h1 className="text-4xl font-bold text-black dark:text-text-primary mb-2">Skill Roadmap: <span className="text-accent-text">{selectedCareer.title}</span> ✨</h1>
            <p className="text-text-secondary mb-8">Your AI-generated journey from foundation to deployment. Check off skills as you learn them.</p>
            {isLoading || roadmap.length === 0 ? <div className="flex justify-center items-center h-64"><GeminiLoader /></div> :
                <div className="flex flex-col lg:flex-row gap-8">
                    <div className="lg:w-2/3">
                        <h3 className="text-xl font-bold mb-4 text-text-secondary">Learning Phases</h3>
                        <div className="relative">{roadmap.map((phase, index) => (
                            <div key={index} className="stepper-item pl-14 relative pb-8">
                                <div className="absolute left-0 top-0 w-10 h-10 bg-sky-500/20 border-2 border-sky-500 text-sky-600 rounded-full flex items-center justify-center font-bold">{index + 1}</div>
                                <h4 className="font-bold text-black dark:text-text-primary text-lg pt-2">{phase.name}</h4>
                                <ul className="mt-2 space-y-2">{phase.skills.map(skill => (
                                    <li key={skill} className="flex items-center gap-3">
                                        <input type="checkbox" id={skill} checked={!!progress[skill]} onChange={() => toggleSkill(skill)} className="checkbox-custom appearance-none w-5 h-5 rounded bg-slate-200 dark:bg-slate-700 border border-input-border checked:bg-sky-500 checked:border-sky-500 transition" />
                                        <label htmlFor={skill} className={`cursor-pointer ${progress[skill] ? 'line-through text-slate-400' : 'text-text-secondary'}`}>{skill}</label>
                                    </li>
                                ))}</ul>
                                <button onClick={() => handleGenerateProject(phase.name, phase.skills)} className="text-sm primary-btn px-3 py-1 mt-3">Generate Mini-Project</button>
                            </div>
                        ))}</div>
                    </div>
                    <div className="lg:w-1/3">
                        <div className="holographic-card p-6 rounded-2xl sticky top-8">
                            <h3 className="text-xl font-bold mb-4 text-text-secondary text-center">Skill Gap Radar</h3>
                            <SkillGapChart skills={allSkills} progress={progress} />
                            {isLoadingProject && <div className="mt-4"><GeminiLoader /></div>}
                            {miniProject && (
                                <div className="mt-4">
                                    <h4 className="font-bold text-lg text-accent-text-dark">{miniProject.title}</h4>
                                    <p className="text-sm text-text-secondary mt-1">{miniProject.description}</p>
                                    <ul className="text-sm list-disc list-inside mt-2 space-y-1">
                                        {miniProject.tasks.map((task, i) => <li key={i}>{task}</li>)}
                                    </ul>
                                    <button onClick={() => { addToken(miniProject.title, miniProject.phase); setMiniProject(null); playFuturisticSound('success'); }} className="primary-btn w-full mt-4 py-2">Mark as Complete & Get Token</button>
                                </div>
                            )}
                        </div>
                    </div>
                </div>}
        </div>
    );
};