import React from 'react';

export default function ProgressTracker({ generatedRoadmaps, progress }) {
    const allSkills = Object.values(generatedRoadmaps).flatMap(roadmap => roadmap.flatMap(phase => phase.skills));
    const uniqueSkills = [...new Set(allSkills)];
    const completedSkills = Object.keys(progress).filter(skill => progress[skill] && uniqueSkills.includes(skill)).length;
    const totalSkills = uniqueSkills.length;
    const progressPercentage = totalSkills > 0 ? Math.round((completedSkills / totalSkills) * 100) : 0;

    return (
        <div>
            <h1 className="text-4xl font-bold text-black dark:text-text-primary mb-2">Your Progress</h1>
            <p className="text-text-secondary mb-8">Track your skill acquisition journey across all recommended career paths.</p>
            <div className="holographic-card p-8 rounded-2xl">
                <h2 className="text-2xl font-bold text-accent-text-dark mb-4">Overall Completion</h2>
                <p className="text-black dark:text-text-primary mb-2">{completedSkills} / {totalSkills} unique skills completed</p>
                <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-4">
                    <div className="bg-sky-500 h-4 rounded-full transition-all duration-500" style={{ width: `${progressPercentage}%` }}></div>
                </div>
                <p className="text-right mt-2 font-bold text-2xl text-accent-text">{progressPercentage}%</p>
            </div>
        </div>
    );
};