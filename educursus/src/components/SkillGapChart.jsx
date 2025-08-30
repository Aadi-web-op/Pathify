import React from 'react';
import { RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ResponsiveContainer, Legend, Tooltip } from 'recharts';
import { useTheme } from '../context/ThemeContext';

export const SkillGapChart = ({ skills, progress }) => {
    const { theme } = useTheme();
    const data = skills.map(skill => ({
        subject: skill,
        'Required': 100,
        'Your Skills': progress[skill] ? 90 : 20, // Gamified values
        fullMark: 100,
    }));

    return (
        <ResponsiveContainer width="100%" height={350}>
            <RadarChart cx="50%" cy="50%" outerRadius="80%" data={data}>
                <PolarGrid stroke={theme === 'light' ? "#E0E7FF" : "#475569"} />
                <PolarAngleAxis dataKey="subject" tick={{ fill: theme === 'light' ? '#475569' : '#94A3B8', fontSize: 12 }} />
                <PolarRadiusAxis angle={30} domain={[0, 100]} tick={false} axisLine={false} />
                <Tooltip
                    contentStyle={{
                        backgroundColor: 'var(--card-bg)',
                        borderColor: 'var(--card-border)',
                        backdropFilter: 'blur(10px)',
                    }}
                />
                <Legend />
                <Radar name="Required" dataKey="Required" stroke="#8884d8" fill="#8884d8" fillOpacity={0.3} />
                <Radar name="Your Skills" dataKey="Your Skills" stroke="var(--accent)" fill="var(--accent)" fillOpacity={0.7} />
            </RadarChart>
        </ResponsiveContainer>
    );
};
