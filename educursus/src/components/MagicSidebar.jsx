import React from 'react';
import { useTheme } from '../context/ThemeContext';
import { playFuturisticSound } from '../hooks/useSound';
import { HomeIcon, MessageIcon, SparklesIcon, MapIcon, ChartIcon, AwardIcon, MoonIcon, SunIcon, EducursusLogo } from '../assets/icons';

export const MagicSidebar = ({ currentPage, navigate }) => {
    const { theme, setTheme } = useTheme();
    const items = [
        { name: 'Dashboard', icon: <HomeIcon /> },
        { name: 'Big Brother', icon: <MessageIcon /> },
        { name: 'Suggestions', icon: <SparklesIcon /> },
        { name: 'Roadmap', icon: <MapIcon /> },
        { name: 'Progress', icon: <ChartIcon /> },
        { name: 'Certificates', icon: <AwardIcon /> }
    ];
    return (
        <nav className="w-20 bg-sidebar-bg backdrop-blur-lg border-r border-sidebar-border p-4 flex flex-col items-center justify-between rounded-r-2xl shadow-lg">
            <div>
                <div className="mb-10"><EducursusLogo /></div>
                <ul className="flex flex-col gap-8">
                    {items.map(item => (
                        <li key={item.name}>
                            <button onClick={() => { navigate(item.name.toLowerCase().replace(' ', '-')); playFuturisticSound('navigate'); }} className="group flex flex-col items-center gap-1 text-xs text-text-secondary hover:text-accent-text transition-colors" title={item.name}>
                                <div className={`transition-transform duration-300 group-hover:scale-110 ${currentPage === item.name.toLowerCase().replace(' ', '-') ? 'text-accent-text' : ''}`}>{item.icon}</div>
                                <span className={`transition-colors ${currentPage === item.name.toLowerCase().replace(' ', '-') ? 'text-accent-text font-bold' : ''}`}>{item.name}</span>
                            </button>
                        </li>
                    ))}
                </ul>
            </div>
            <button onClick={() => { setTheme(theme === 'light' ? 'dark' : 'light'); playFuturisticSound(); }} className="p-2 rounded-full hover:bg-black/10 dark:hover:bg-white/10 transition-colors">
                {theme === 'light' ? <MoonIcon /> : <SunIcon />}
            </button>
        </nav>
    );
};

export default MagicSidebar;