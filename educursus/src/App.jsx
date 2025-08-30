import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

// Context and Providers
import { ThemeProvider } from './context/ThemeContext';

// Pages
import LandingPage from './pages/LandingPage';
import OnboardingSequence from './pages/OnboardingSequence';
import LoginPage from './pages/LoginPage';
import Dashboard from './pages/Dashboard';
import BigBrotherChat from './pages/BigBrotherChat';
import CareerSuggestions from './pages/CareerSuggestions';
import SkillRoadmap from './pages/SkillRoadmap';
import ProgressTracker from './pages/ProgressTracker';
import CertificatesPage from './pages/CertificatesPage';

// Components
import MagicSidebar from './components/MagicSidebar';
import GlobalStyles from './styles/GlobalStyles';
import { AuroraBackground, ThreeJSLandingBackground } from './assets/backgrounds';
import { playFuturisticSound } from './hooks/useSound';

export default function App() {
  const [userState, setUserState] = useState('loading'); // loading | landing | onboarding | login | app
  const [profile, setProfile] = useState(null);
  const [currentPage, setCurrentPage] = useState('dashboard');
  const [selectedCareer, setSelectedCareer] = useState(null);
  const [progress, setProgress] = useState({});
  const [tokens, setTokens] = useState([]);
  const [generatedRoadmaps, setGeneratedRoadmaps] = useState({});
  const [threeJsLoaded, setThreeJsLoaded] = useState(false);
  
  useEffect(() => { 
    if (!document.getElementById('threejs-script')) { 
        const script = document.createElement('script'); 
        script.id = 'threejs-script'; 
        script.src = "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"; 
        script.async = true; 
        script.onload = () => setThreeJsLoaded(true); 
        document.body.appendChild(script); 
    } else { 
        setThreeJsLoaded(true); 
    }
    
    const storedProfile = localStorage.getItem('educursus-profile'); 
    if (storedProfile) { 
        setProfile(JSON.parse(storedProfile)); 
        setUserState('login'); 
    } else { 
        setUserState('landing'); 
    } 
    
    const storedProgress = localStorage.getItem('educursus-progress'); 
    if(storedProgress) setProgress(JSON.parse(storedProgress));
    
    const storedTokens = localStorage.getItem('educursus-tokens');
    if(storedTokens) setTokens(JSON.parse(storedTokens));
  }, []);
  
  const handleOnboardingComplete = (newProfile) => { 
      localStorage.setItem('educursus-profile', JSON.stringify(newProfile)); 
      setProfile(newProfile); 
      setUserState('login'); 
  };
  
  const handleLogin = () => setUserState('app');
  const startOnboarding = () => setUserState('onboarding');
  
  const toggleSkill = (skill) => { 
      const newProgress = { ...progress, [skill]: !progress[skill] }; 
      setProgress(newProgress); 
      localStorage.setItem('educursus-progress', JSON.stringify(newProgress)); 
      playFuturisticSound(); 
  };
  
  const addToken = (project, phase) => {
      const newToken = { project, phase, date: new Date().toLocaleDateString() };
      const newTokens = [...tokens, newToken];
      setTokens(newTokens);
      localStorage.setItem('educursus-tokens', JSON.stringify(newTokens));
  }

  const renderContent = () => {
    switch (userState) {
      case 'landing': 
        return threeJsLoaded ? <LandingPage startOnboarding={startOnboarding} /> : <div className="h-screen w-full flex items-center justify-center">Loading Visuals...</div>;
      case 'onboarding': 
        return <OnboardingSequence onComplete={handleOnboardingComplete} />;
      case 'login': 
        return <LoginPage onLogin={handleLogin} />;
      case 'app': 
        return ( 
          <div className="flex h-screen"> 
            <MagicSidebar currentPage={currentPage} navigate={setCurrentPage} /> 
            <main className="flex-1 p-8 md:p-12 overflow-y-auto"> 
              <AnimatePresence mode="wait"> 
                <motion.div 
                  key={currentPage} 
                  initial={{ opacity: 0, y: 20 }} 
                  animate={{ opacity: 1, y: 0 }} 
                  exit={{ opacity: 0, y: -20 }} 
                  transition={{ duration: 0.5, type: 'spring', stiffness: 80, damping: 15 }}
                > 
                  {currentPage === 'dashboard' && <Dashboard profile={profile} progress={progress} tokens={tokens} />} 
                  {currentPage === 'big-brother' && <BigBrotherChat />} 
                  {currentPage === 'suggestions' && <CareerSuggestions profile={profile} navigate={setCurrentPage} setSelectedCareer={setSelectedCareer} setGeneratedRoadmaps={setGeneratedRoadmaps}/>} 
                  {currentPage === 'roadmap' && <SkillRoadmap selectedCareer={selectedCareer} progress={progress} toggleSkill={toggleSkill} generatedRoadmaps={generatedRoadmaps} setGeneratedRoadmaps={setGeneratedRoadmaps} addToken={addToken}/>} 
                  {currentPage === 'progress' && <ProgressTracker generatedRoadmaps={generatedRoadmaps} progress={progress} />} 
                  {currentPage === 'certificates' && <CertificatesPage tokens={tokens} profile={profile}/>} 
                </motion.div> 
              </AnimatePresence> 
            </main> 
          </div> 
        );
      default: 
        return <div className="h-screen w-full flex items-center justify-center">Initializing Console...</div>;
    }
  };

  return ( 
    <ThemeProvider> 
      <GlobalStyles /> 
      {userState === 'landing' && threeJsLoaded ? <ThreeJSLandingBackground /> : <AuroraBackground />} 
      {renderContent()} 
    </ThemeProvider> 
  );
}
