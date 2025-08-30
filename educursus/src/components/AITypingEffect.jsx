import React, { useState, useEffect } from 'react';

export const AITypingEffect = ({ text }) => {
    const [displayedText, setDisplayedText] = useState('');
    useEffect(() => {
        setDisplayedText('');
        let i = 0;
        const intervalId = setInterval(() => {
            setDisplayedText(text.substring(0, i + 1));
            i++;
            if (i > text.length) {
                clearInterval(intervalId);
            }
        }, 20);
        return () => clearInterval(intervalId);
    }, [text]);
    return <p className="text-black dark:text-text-primary text-center">{displayedText}</p>;
};