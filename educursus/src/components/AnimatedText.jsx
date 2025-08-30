import React, { useRef } from 'react';
import { motion, useInView } from 'framer-motion';

export const AnimatedText = ({ text, el: Wrapper = 'p', className }) => {
    const textChars = text.split('');
    const ref = useRef(null);
    const isInView = useInView(ref, { once: true, amount: 0.5 });
    const containerVariants = {
        hidden: { opacity: 0 },
        visible: { opacity: 1, transition: { staggerChildren: 0.02, delayChildren: 0.1 } },
    };
    const childVariants = {
        hidden: { opacity: 0, y: 15, x: -10 },
        visible: { opacity: 1, y: 0, x: 0, transition: { type: 'spring', damping: 12, stiffness: 100 } },
    };
    return (
        <Wrapper ref={ref} className={className}>
            <motion.span aria-label={text} role="heading" variants={containerVariants} initial="hidden" animate={isInView ? 'visible' : 'hidden'} >
                {textChars.map((char, index) => (
                    <motion.span key={index} variants={childVariants} aria-hidden="true" style={{ display: 'inline-block' }}>
                        {char === ' ' ? '\u00A0' : char}
                    </motion.span>
                ))}
            </motion.span>
        </Wrapper>
    );
};