import React from 'react';
import { motion } from 'framer-motion';
import { AwardIcon } from '../assets/icons';

export default function CertificatesPage({ tokens, profile }) {
    return (
        <div>
            <h1 className="text-4xl font-bold text-black dark:text-text-primary mb-2">Proof-of-Skill Tokens</h1>
            <p className="text-text-secondary mb-8">Your collection of micro-certificates earned from completing projects.</p>
            {tokens.length === 0 ? (
                <p className="text-text-secondary">No tokens earned yet. Complete mini-projects on the Roadmap page to earn them!</p>
            ) : (
                <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {tokens.map((token, index) => (
                        <motion.div
                            key={index}
                            className="holographic-card p-6 rounded-2xl text-center"
                            initial={{ opacity: 0, scale: 0.8 }}
                            animate={{ opacity: 1, scale: 1 }}
                            transition={{ delay: index * 0.1 }}
                        >
                            <div className="text-accent-text mb-4 mx-auto w-fit"><AwardIcon /></div>
                            <h3 className="text-lg font-bold text-black dark:text-text-primary">{token.project}</h3>
                            <p className="text-sm text-text-secondary mt-1">Phase: {token.phase}</p>
                            <p className="text-xs text-text-secondary mt-4">Issued to: {profile.name}</p>
                            <p className="text-xs text-text-secondary">Date: {token.date}</p>
                        </motion.div>
                    ))}
                </div>
            )}
        </div>
    );
};
