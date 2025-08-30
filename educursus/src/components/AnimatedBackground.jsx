import { motion } from 'framer-motion';

export function AnimatedBackground() {
  return (
    <motion.div
      initial={{
        background: 'linear-gradient(120deg, #89f7fe 0%, #66a6ff 100%)'
      }}
      animate={{
        background: [
          'linear-gradient(120deg, #89f7fe 0%, #66a6ff 100%)',
          'linear-gradient(120deg, #fcb6f6 0%, #5ee7df 100%)',
          'linear-gradient(120deg, #89f7fe 0%, #66a6ff 100%)',
        ],
      }}
      transition={{
        duration: 8,
        repeat: Infinity,
        ease: "linear"
      }}
      className="fixed inset-0 -z-10"
      style={{ minHeight: '100vh', minWidth: '100vw' }}
    />
  );
}
