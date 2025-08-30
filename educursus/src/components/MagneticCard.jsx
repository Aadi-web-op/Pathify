import { useRef, useState } from 'react';
import { motion } from 'framer-motion';

export function MagneticCard({ children }) {
  const ref = useRef(null);
  const [pos, setPos] = useState({ x: 0, y: 0 });

  const handleMouse = e => {
    const { left, top, width, height } = ref.current.getBoundingClientRect();
    const x = e.clientX - (left + width / 2);
    const y = e.clientY - (top + height / 2);
    setPos({ x: x * 0.08, y: y * 0.08 });
  };
  const reset = () => setPos({ x: 0, y: 0 });

  return (
    <motion.div
      ref={ref}
      onMouseMove={handleMouse}
      onMouseLeave={reset}
      animate={pos}
      transition={{ type: "spring", stiffness: 150, damping: 15 }}
      className="holographic-card flex-1 flex flex-col p-4 rounded-2xl"
    >
      {children}
    </motion.div>
  );
}
