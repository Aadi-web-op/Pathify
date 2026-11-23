"use client";

import clsx from "clsx";
import { animate, motion, useInView, useMotionValue, useReducedMotion, useScroll, useSpring, useTransform } from "motion/react";
import { type ReactNode, useEffect, useRef, useState } from "react";

/** Endless horizontal ticker. Content is duplicated once, so the second copy is hidden from screen readers. */
export function Marquee({
  children,
  reverse,
  duration = 40,
  className,
}: {
  children: ReactNode;
  reverse?: boolean;
  duration?: number;
  className?: string;
}) {
  return (
    <div className={clsx("group flex overflow-hidden [mask-image:linear-gradient(90deg,transparent,#000_8%,#000_92%,transparent)]", className)}>
      <div
        className="flex w-max shrink-0 animate-marquee group-hover:[animation-play-state:paused]"
        style={{ ["--marquee-duration" as string]: `${duration}s`, animationDirection: reverse ? "reverse" : "normal" }}
      >
        <div className="flex shrink-0">{children}</div>
        <div className="flex shrink-0" aria-hidden="true">
          {children}
        </div>
      </div>
    </div>
  );
}

/** Sets --mx / --my on every `.spotlight` child so its border and wash follow the pointer. */
export function SpotlightGroup({ children, className }: { children: ReactNode; className?: string }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const root = ref.current;
    if (!root) return;
    const onMove = (e: PointerEvent) => {
      root.querySelectorAll<HTMLElement>(".spotlight").forEach((el) => {
        const r = el.getBoundingClientRect();
        el.style.setProperty("--mx", `${e.clientX - r.left}px`);
        el.style.setProperty("--my", `${e.clientY - r.top}px`);
      });
    };
    root.addEventListener("pointermove", onMove);
    return () => root.removeEventListener("pointermove", onMove);
  }, []);
  return (
    <div ref={ref} className={className}>
      {children}
    </div>
  );
}

/** Fades and lifts its children in the first time they scroll into view. */
export function Reveal({ children, delay = 0, className }: { children: ReactNode; delay?: number; className?: string }) {
  const reduce = useReducedMotion();
  return (
    <motion.div
      className={className}
      initial={reduce ? false : { opacity: 0, y: 28, filter: "blur(8px)" }}
      whileInView={{ opacity: 1, y: 0, filter: "blur(0px)" }}
      viewport={{ once: true, margin: "0px 0px -12% 0px" }}
      transition={{ duration: 0.8, delay, ease: [0.2, 0.7, 0.2, 1] }}
    >
      {children}
    </motion.div>
  );
}

/** Counts up to `value` when it scrolls into view. */
export function CountUp({ value, suffix = "", className }: { value: number; suffix?: string; className?: string }) {
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true });
  const reduce = useReducedMotion();
  const [shown, setShown] = useState(reduce ? value : 0);
  useEffect(() => {
    if (!inView || reduce) return;
    const controls = animate(0, value, { duration: 1.6, ease: [0.2, 0.7, 0.2, 1], onUpdate: (v) => setShown(Math.round(v)) });
    return () => controls.stop();
  }, [inView, reduce, value]);
  return (
    <span ref={ref} className={clsx("tabular", className)}>
      {shown}
      {suffix}
    </span>
  );
}

/** A ruler down the right edge that fills as the page scrolls (wide screens only). */
export function ScrollRuler() {
  const { scrollYProgress } = useScroll();
  const smooth = useSpring(scrollYProgress, { stiffness: 140, damping: 30, mass: 0.4 });
  const top = useTransform(smooth, (v) => `${v * 100}%`);
  const [pct, setPct] = useState(0);
  useEffect(() => smooth.on("change", (v) => setPct(Math.round(v * 100))), [smooth]);
  return (
    <div aria-hidden="true" className="pointer-events-none fixed inset-y-24 right-3 z-40 hidden w-8 xl:block">
      <div className="absolute inset-y-0 right-0 flex w-full flex-col justify-between">
        {Array.from({ length: 41 }, (_, i) => (
          <span key={i} className={clsx("ml-auto h-px bg-current opacity-30", i % 5 === 0 ? "w-4" : "w-2")} />
        ))}
      </div>
      <motion.div className="absolute right-0 flex -translate-y-1/2 items-center gap-1.5" style={{ top }}>
        <span className="label-mono text-[0.6rem] text-sign">{String(pct).padStart(3, "0")}%</span>
        <span className="h-px w-6 bg-sign" />
      </motion.div>
    </div>
  );
}

/** Pulls its child gently towards the pointer while hovered. */
export function Magnetic({ children, strength = 0.25, className }: { children: ReactNode; strength?: number; className?: string }) {
  const ref = useRef<HTMLSpanElement>(null);
  const x = useMotionValue(0);
  const y = useMotionValue(0);
  const sx = useSpring(x, { stiffness: 260, damping: 18 });
  const sy = useSpring(y, { stiffness: 260, damping: 18 });
  const reduce = useReducedMotion();
  return (
    <motion.span
      ref={ref}
      className={clsx("inline-flex", className)}
      style={reduce ? undefined : { x: sx, y: sy }}
      onPointerMove={(e) => {
        if (e.pointerType !== "mouse") return;
        const r = ref.current!.getBoundingClientRect();
        x.set((e.clientX - (r.left + r.width / 2)) * strength);
        y.set((e.clientY - (r.top + r.height / 2)) * strength);
      }}
      onPointerLeave={() => {
        x.set(0);
        y.set(0);
      }}
    >
      {children}
    </motion.span>
  );
}
