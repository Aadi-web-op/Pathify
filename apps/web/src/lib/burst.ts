/**
 * Impact bursts: a short, drawn celebration on one shared canvas over the page.
 *
 * The idea (springy controls that throw drawn effects on confirm) comes from
 * sakuga-vfx by Justin Finkelman (MIT). This is a smaller, original
 * implementation with Pathify's route colours and no lettering.
 *
 * Effects never block input (the canvas ignores the pointer), stop drawing when
 * idle, and do nothing when the user prefers reduced motion.
 */

type Particle = {
  kind: "wedge" | "puff" | "chip" | "ring" | "flash";
  x: number;
  y: number;
  vx: number;
  vy: number;
  angle: number;
  spin: number;
  size: number;
  life: number;
  age: number;
  color: string;
};

export type BurstOptions = {
  /** Scale of the whole effect: 1 for a button, 1.6 for a page moment. */
  power?: number;
  /** Add falling confetti chips. */
  confetti?: boolean;
};

let canvas: HTMLCanvasElement | null = null;
let ctx: CanvasRenderingContext2D | null = null;
let particles: Particle[] = [];
let frame = 0;
let last = 0;

function reducedMotion() {
  return typeof window === "undefined" || window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function palette(): string[] {
  const css = getComputedStyle(document.documentElement);
  const read = (name: string, fallback: string) => css.getPropertyValue(name).trim() || fallback;
  return [
    read("--route-violet", "#a48dff"),
    read("--route-pink", "#ff5ad9"),
    read("--route-saffron", "#ff9450"),
    read("--route-blue", "#3fd8ff"),
    read("--pop", "#d7ff3c"),
  ];
}

function ensureCanvas() {
  if (canvas) return;
  canvas = document.createElement("canvas");
  canvas.setAttribute("aria-hidden", "true");
  Object.assign(canvas.style, {
    position: "fixed",
    inset: "0",
    width: "100vw",
    height: "100vh",
    pointerEvents: "none",
    zIndex: "70",
  });
  document.body.appendChild(canvas);
  ctx = canvas.getContext("2d");
  resize();
  window.addEventListener("resize", resize);
}

function resize() {
  if (!canvas || !ctx) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(window.innerWidth * dpr);
  canvas.height = Math.round(window.innerHeight * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

const rand = (a: number, b: number) => a + Math.random() * (b - a);
const easeOut = (t: number) => 1 - Math.pow(1 - t, 3);

export function burst(x: number, y: number, options: BurstOptions = {}) {
  if (reducedMotion()) return;
  ensureCanvas();
  const power = options.power ?? 1;
  const colors = palette();
  const pick = () => colors[Math.floor(Math.random() * colors.length)];

  particles.push({ kind: "flash", x, y, vx: 0, vy: 0, angle: 0, spin: 0, size: 22 * power, life: 0.12, age: 0, color: "#ffffff" });
  particles.push({ kind: "ring", x, y, vx: 0, vy: 0, angle: 0, spin: 0, size: 90 * power, life: 0.55, age: 0, color: colors[0] });
  particles.push({ kind: "ring", x, y, vx: 0, vy: 0, angle: 0, spin: 0, size: 60 * power, life: 0.42, age: -0.06, color: colors[4] });

  const wedges = Math.round(12 * power);
  for (let i = 0; i < wedges; i++) {
    const a = (i / wedges) * Math.PI * 2 + rand(-0.12, 0.12);
    const speed = rand(420, 720) * power;
    particles.push({
      kind: "wedge", x, y, vx: Math.cos(a) * speed, vy: Math.sin(a) * speed, angle: a, spin: 0,
      size: rand(14, 26) * power, life: rand(0.28, 0.42), age: 0, color: i % 2 ? colors[1] : colors[0],
    });
  }
  for (let i = 0; i < 5; i++) {
    const a = rand(0, Math.PI * 2);
    const speed = rand(60, 150) * power;
    particles.push({
      kind: "puff", x, y, vx: Math.cos(a) * speed, vy: Math.sin(a) * speed - 30, angle: 0, spin: 0,
      size: rand(14, 26) * power, life: rand(0.55, 0.8), age: 0, color: colors[3],
    });
  }
  if (options.confetti) {
    for (let i = 0; i < Math.round(34 * power); i++) {
      const a = rand(-Math.PI * 0.95, -Math.PI * 0.05);
      const speed = rand(320, 760) * power;
      particles.push({
        kind: "chip", x, y, vx: Math.cos(a) * speed, vy: Math.sin(a) * speed, angle: rand(0, 6), spin: rand(-12, 12),
        size: rand(5, 10), life: rand(1.1, 1.8), age: 0, color: pick(),
      });
    }
  }
  if (!frame) {
    last = performance.now();
    frame = requestAnimationFrame(tick);
  }
}

/** Burst from the centre of an element, e.g. the button that was just pressed. */
export function burstFrom(el: Element | null | undefined, options?: BurstOptions) {
  if (!el) return;
  const r = el.getBoundingClientRect();
  burst(r.left + r.width / 2, r.top + r.height / 2, options);
}

function tick(now: number) {
  if (!ctx || !canvas) return;
  const dt = Math.min((now - last) / 1000, 1 / 30);
  last = now;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  particles = particles.filter((p) => p.age < p.life);
  for (const p of particles) {
    p.age += dt;
    if (p.age < 0) continue;
    const t = Math.min(p.age / p.life, 1);
    ctx.save();
    switch (p.kind) {
      case "flash": {
        ctx.globalAlpha = 1 - t;
        ctx.fillStyle = p.color;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size * (0.6 + easeOut(t) * 0.6), 0, Math.PI * 2);
        ctx.fill();
        break;
      }
      case "ring": {
        ctx.globalAlpha = (1 - t) * 0.9;
        ctx.strokeStyle = p.color;
        ctx.lineWidth = 3 * (1 - t) + 0.5;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size * easeOut(t), 0, Math.PI * 2);
        ctx.stroke();
        break;
      }
      case "wedge": {
        // A speed line: a thin triangle pointing outwards, stretching then thinning.
        p.x += p.vx * dt;
        p.y += p.vy * dt;
        p.vx *= 0.9;
        p.vy *= 0.9;
        const len = p.size * (1.6 - t);
        const w = (p.size / 5) * (1 - t);
        ctx.globalAlpha = 1 - t * t;
        ctx.fillStyle = p.color;
        ctx.translate(p.x, p.y);
        ctx.rotate(p.angle);
        ctx.beginPath();
        ctx.moveTo(len, 0);
        ctx.lineTo(-len * 0.4, w);
        ctx.lineTo(-len * 0.4, -w);
        ctx.closePath();
        ctx.fill();
        break;
      }
      case "puff": {
        p.x += p.vx * dt;
        p.y += p.vy * dt;
        p.vx *= 0.94;
        p.vy *= 0.94;
        ctx.globalAlpha = (1 - t) * 0.85;
        ctx.strokeStyle = p.color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size * (0.5 + easeOut(t) * 0.7), 0, Math.PI * 2);
        ctx.stroke();
        break;
      }
      case "chip": {
        p.vy += 1100 * dt;
        p.vx *= 0.985;
        p.x += p.vx * dt;
        p.y += p.vy * dt;
        p.angle += p.spin * dt;
        ctx.globalAlpha = t > 0.75 ? (1 - t) / 0.25 : 1;
        ctx.fillStyle = p.color;
        ctx.translate(p.x, p.y);
        ctx.rotate(p.angle);
        ctx.scale(1, Math.cos(p.age * 9));
        ctx.fillRect(-p.size / 2, -p.size / 4, p.size, p.size / 2);
        break;
      }
    }
    ctx.restore();
  }

  if (particles.length) {
    frame = requestAnimationFrame(tick);
  } else {
    frame = 0;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }
}
