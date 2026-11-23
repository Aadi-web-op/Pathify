"use client";

import { useEffect, useRef } from "react";

/**
 * A night transit map drawn on canvas: neon routes that draw themselves in,
 * then light "trains" that run along them. The map leans with the pointer.
 *
 * Purely decorative (aria-hidden). It pauses off-screen and in background tabs,
 * and with reduced motion it draws the finished map once and stops.
 */

type Pt = { x: number; y: number };
type Line = { pts: Pt[]; lens: number[]; total: number; color: string; depth: number; width: number; delay: number; stations: number[] };

// Small seeded random so the map is the same on every visit (and in screenshots).
function seeded(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function buildLines(w: number, h: number, colors: string[]): Line[] {
  // A hidden canvas has no size; with nothing to span, the route builder below would never finish.
  if (w < 40 || h < 40) return [];
  const rnd = seeded(20251123);
  const count = w < 640 ? 6 : 9;
  const lines: Line[] = [];
  for (let i = 0; i < count; i++) {
    const depth = 0.35 + rnd() * 0.65;
    let y = h * (0.08 + (0.84 * i) / (count - 1)) + (rnd() - 0.5) * 40;
    let x = -40;
    const pts: Pt[] = [{ x, y }];
    const stations: number[] = [];
    let horizontal = true;
    while (x < w + 40) {
      if (horizontal) {
        x += w * (0.08 + rnd() * 0.2);
        pts.push({ x, y });
        if (rnd() > 0.35) stations.push(pts.length - 1);
      } else {
        const target = Math.max(h * 0.04, Math.min(h * 0.96, y + (rnd() - 0.5) * h * 0.55));
        const dy = target - y;
        x += Math.abs(dy);
        y = target;
        pts.push({ x, y });
      }
      horizontal = !horizontal;
    }
    const lens: number[] = [0];
    for (let k = 1; k < pts.length; k++) lens.push(lens[k - 1] + Math.hypot(pts[k].x - pts[k - 1].x, pts[k].y - pts[k - 1].y));
    lines.push({
      pts,
      lens,
      total: lens[lens.length - 1],
      color: colors[i % colors.length],
      depth,
      width: 2 + depth * 3.5,
      delay: i * 0.12,
      stations,
    });
  }
  return lines.sort((a, b) => a.depth - b.depth);
}

function pointAt(line: Line, d: number): Pt {
  const { pts, lens } = line;
  if (d <= 0) return pts[0];
  if (d >= line.total) return pts[pts.length - 1];
  let k = 1;
  while (lens[k] < d) k++;
  const t = (d - lens[k - 1]) / (lens[k] - lens[k - 1]);
  return { x: pts[k - 1].x + (pts[k].x - pts[k - 1].x) * t, y: pts[k - 1].y + (pts[k].y - pts[k - 1].y) * t };
}

function tracePath(ctx: CanvasRenderingContext2D, line: Line, from: number, to: number) {
  const start = pointAt(line, from);
  ctx.beginPath();
  ctx.moveTo(start.x, start.y);
  for (let k = 1; k < line.pts.length; k++) {
    if (line.lens[k] <= from) continue;
    if (line.lens[k] >= to) break;
    ctx.lineTo(line.pts[k].x, line.pts[k].y);
  }
  const end = pointAt(line, to);
  ctx.lineTo(end.x, end.y);
}

export function NeonMetro({ className }: { className?: string }) {
  const ref = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    let w = 0;
    let h = 0;
    let lines: Line[] = [];
    let raf = 0;
    let visible = true;
    const started = performance.now();
    const pointer = { x: 0, y: 0, tx: 0, ty: 0 };

    const css = getComputedStyle(canvas);
    const colors = ["--route-violet", "--route-pink", "--route-blue", "--route-saffron", "--pop"].map(
      (v) => css.getPropertyValue(v).trim() || "#a48dff",
    );
    const paper = css.getPropertyValue("--paper").trim() || "#08080d";

    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
      w = rect.width;
      h = rect.height;
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      lines = buildLines(w, h, colors);
      if (reduce) draw(performance.now() + 10_000);
    };

    const draw = (now: number) => {
      const t = (now - started) / 1000;
      pointer.x += (pointer.tx - pointer.x) * 0.06;
      pointer.y += (pointer.ty - pointer.y) * 0.06;
      ctx.clearRect(0, 0, w, h);
      ctx.lineCap = "round";
      ctx.lineJoin = "round";

      for (const line of lines) {
        const ox = pointer.x * 26 * line.depth;
        const oy = pointer.y * 18 * line.depth;
        ctx.save();
        ctx.translate(ox, oy);
        const grow = Math.min(Math.max((t - line.delay) / 2.2, 0), 1);
        const eased = 1 - Math.pow(1 - grow, 3);
        const drawn = line.total * eased;
        const alpha = 0.18 + line.depth * 0.5;

        // Glow, then the line itself.
        ctx.globalAlpha = alpha * 0.25;
        ctx.strokeStyle = line.color;
        ctx.lineWidth = line.width * 4;
        tracePath(ctx, line, 0, drawn);
        ctx.stroke();
        ctx.globalAlpha = alpha;
        ctx.lineWidth = line.width;
        tracePath(ctx, line, 0, drawn);
        ctx.stroke();

        // Stations: hollow rings once the line has reached them.
        for (const s of line.stations) {
          if (line.lens[s] > drawn) continue;
          const p = line.pts[s];
          ctx.globalAlpha = Math.min(alpha + 0.2, 1);
          ctx.fillStyle = paper;
          ctx.strokeStyle = line.color;
          ctx.lineWidth = Math.max(line.width * 0.7, 2);
          ctx.beginPath();
          ctx.arc(p.x, p.y, line.width * 1.4 + 2, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
        }

        // Trains: a bright head with a fading tail, once the line is drawn.
        if (!reduce && grow >= 1) {
          const speed = 90 + line.depth * 160;
          const tail = 120 + line.depth * 120;
          for (let k = 0; k < 2; k++) {
            const head = ((t * speed + k * line.total * 0.5 + line.delay * 900) % (line.total + tail)) - 0;
            const from = Math.max(head - tail, 0);
            const to = Math.min(head, line.total);
            if (to <= from) continue;
            const a = pointAt(line, from);
            const b = pointAt(line, to);
            const grad = ctx.createLinearGradient(a.x, a.y, b.x, b.y);
            grad.addColorStop(0, "rgba(255,255,255,0)");
            grad.addColorStop(1, "rgba(255,255,255,0.95)");
            ctx.globalAlpha = 0.35 + line.depth * 0.65;
            ctx.strokeStyle = grad;
            ctx.lineWidth = line.width * 0.9;
            tracePath(ctx, line, from, to);
            ctx.stroke();
            if (head <= line.total) {
              // A soft halo, then the bright head (cheaper than canvas shadows).
              ctx.fillStyle = line.color;
              ctx.globalAlpha = 0.35;
              ctx.beginPath();
              ctx.arc(b.x, b.y, line.width * 2.6 + 3, 0, Math.PI * 2);
              ctx.fill();
              ctx.globalAlpha = 1;
              ctx.fillStyle = "#ffffff";
              ctx.beginPath();
              ctx.arc(b.x, b.y, line.width * 0.9 + 1, 0, Math.PI * 2);
              ctx.fill();
            }
          }
        }
        ctx.restore();
      }
    };

    let lastFrame = 0;
    const loop = (now: number) => {
      if (now - lastFrame > 21) {
        lastFrame = now;
        draw(now);
      }
      raf = visible && !document.hidden ? requestAnimationFrame(loop) : 0;
    };
    const wake = () => {
      if (!reduce && !raf && visible && !document.hidden) raf = requestAnimationFrame(loop);
    };

    const onPointer = (e: PointerEvent) => {
      pointer.tx = e.clientX / window.innerWidth - 0.5;
      pointer.ty = e.clientY / window.innerHeight - 0.5;
    };

    const ro = new ResizeObserver(resize);
    ro.observe(canvas);
    const io = new IntersectionObserver(([entry]) => {
      visible = entry.isIntersecting;
      wake();
    });
    io.observe(canvas);
    document.addEventListener("visibilitychange", wake);
    window.addEventListener("pointermove", onPointer, { passive: true });
    resize();
    wake();

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      io.disconnect();
      document.removeEventListener("visibilitychange", wake);
      window.removeEventListener("pointermove", onPointer);
    };
  }, []);

  return <canvas ref={ref} aria-hidden="true" className={className} />;
}
