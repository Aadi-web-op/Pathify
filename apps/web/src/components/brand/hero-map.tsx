/**
 * Three real routes out of Class 10, drawn as a transit map.
 * Each route is a different stream; the interchange at the end shows that
 * different streams can lead to the same kind of work.
 */

type Stop = { x: number; y: number; label: string; sub?: string; dy?: number };

const ROUTES = [
  {
    color: "var(--sign)",
    d: "M70 200 H250 L310 140 H520 L580 200 H690",
    delay: "0s",
    stops: [
      { x: 350, y: 140, label: "Science (PCM)", sub: "Class 11–12", dy: 34 },
      { x: 490, y: 140, label: "JEE / CUET", sub: "B.Tech · B.Sc", dy: 34 },
    ],
  },
  {
    color: "var(--route-saffron)",
    d: "M70 212 H180 L228 260 H540 L588 212 H690",
    delay: "0.2s",
    stops: [
      { x: 320, y: 260, label: "Commerce", sub: "Class 11–12", dy: 34 },
      { x: 460, y: 260, label: "CUET", sub: "B.Com · BBA", dy: 34 },
    ],
  },
  {
    color: "var(--route-blue)",
    d: "M70 188 H130 L238 80 H560 L668 188 H690",
    delay: "0.4s",
    stops: [
      { x: 320, y: 80, label: "Humanities", sub: "Class 11–12", dy: -40 },
      { x: 470, y: 80, label: "NID DAT / CUET", sub: "B.Des · BA Psych", dy: -40 },
    ],
  },
] satisfies { color: string; d: string; delay: string; stops: Stop[] }[];

export function HeroMap() {
  return (
    <figure className="m-0 overflow-x-auto">
      <svg
        viewBox="0 0 800 330"
        className="block h-auto w-full min-w-[560px]"
        role="img"
        aria-labelledby="hero-map-title"
      >
        <title id="hero-map-title">
          Three routes from Class 10: Science to engineering, Commerce to business degrees, and Humanities to design or
          psychology. All three can lead to product and data roles.
        </title>
        {ROUTES.map((route) => (
          <path
            key={route.d}
            d={route.d}
            pathLength={1}
            fill="none"
            stroke={route.color}
            strokeWidth={9}
            strokeLinecap="round"
            strokeLinejoin="round"
            className="route-draw"
            style={{ ["--route-delay" as string]: route.delay }}
          />
        ))}

        {ROUTES.flatMap((route) =>
          route.stops.map((stop) => (
            <g key={`${route.d}-${stop.label}`}>
              <circle cx={stop.x} cy={stop.y} r={9} fill="var(--surface)" stroke="var(--ink)" strokeWidth={3.5} />
              <text
                x={stop.x}
                y={stop.y + (stop.dy ?? 30)}
                textAnchor="middle"
                className="fill-ink font-display text-[15px] font-bold"
              >
                {stop.label}
              </text>
              {stop.sub && (
                <text
                  x={stop.x}
                  y={stop.y + (stop.dy ?? 30) + 17}
                  textAnchor="middle"
                  className="fill-muted font-mono text-[12px]"
                >
                  {stop.sub}
                </text>
              )}
            </g>
          )),
        )}

        {/* Start: the student */}
        <circle cx={70} cy={200} r={19} fill="var(--surface)" stroke="var(--ink)" strokeWidth={4} />
        <circle cx={70} cy={200} r={6} fill="var(--ink)" />
        <text x={70} y={246} textAnchor="middle" className="fill-ink font-display text-[15px] font-bold">
          You, Class 10
        </text>

        {/* Interchange: where the routes meet */}
        <rect x={668} y={176} width={64} height={48} rx={24} fill="var(--surface)" stroke="var(--ink)" strokeWidth={4} />
        <text x={700} y={250} textAnchor="middle" className="fill-ink font-display text-[15px] font-bold">
          Product, data,
        </text>
        <text x={700} y={268} textAnchor="middle" className="fill-ink font-display text-[15px] font-bold">
          research roles
        </text>
      </svg>
    </figure>
  );
}
