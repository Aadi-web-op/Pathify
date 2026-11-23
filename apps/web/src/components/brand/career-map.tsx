/**
 * A transit map of the student's best-fit careers, drawn from where they are now.
 * Each line is one career; each station is a real step on its route in India.
 * Lines link to the matching detail card on the page.
 */

export type MapStation = { label: string; sub?: string };

export type MapLine = {
  id: string;
  title: string;
  color: string;
  stations: MapStation[];
  /** Shown under the career name, e.g. "88% fit". */
  caption: string;
  /** A route the student can't take from their current stream is drawn dashed. */
  closed?: boolean;
  closedNote?: string;
};

type Props = { start: MapStation; lines: MapLine[]; label: string };

const WIDTH = 1040;
const START_X = 70;
const BRANCH_X = 150;
const TERMINUS_X = 830;
const LAST_STATION_X = 755;
const TRACK_GAP = 120;

/** Split text into at most two lines of roughly `max` characters, adding an ellipsis if it still overflows. */
function wrap(text: string, max: number): string[] {
  const words = text.split(" ");
  const lines: string[] = [""];
  for (const word of words) {
    const current = lines[lines.length - 1];
    if (!current) lines[lines.length - 1] = word;
    else if ((current + " " + word).length <= max) lines[lines.length - 1] = `${current} ${word}`;
    else if (lines.length < 2) lines.push(word);
    else {
      lines[1] = `${lines[1]} ${word}`;
    }
  }
  return lines.map((line) => (line.length > max + 2 ? `${line.slice(0, max).trimEnd()}…` : line));
}

export function CareerMap({ start, lines, label }: Props) {
  const count = lines.length;
  const height = 120 + count * TRACK_GAP;
  const midY = height / 2;
  const tracks = lines.map((_, i) => midY + (i - (count - 1) / 2) * TRACK_GAP);
  // Stations start after every line has branched, so no label sits on another line's diagonal.
  const maxDy = Math.max(...tracks.map((y) => Math.abs(y - midY)));
  const firstX = BRANCH_X + maxDy + 60;

  return (
    <figure className="m-0 overflow-x-auto">
      <svg viewBox={`0 0 ${WIDTH} ${height}`} className="block h-auto w-full min-w-[720px]" role="img" aria-labelledby="career-map-title">
        <title id="career-map-title">{label}</title>

        {lines.map((line, i) => {
          const y = tracks[i];
          const dy = Math.abs(y - midY);
          const n = line.stations.length;
          const xs = line.stations.map((_, k) =>
            n === 1 ? (firstX + LAST_STATION_X) / 2 : firstX + ((LAST_STATION_X - firstX) * k) / (n - 1),
          );
          const path = `M${START_X} ${midY} H${BRANCH_X} L${BRANCH_X + dy} ${y} H${TERMINUS_X}`;
          // Labels go above the top line and below the others, so they never sit on another track.
          const above = i === 0 && count > 1;
          const labelY = (lineIndex: number) => (above ? y - 44 + lineIndex * 15 : y + 30 + lineIndex * 15);
          const titleLines = wrap(line.title, 20);
          return (
            <a key={line.id} href={`#match-${line.id}`} aria-label={`${line.title}, ${line.caption}`}>
              <path
                d={path}
                pathLength={1}
                fill="none"
                stroke={line.color}
                strokeWidth={9}
                strokeLinecap="round"
                strokeLinejoin="round"
                className={line.closed ? undefined : "route-draw"}
                strokeDasharray={line.closed ? "0.02 0.018" : undefined}
                opacity={line.closed ? 0.55 : 1}
                style={{ ["--route-delay" as string]: `${i * 0.2}s` }}
              />
              {line.stations.map((station, k) => {
                const label = wrap(station.label, 15);
                return (
                  <g key={`${station.label}-${k}`}>
                    <circle cx={xs[k]} cy={y} r={8} fill="var(--surface)" stroke="var(--ink)" strokeWidth={3} />
                    {label.map((text, j) => (
                      <text key={j} x={xs[k]} y={labelY(j)} textAnchor="middle" className="fill-ink font-display text-[13px] font-bold">
                        {text}
                      </text>
                    ))}
                    {station.sub && (
                      <text x={xs[k]} y={labelY(label.length)} textAnchor="middle" className="fill-muted font-mono text-[11px]">
                        {station.sub}
                      </text>
                    )}
                  </g>
                );
              })}
              {/* Terminus: the career */}
              <circle cx={TERMINUS_X} cy={y} r={14} fill="var(--surface)" stroke={line.color} strokeWidth={5} />
              <circle cx={TERMINUS_X} cy={y} r={5} fill={line.color} />
              {titleLines.map((text, j) => (
                <text key={j} x={TERMINUS_X + 26} y={y - (titleLines.length - 1) * 9 + j * 18 - 4} className="fill-ink font-display text-[15px] font-bold">
                  {text}
                </text>
              ))}
              <text x={TERMINUS_X + 26} y={y + (titleLines.length - 1) * 9 + 16} className="fill-muted font-mono text-[12px]">
                {line.closed && line.closedNote ? line.closedNote : line.caption}
              </text>
            </a>
          );
        })}

        <circle cx={START_X} cy={midY} r={18} fill="var(--surface)" stroke="var(--ink)" strokeWidth={4} />
        <circle cx={START_X} cy={midY} r={6} fill="var(--ink)" />
        <text x={START_X} y={midY + 40} textAnchor="middle" className="fill-ink font-display text-[14px] font-bold">
          {start.label}
        </text>
        {start.sub && (
          <text x={START_X} y={midY + 56} textAnchor="middle" className="fill-muted font-mono text-[11px]">
            {start.sub}
          </text>
        )}
      </svg>
    </figure>
  );
}
