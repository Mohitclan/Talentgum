import type { ReactNode } from "react";
import type { Assessment } from "../api";
import { STATUS_LABEL, level, type GroupScore } from "../model";

const PATHS: Record<string, ReactNode> = {
  home: <path d="M3 11l9-7 9 7v9a1 1 0 0 1-1 1h-5v-6H9v6H4a1 1 0 0 1-1-1z" />,
  plus: <path d="M12 5v14M5 12h14" />,
  newc: <><circle cx="12" cy="12" r="9" /><path d="M12 8v8M8 12h8" /></>,
  chart: <path d="M4 20V10M10 20V4M16 20v-7M22 20H2" />,
  people: <><circle cx="9" cy="8" r="3.5" /><path d="M2.5 20c.8-3.5 3.4-5.5 6.5-5.5s5.7 2 6.5 5.5M16 4.5a3.5 3.5 0 0 1 0 7M18.5 14.8c1.6.8 2.6 2.6 3 5.2" /></>,
  inbox: <><path d="M3 13h5l2 3h4l2-3h5" /><path d="M5 5h14l2 8v6a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1v-6z" /></>,
  upload: <path d="M12 16V4M7 9l5-5 5 5M4 20h16" />,
  info: <><circle cx="12" cy="12" r="9" /><path d="M12 11v5M12 8h.01" /></>,
  check: <path d="M5 12.5l4.5 4.5L19 7.5" />,
  arrow: <path d="M5 12h14M13 6l6 6-6 6" />,
  chev: <path d="M9 6l6 6-6 6" />,
  report: <><path d="M14 3H6a1 1 0 0 0-1 1v16a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V8z" /><path d="M14 3v5h5M9 13h6M9 17h4" /></>,
  play: <path d="M7 5l12 7-12 7z" />,
  key: <><circle cx="8" cy="15" r="4" /><path d="M11 12l9-9M17 6l3 3" /></>,
  tools: <path d="M14.7 6.3a4 4 0 0 0-5.4 5.4L3 18l3 3 6.3-6.3a4 4 0 0 0 5.4-5.4l-2.6 2.6-2.4-.6-.6-2.4z" />,
};

export function Icon({ name, size = 18, stroke = 1.8 }: { name: keyof typeof PATHS | string; size?: number; stroke?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={stroke}
      strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{PATHS[name]}</svg>
  );
}

export function Ring({ value, size = 132, width = 11, children }: { value: number | null; size?: number; width?: number; children?: ReactNode }) {
  const r = (size - width) / 2 - 1;
  const c = 2 * Math.PI * r;
  const v = Math.max(0, Math.min(100, value ?? 0));
  return (
    <div className="ring" style={{ width: size, height: size }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--brand-soft)" strokeWidth={width} />
        {value != null && (
          <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--brand)" strokeWidth={width} strokeLinecap="round"
            strokeDasharray={`${(v / 100) * c} ${c}`} transform={`rotate(-90 ${size / 2} ${size / 2})`} />
        )}
      </svg>
      <div>{children}</div>
    </div>
  );
}

export function Meter({ value }: { value: number | null }) {
  return <div className="meter"><i style={{ width: `${Math.max(0, Math.min(100, value ?? 0))}%` }} /></div>;
}

export function StatusPill({ a, view }: { a: Assessment; view: "teacher" | "student" }) {
  const s = STATUS_LABEL[a.status] ?? { teacher: a.status, student: a.status, tone: "wait" as const };
  return <span className={`pill ${s.tone}`}>{s[view]}</span>;
}

export function CategoryCard({ g }: { g: GroupScore }) {
  const lv = g.score != null ? level(g.score) : null;
  return (
    <div className="card cat">
      <span style={{ fontSize: 13.5, fontWeight: 700 }}>{g.label}</span>
      <span className="big tnum">{g.score != null ? `${g.score}%` : "–"}</span>
      <Meter value={g.score} />
      {lv ? <span className={`tag ${lv.tone}`}>{lv.text}</span> : <span className="tag na">Not assessed</span>}
      <span className="muted" style={{ fontSize: 12, lineHeight: 1.35 }}>
        {g.score != null ? g.blurb : "Not measured for this song yet"}
      </span>
    </div>
  );
}

export function SongThumb({ seed }: { seed: string }) {
  let h = 0;
  for (const ch of seed) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  const bars = [0, 1, 2, 3].map((i) => 10 + ((h >> (i * 4)) % 16));
  return <div className="thumb">{bars.map((b, i) => <i key={i} style={{ height: b }} />)}</div>;
}

export function Loading({ what = "Loading" }: { what?: string }) {
  return <div className="empty">{what}…</div>;
}

export function ErrorNote({ error, onRetry }: { error: string; onRetry?: () => void }) {
  return (
    <div className="note-banner bad">
      <Icon name="info" />
      <span style={{ flex: 1 }}>{error}</span>
      {onRetry && <button className="linkbtn" onClick={onRetry}>Retry</button>}
    </div>
  );
}
