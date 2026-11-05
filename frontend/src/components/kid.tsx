// Kid-friendly pieces for the student side: stars instead of scores, a flower per skill, badges, the song path.
import type { Assessment } from "../api";
import type { GroupScore, Roll } from "../model";

/** 1 to 5 stars; never zero, so every attempt earns something. */
export function starsFor(score: number | null): number {
  if (score == null) return 0;
  if (score >= 90) return 5;
  if (score >= 75) return 4;
  if (score >= 60) return 3;
  if (score >= 40) return 2;
  return 1;
}

export const STAR_CHEER = ["", "Nice start!", "Good effort!", "Great job!", "Amazing playing!", "Superstar!"];

export function Star({ on, size = 28 }: { on: boolean; size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 2.6l2.9 5.9 6.5.9-4.7 4.6 1.1 6.5L12 17.4l-5.8 3.1 1.1-6.5L2.6 9.4l6.5-.9z"
        fill={on ? "var(--star)" : "var(--star-off)"} stroke={on ? "var(--star-edge)" : "none"} strokeWidth="1" strokeLinejoin="round" />
    </svg>
  );
}

export function Stars({ n, size = 28, label = true }: { n: number; size?: number; label?: boolean }) {
  return (
    <span className="stars" role={label ? "img" : undefined} aria-label={label ? `${n} out of 5 stars` : undefined}>
      {[1, 2, 3, 4, 5].map((i) => <Star key={i} on={i <= n} size={size} />)}
    </span>
  );
}

// ---- the garden: one flower per skill, growing with the score

export const KID_SKILL: Record<string, { name: string; color: string; hint: string }> = {
  note: { name: "Right notes", color: "#e86fa0", hint: "Playing the notes in the music" },
  timing: { name: "Keeping the beat", color: "#f29a4a", hint: "Notes on time, steady speed" },
  chord: { name: "Chords", color: "#9b7be0", hint: "Notes played together" },
  performance: { name: "Smooth playing", color: "#3fb5a5", hint: "Keeping the music flowing" },
};

export type Stage = 0 | 1 | 2 | 3; // seedling, bud, flower, full bloom
export const STAGE_NAME = ["Sprouting", "Growing", "Blooming", "In full bloom!"];

export function stageFor(score: number): Stage {
  if (score >= 85) return 3;
  if (score >= 65) return 2;
  if (score >= 45) return 1;
  return 0;
}

export function Flower({ stage, color, size = 84 }: { stage: Stage; color: string; size?: number }) {
  const petals = stage === 3 ? 7 : 5;
  const r = stage === 3 ? 11 : 8.5;
  return (
    <svg width={size} height={size * 1.15} viewBox="0 0 80 92" aria-hidden="true" className={stage === 3 ? "sway" : undefined}>
      <ellipse cx="40" cy="88" rx="22" ry="4" fill="#e3dccd" />
      <path d={`M40 88 V${stage === 0 ? 62 : 40}`} stroke="#5aa55a" strokeWidth="3.5" strokeLinecap="round" />
      <path d="M40 76 C30 72 25 64 26 58 C34 60 39 67 40 76Z" fill="#79bf6c" />
      <path d="M40 70 C50 66 55 58 54 52 C46 54 41 61 40 70Z" fill="#8fcf7f" />
      {stage === 1 && <path d="M40 26 C31 34 33 44 40 46 C47 44 49 34 40 26Z" fill={color} />}
      {stage >= 2 && (
        <g transform="translate(40 32)">
          {Array.from({ length: petals }, (_, i) => (
            <ellipse key={i} cx="0" cy={-r} rx={r * 0.62} ry={r} fill={color} opacity={0.92}
              transform={`rotate(${(360 / petals) * i})`} />
          ))}
          <circle r={r * 0.62} fill="#ffd25e" stroke="#f0b53a" strokeWidth="1" />
        </g>
      )}
      {stage === 3 && <g fill="#ffd25e"><circle cx="66" cy="14" r="2.2" /><circle cx="14" cy="20" r="1.8" /><circle cx="70" cy="34" r="1.4" /></g>}
    </svg>
  );
}

export function Garden({ groups, size = 84 }: { groups: GroupScore[]; size?: number }) {
  const shown = groups.filter((g) => g.score != null && KID_SKILL[g.id]);
  return (
    <div className="garden">
      {shown.map((g) => {
        const k = KID_SKILL[g.id];
        const st = stageFor(g.score!);
        return (
          <div key={g.id} className="plant" title={k.hint}>
            <Flower stage={st} color={k.color} size={size} />
            <b>{k.name}</b>
            <span>{STAGE_NAME[st]}</span>
          </div>
        );
      })}
    </div>
  );
}

// ---- badges, from the rubric's criterion scores

const BADGES: { id: string; crit: string; min: number; name: string; desc: string; icon: string }[] = [
  { id: "ninja", crit: "correct_notes", min: 85, name: "Note Ninja", desc: "Almost every note was right", icon: "♪" },
  { id: "beat", crit: "tempo_consistency", min: 80, name: "Steady Beat", desc: "Kept an even speed", icon: "♥" },
  { id: "rhythm", crit: "rhythm_accuracy", min: 75, name: "Rhythm Star", desc: "Long and short notes just right", icon: "★" },
  { id: "nonstop", crit: "unintended_pauses", min: 100, name: "Non-Stop", desc: "Played without stopping", icon: "➜" },
  { id: "finish", crit: "missed_notes", min: 85, name: "Finisher", desc: "Played the whole song", icon: "⚑" },
  { id: "smooth", crit: "fluency", min: 85, name: "Smooth Sailor", desc: "The music kept flowing", icon: "~" },
];

export function badgesFor(a: Assessment) {
  const crit = a.scoring?.criteria ?? {};
  return BADGES.filter((b) => crit[b.crit]?.status === "assessed").map((b) => ({ ...b, earned: (crit[b.crit].score ?? 0) >= b.min }));
}

export function Badge({ b }: { b: ReturnType<typeof badgesFor>[number] }) {
  return (
    <div className={`badge ${b.earned ? "won" : ""}`} title={b.earned ? b.desc : `Keep practising to unlock: ${b.desc.toLowerCase()}`}>
      <span className="badge-coin">{b.earned ? b.icon : "?"}</span>
      <b>{b.name}</b>
      <span>{b.earned ? b.desc : "Not yet, keep going!"}</span>
    </div>
  );
}

// ---- tips in kids' words (RAS writes them for adults: note names, beats)

export interface KidTip { bar: number | null; text: string }

export function kidTip(text: string): KidTip {
  const m = text.match(/\bbar (\d+)/i);
  const bar = m ? Number(m[1]) : null;
  const at = bar ? `In bar ${bar}` : "Somewhere in the song";
  let out = text;
  if (/an octave off/i.test(text)) out = `${at}, one note jumped too high or too low. Find its home on the keyboard!`;
  else if (/as a different note/i.test(text)) out = `${at}, one note sounded different. Play that part slowly and listen.`;
  else if (/was not heard/i.test(text)) out = `${at}, a note went missing. Can you find it in the music?`;
  else if (/a little late/i.test(text)) out = `${at}, a note was a tiny bit late. Clap the beat, then play!`;
  else if (/a little early/i.test(text)) out = `${at}, a note was a tiny bit early. Count the beat in your head.`;
  else if (/started again/i.test(text)) out = `You started again at bar ${bar ?? "?"}. Next time, try to keep going, even after a slip!`;
  else if (/long pause/i.test(text)) out = `${at}, the music stopped for a while. Keep going, even slowly!`;
  else if (/extra notes/i.test(text)) out = "A few extra notes sneaked in. Keep your fingers ready on the right keys!";
  else if (/wrong order/i.test(text)) out = `${at}, two notes swapped places. Play that bit slowly first.`;
  else if (/hesitation/i.test(text)) out = "A few tiny wobbles. Playing slowly helps the music flow.";
  return { bar, text: out };
}

/** Up to n tips, one per bar where possible. */
export function kidTips(lines: string[], n = 3): KidTip[] {
  const seen = new Set<string>();
  const out: KidTip[] = [];
  for (const t of lines.map(kidTip)) {
    const key = t.bar != null ? `b${t.bar}` : t.text;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(t);
    if (out.length >= n) break;
  }
  return out;
}

// ---- the song path: one stepping stone per bar

export type BarMark = "star" | "good" | "practice";

/** Judged on right notes only; timing already shows in the "Keeping the beat" flower. A bar with one slip is still "good". */
export function barMarks(roll: Roll): { m: number; mark: BarMark }[] {
  const byBar = new Map<number, { wrong: number; late: number; all: number }>();
  for (const r of roll.notes) {
    if (r.measure == null || r.note.ref_index == null) continue;
    const b = byBar.get(r.measure) ?? { wrong: 0, late: 0, all: 0 };
    b.all++;
    if (r.kind === "wrong" || r.kind === "missed") b.wrong++;
    else if (r.kind === "timing") b.late++;
    byBar.set(r.measure, b);
  }
  return [...byBar.entries()].sort((x, y) => x[0] - y[0]).map(([m, b]) => ({
    m, mark: b.wrong === 0 && b.late === 0 ? "star" : b.wrong <= 1 && b.wrong / b.all < 0.5 ? "good" : "practice",
  }));
}
