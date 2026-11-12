// Turns raw RAS results into what the screens show: category scores, sections, the note roll and feedback.
import type { Assessment, Criterion, Note, Review, Status } from "./api";

export const GROUPS: { id: string; label: string; blurb: string }[] = [
  { id: "note", label: "Notes", blurb: "Right notes, in the right order" },
  { id: "timing", label: "Timing & rhythm", blurb: "Steady beat and note timing" },
  { id: "chord", label: "Chords", blurb: "All chord notes together" },
  { id: "performance", label: "Expression", blurb: "Dynamics, touch and flow" },
];

export interface GroupScore { id: string; label: string; blurb: string; score: number | null; assessed: number; total: number }

/** Weighted mean of the assessed criteria in each group; null when nothing in the group could be measured. */
export function groupScores(a: Assessment, criteria: Criterion[]): GroupScore[] {
  const groupOf = new Map(criteria.map((c) => [c.id, c.group]));
  const crit = a.scoring?.criteria ?? {};
  return GROUPS.map((g) => {
    let w = 0, s = 0, assessed = 0, total = 0;
    for (const [id, c] of Object.entries(crit)) {
      if ((groupOf.get(id) ?? "other") !== g.id) continue;
      total++;
      if (c.status === "assessed" && c.score != null) { w += c.weight; s += c.weight * c.score; assessed++; }
    }
    return { ...g, score: w > 0 ? Math.round(s / w) : null, assessed, total };
  });
}

export function criterionLabel(id: string, criteria: Criterion[]): string {
  return criteria.find((c) => c.id === id)?.label ?? id.replace(/_/g, " ");
}

/** Encouraging level words, never harsh. */
export function level(score: number): { text: string; tone: "great" | "good" | "grow" } {
  if (score >= 85) return { text: "Excellent", tone: "great" };
  if (score >= 70) return { text: "Good", tone: "good" };
  if (score >= 50) return { text: "Getting there", tone: "grow" };
  return { text: "Keep practising", tone: "grow" };
}

export function headline(score: number): string {
  if (score >= 85) return "Great performance!";
  if (score >= 70) return "Well done!";
  if (score >= 50) return "Good effort!";
  return "Nice try, keep going!";
}

// ---- who may see what (teacher-first: a student sees a result once a teacher has reviewed it)

export function visibleToStudent(a: Assessment): boolean {
  return a.status === "reviewed";
}

export function displayScore(a: Assessment, review?: Review | null): number | null {
  const s = review?.overall_score ?? a.total_score;
  return s == null ? null : Math.round(s);
}

export const STATUS_LABEL: Record<Status, { teacher: string; student: string; tone: "ok" | "wait" | "warn" | "bad" }> = {
  queued: { teacher: "Queued", student: "Waiting", tone: "wait" },
  processing: { teacher: "Analysing", student: "Analysing", tone: "wait" },
  completed: { teacher: "To check", student: "With your teacher", tone: "wait" },
  review_required: { teacher: "Needs review", student: "With your teacher", tone: "warn" },
  reviewed: { teacher: "Shared", student: "Ready", tone: "ok" },
  rejected: { teacher: "Not assessable", student: "Please record again", tone: "bad" },
  failed: { teacher: "Failed", student: "Something went wrong", tone: "bad" },
};

export const REVIEW_REASON: Record<string, string> = {
  audio_quality: "Recording quality is low",
  chords_or_second_hand_heard: "Chords or a second hand were heard",
  few_matched_notes: "Few notes matched the song",
  few_notes_matched: "Few notes matched the song",
  low_score_coverage: "Only part of the song was heard",
  many_error_claims: "Many mistakes reported, worth a check",
  many_unexplained_notes: "Many notes not in the song",
  many_wrong_steps: "Melody steps often differ",
  notes_uncertain: "Some notes are uncertain",
  performance_transposed: "Played in a different key",
  reference_mismatch: "May be a different song",
  start_timestamp_suspect: "Start of the recording is unclear",
  transcribers_disagree: "The two transcribers disagree",
};

export const REJECT_REASON: Record<string, string> = {
  too_quiet: "The recording is too quiet.",
  too_short: "The recording is too short.",
  too_long: "The recording is longer than 5 minutes.",
  mostly_silence: "We could not hear enough playing.",
  clipping: "The recording is distorted (too loud).",
  recording_quality_low: "The recording quality is too low.",
  decode_failed: "The audio file could not be opened.",
  alignment_failed: "We could not match the playing to the song.",
  recording_deleted: "The recording was deleted.",
};

// ---- dates and names

export function fmtDate(iso: string): string {
  return new Date(iso).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
}

export function initials(name: string): string {
  const parts = name.replace(/[_-]+/g, " ").trim().split(/\s+/);
  return ((parts[0]?.[0] ?? "?") + (parts[1]?.[0] ?? "")).toUpperCase();
}

const NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"];
export const pitchName = (p: number) => `${NAMES[p % 12]}${Math.floor(p / 12) - 1}`;
export const isBlack = (p: number) => [1, 3, 6, 8, 10].includes(p % 12);

// ---- note classification for display

export type Kind = "ok" | "timing" | "wrong" | "missed" | "extra" | "unsure";

export const TIMING_MS = 120; // the rubric's onset window

export function kindOf(n: Note): Kind {
  switch (n.status) {
    case "correct":
    case "replayed":
      return n.onset_dev_ms != null && Math.abs(n.onset_dev_ms) > TIMING_MS ? "timing" : "ok";
    case "wrong_pitch":
    case "wrong_octave":
      return "wrong";
    case "missing":
      return "missed";
    case "extra":
      return "extra";
    default:
      return "unsure"; // uncertain, unconfirmed_extra: the second model did not back the claim
  }
}

export const KIND_LABEL: Record<Kind, string> = {
  ok: "Played correctly", timing: "Timing off", wrong: "Wrong note", missed: "Missed", extra: "Extra", unsure: "Not sure",
};

// ---- the note roll: every note placed on the performance's own timeline

export interface RollNote {
  kind: Kind;
  t: number; // seconds
  expected: number | null; // pitch the score asks for
  played: number | null; // pitch heard
  measure: number | null;
  note: Note;
}

export interface Roll {
  notes: RollNote[];
  t0: number;
  t1: number;
  lo: number;
  hi: number;
  measures: { n: number; t: number }[];
  outOfRange: number;
  step: number; // typical seconds between score notes
}

/** Score notes the student did not play have no performed time. Place them between their played neighbours. */
export function buildRoll(notes: Note[]): Roll {
  const ref = notes.filter((n) => n.ref_index != null).sort((a, b) => a.ref_index! - b.ref_index!);
  const known = ref.filter((n) => n.perf_onset_s != null).map((n) => [n.ref_index!, n.perf_onset_s!] as const);
  const gaps = known.slice(1).map((k, i) => (k[1] - known[i][1]) / Math.max(1, k[0] - known[i][0])).filter((g) => g > 0);
  const step = gaps.length ? gaps.sort((a, b) => a - b)[Math.floor(gaps.length / 2)] : 0.5;

  const timeOf = (idx: number): number => {
    let before: readonly [number, number] | undefined, after: readonly [number, number] | undefined;
    for (const k of known) {
      if (k[0] <= idx) before = k;
      if (k[0] >= idx) { after = k; break; }
    }
    if (before && after && after[0] !== before[0]) return before[1] + ((idx - before[0]) / (after[0] - before[0])) * (after[1] - before[1]);
    if (before) return before[1] + (idx - before[0]) * step;
    if (after) return Math.max(0, after[1] - (after[0] - idx) * step);
    return idx * step;
  };

  const out: RollNote[] = notes.map((n) => {
    const kind = kindOf(n);
    const t = n.perf_onset_s ?? (n.ref_index != null ? timeOf(n.ref_index) : 0);
    return { kind, t, expected: n.ref_pitch, played: n.status === "missing" ? null : n.perf_pitch, measure: n.measure, note: n };
  });

  // pitch range follows the score; extras far outside it (often harmonics) are pinned to the edge
  const refP = ref.map((n) => n.ref_pitch!).filter((p) => p != null);
  const lo = (refP.length ? Math.min(...refP) : 60) - 3;
  const hi = (refP.length ? Math.max(...refP) : 72) + 3;
  let outOfRange = 0;
  for (const r of out) if (r.played != null && (r.played < lo || r.played > hi)) outOfRange++;

  const measures: { n: number; t: number }[] = [];
  const seen = new Map<number, number>();
  for (const r of out) if (r.measure != null && r.note.ref_index != null) {
    const prev = seen.get(r.measure);
    if (prev === undefined || r.t < prev) seen.set(r.measure, r.t);
  }
  [...seen.entries()].sort((a, b) => a[0] - b[0]).forEach(([n, t]) => measures.push({ n, t }));

  const ts = out.map((r) => r.t);
  const t0 = ts.length ? Math.min(...ts) : 0;
  const t1 = ts.length ? Math.max(...ts) + step : 1;
  return { notes: out.sort((a, b) => a.t - b.t), t0: Math.max(0, t0 - step / 2), t1, lo, hi, measures, outOfRange, step };
}

// ---- sections: four bars each, scored by the share of score notes played right

export interface Section { label: string; m0: number; m1: number; score: number | null; t0: number; t1: number; issues: number }

export function buildSections(roll: Roll, barsPer = 4): Section[] {
  const maxM = Math.max(0, ...roll.notes.map((r) => r.measure ?? 0));
  const secs: Section[] = [];
  for (let m0 = 1; m0 <= maxM; m0 += barsPer) {
    const m1 = Math.min(maxM, m0 + barsPer - 1);
    const inSec = roll.notes.filter((r) => r.measure != null && r.measure >= m0 && r.measure <= m1);
    const ok = inSec.filter((r) => r.kind === "ok" || r.kind === "unsure").length;
    const timing = inSec.filter((r) => r.kind === "timing").length;
    const ts = inSec.map((r) => r.t);
    const t0 = ts.length ? Math.min(...ts) : 0;
    const t1 = ts.length ? Math.max(...ts) : 0;
    const extras = roll.notes.filter((r) => r.kind === "extra" && r.t >= t0 && r.t <= t1).length;
    secs.push({
      label: `Bars ${m0}–${m1}`, m0, m1, t0, t1,
      score: inSec.length ? Math.round(((ok + timing * 0.5) / inSec.length) * 100) : null,
      issues: inSec.filter((r) => r.kind !== "ok").length + extras,
    });
  }
  return secs;
}

// ---- feedback: RAS writes one opening line, then specific tips that name a bar

export interface Tip { text: string; bar: number | null }

export function splitFeedback(lines: string[] = []): { opening: string | null; tips: Tip[] } {
  const [opening, ...rest] = lines;
  return {
    opening: opening ?? null,
    tips: rest.map((text) => {
      const m = text.match(/\bbar (\d+)/i);
      return { text, bar: m ? Number(m[1]) : null };
    }),
  };
}

/** What went well: the strongest measured criteria, in plain words. */
export function strengths(a: Assessment, criteria: Criterion[], n = 3): { title: string; detail: string }[] {
  const crit = Object.entries(a.scoring?.criteria ?? {})
    .filter(([, c]) => c.status === "assessed" && c.score != null && c.score >= 75)
    .sort((x, y) => (y[1].score! * y[1].weight) - (x[1].score! * x[1].weight));
  return crit.slice(0, n).map(([id, c]) => ({
    title: STRENGTH_TITLE[id] ?? criterionLabel(id, criteria),
    detail: STRENGTH_DETAIL[id] ?? `Scored ${Math.round(c.score!)} out of 100.`,
  }));
}

const STRENGTH_TITLE: Record<string, string> = {
  correct_notes: "Accurate notes", incorrect_notes: "Very few wrong notes", missed_notes: "Played the whole piece",
  extra_notes: "Clean playing", wrong_octave: "Right part of the keyboard", note_order: "Notes in the right order",
  onset_timing: "Notes on time", rhythm_accuracy: "Good rhythm", tempo: "Good speed",
  tempo_consistency: "Steady tempo", timing_deviation: "Precise timing", unintended_pauses: "No unplanned stops",
  fluency: "Smooth and flowing", chord_correctness: "Strong chords", dynamics: "Expressive dynamics",
};

const STRENGTH_DETAIL: Record<string, string> = {
  correct_notes: "Most notes matched the music.", missed_notes: "You kept going through the song.",
  tempo_consistency: "You kept an even beat from start to finish.", unintended_pauses: "You kept the music moving.",
  rhythm_accuracy: "Long and short notes had the right lengths.", fluency: "The piece flowed without hesitations.",
  note_order: "Every phrase came in the right sequence.", wrong_octave: "You played in the right octave.",
};

// ---- levels: each has its own rubric (data/rubrics/talentgum_<level>_v1.json)

export const LEVELS = [
  { id: "beginner", label: "Beginner", rules: "Right hand only · octave not judged" },
  { id: "advanced", label: "Advanced", rules: "Hands as written in the song · octave judged" },
] as const;

export function levelOfRubric(rubricId: string): string | null {
  const m = rubricId.match(/^talentgum-(beginner|advanced)$/);
  return m ? m[1][0].toUpperCase() + m[1].slice(1) : null;
}
