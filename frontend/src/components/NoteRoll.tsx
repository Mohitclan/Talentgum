import { useState, type MouseEvent } from "react";
import { KIND_LABEL, isBlack, pitchName, type Kind, type Roll, type RollNote } from "../model";

/** Piano roll of the performance: outlines are the notes the music asks for, filled bars what was heard. */
export function NoteRoll({ roll, from, to, rowH = 12, showKeys = true, highlight, filter = "all" }: {
  roll: Roll;
  from?: number;
  to?: number;
  rowH?: number;
  showKeys?: boolean;
  highlight?: [number, number] | null; // time span to shade
  filter?: Kind | "all";
}) {
  const [hover, setHover] = useState<{ r: RollNote; x: number; y: number } | null>(null);
  const t0 = from ?? roll.t0;
  const t1 = Math.max(to ?? roll.t1, t0 + 0.5);
  const pitches: number[] = [];
  for (let p = roll.hi; p >= roll.lo; p--) pitches.push(p);
  const height = pitches.length * rowH;
  const x = (t: number) => ((t - t0) / (t1 - t0)) * 100;
  const y = (p: number) => (roll.hi - Math.max(roll.lo, Math.min(roll.hi, p))) * rowH;
  const w = Math.max(0.35, (roll.step * 0.75 / (t1 - t0)) * 100);
  const visible = roll.notes.filter((r) => r.t >= t0 - roll.step && r.t <= t1);
  const shown = (r: RollNote) => filter === "all" || r.kind === filter;

  return (
    <div>
      <div className="roll">
        {showKeys && (
          <div className="roll-keys">
            {pitches.map((p) => (
              <div key={p} className={isBlack(p) ? "black" : ""} style={{ height: rowH }}>{rowH >= 11 && p % 12 === 0 ? pitchName(p) : rowH >= 14 ? pitchName(p) : ""}</div>
            ))}
          </div>
        )}
        <div className="roll-body" style={{ height }} onMouseLeave={() => setHover(null)}>
          {pitches.map((p, i) => isBlack(p) && <div key={p} className="lane" style={{ top: i * rowH, height: rowH }} />)}
          {highlight && <div className="zone" style={{ left: `${x(highlight[0])}%`, width: `${x(highlight[1]) - x(highlight[0]) + w}%` }} />}
          {roll.measures.filter((m) => m.t >= t0 && m.t <= t1).map((m) => (
            <div key={m.n} className="bar" style={{ left: `${x(m.t) - 0.4}%` }} />
          ))}
          {visible.map((r, i) => {
            const dim = shown(r) ? 1 : 0.15;
            const pad = Math.max(1, rowH * 0.12);
            const hit = { onMouseMove: (e: MouseEvent) => setHover({ r, x: e.clientX, y: e.clientY }) };
            return (
              <div key={i} style={{ opacity: dim }}>
                {r.expected != null && r.kind !== "ok" && r.kind !== "timing" && r.kind !== "unsure" && (
                  <div className={`n ${r.kind === "missed" ? "missed hit" : "exp"}`} {...(r.kind === "missed" ? hit : {})}
                    style={{ left: `${x(r.t)}%`, width: `${w}%`, top: y(r.expected) + pad, height: rowH - pad * 2 }} />
                )}
                {r.played != null && (
                  <div className={`n ${r.kind} hit`} {...hit}
                    style={{ left: `${x(r.t)}%`, width: `${w}%`, top: y(r.played) + pad, height: rowH - pad * 2 }} />
                )}
              </div>
            );
          })}
        </div>
      </div>
      <div className="roll" style={{ borderTop: 0, borderRadius: "0 0 8px 8px", marginTop: -1 }}>
        {showKeys && <div style={{ width: 46, flex: "none", borderRight: "1px solid var(--line)", background: "var(--paper)" }} />}
        <div className="roll-axis" style={{ flex: 1, borderTop: 0 }}>
          {roll.measures.filter((m) => m.t >= t0 && m.t <= t1).map((m, i, all) => (
            (all.length < 24 || m.n % 2 === 1) && <span key={m.n} style={{ left: `${x(m.t) - 0.4}%` }}>{i === 0 ? `Bar ${m.n}` : m.n}</span>
          ))}
        </div>
      </div>
      {hover && <NoteTip r={hover.r} x={hover.x} y={hover.y} />}
    </div>
  );
}

function NoteTip({ r, x, y }: { r: RollNote; x: number; y: number }) {
  const n = r.note;
  const where = n.measure != null ? `Bar ${n.measure}, beat ${n.beat_in_measure}` : `At ${r.t.toFixed(1)} s`;
  let what: string = KIND_LABEL[r.kind];
  if (r.kind === "wrong" && n.ref_pitch != null && n.perf_pitch != null) what = `Played ${pitchName(n.perf_pitch)} instead of ${pitchName(n.ref_pitch)}`;
  else if (r.kind === "missed" && n.ref_pitch != null) what = `${pitchName(n.ref_pitch)} was not heard`;
  else if (r.kind === "extra" && n.perf_pitch != null) what = `Extra ${pitchName(n.perf_pitch)}`;
  else if (n.ref_pitch != null) what = `${pitchName(n.ref_pitch)} · ${KIND_LABEL[r.kind].toLowerCase()}`;
  const dev = n.onset_dev_ms != null ? `${n.onset_dev_ms > 0 ? "late" : "early"} by ${Math.abs(Math.round(n.onset_dev_ms))} ms` : null;
  return (
    <div className="tip" style={{ left: x + 14, top: y + 14 }}>
      <b>{where}</b><br />{what}{dev && <><br />{dev}</>}
    </div>
  );
}

export function RollLegend({ expected = false }: { expected?: boolean }) {
  return (
    <div className="legend">
      {expected && <span><i className="sw exp" />Expected</span>}
      <span><i className="sw ok" />Played correctly</span>
      <span><i className="sw timing" />Timing off</span>
      <span><i className="sw wrong" />Wrong note</span>
      <span><i className="sw missed" />Missed</span>
      <span><i className="sw extra" />Extra</span>
      <span><i className="sw unsure" />Not sure</span>
      <span><i className="sw replayed" />Played again</span>
    </div>
  );
}
