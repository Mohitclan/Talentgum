import { useEffect, useMemo } from "react";
import { api } from "../api";
import { ErrorNote, Icon } from "../components/ui";
import { REJECT_REASON } from "../model";
import { useApp, useLoad } from "../state";

const STEPS = [
  { label: "Uploaded", sub: "Recording received" },
  { label: "In the queue", sub: "Waiting for a free worker" },
  { label: "Listening", sub: "Turning the audio into notes" },
  { label: "Comparing", sub: "Matching notes with the music" },
  { label: "Done", sub: "Score and feedback ready" },
];

const KID_STEPS = [
  { label: "Sent!", sub: "We got your song" },
  { label: "Waiting", sub: "Your turn is coming" },
  { label: "Listening", sub: "Hearing every note" },
  { label: "Checking", sub: "Comparing with the music" },
  { label: "Done!", sub: "Off to your teacher" },
];

export function Processing({ id }: { id: string }) {
  const { go, songTitle, me } = useApp();
  const a = useLoad(() => api.assessment(id), [id]);
  const s = a.data?.status;
  const finished = s != null && !["queued", "processing"].includes(s);

  useEffect(() => {
    if (finished) return;
    const t = setInterval(a.reload, 2500);
    return () => clearInterval(t);
  }, [finished, a.reload]);

  // the API reports queued -> processing -> finished; the middle steps share "processing"
  const at = s === "queued" ? 1 : s === "processing" ? 2 : finished ? 5 : 0;
  const bad = s === "rejected" || s === "failed";

  const dots = useMemo(() => Array.from({ length: 34 }, (_, i) => ({
    l: (i / 34) * 100, t: 14 + ((i * 37) % 9) * 10, w: 1.6 + ((i * 13) % 3),
  })), []);

  const kid = me.role === "student";
  const steps = kid ? KID_STEPS : STEPS;
  const title = kid
    ? (bad ? "Oops, we couldn't hear it well" : finished ? "All done!" : "We're listening to your song…")
    : (bad ? "We couldn't assess this recording" : finished ? "Analysis complete" : "Analysing your performance");
  const sub = kid
    ? (bad ? "Try again in a quiet room, with the phone close to the keyboard." : finished ? "Your teacher will look at it and send you your stars soon." : "This takes about as long as your song.")
    : (bad ? (REJECT_REASON[a.data?.error_code ?? ""] ?? "Please try recording again.")
      : finished ? "The result is ready to check." : "This usually takes about as long as the recording itself.");

  return (
    <div className={`col ${kid ? "kid" : ""}`} style={{ alignItems: "center", gap: 28, paddingTop: 24 }}>
      <div className="col" style={{ alignItems: "center", textAlign: "center", gap: 8 }}>
        <span className="muted" style={{ fontSize: 13.5, fontWeight: 600 }}>{a.data ? songTitle(a.data.song_id) : " "}</span>
        <h1 className={kid ? "kid-title" : undefined} style={{ fontSize: 34 }}>{title}</h1>
        <span className="sub">{sub}</span>
      </div>
      {a.error && <ErrorNote error={a.error} onRetry={a.reload} />}
      <div className="card" style={{ width: "100%", maxWidth: 880, padding: 0, overflow: "hidden" }}>
        <div className="scan">
          {dots.map((d, i) => <i key={i} style={{ left: `${d.l}%`, top: d.t, width: `${d.w}%`, opacity: finished ? 0.8 : 0.25 }} />)}
          {!finished && <div className="beam" />}
        </div>
        <div className="steps6">
          {steps.map((st, i) => {
            const state = i < at || (finished && !bad) ? "done" : i === at && !bad ? "now" : "";
            return (
              <div className="pstep" key={st.label}>
                <div className="row" style={{ gap: 8 }}>
                  <div className={`pdot ${state}`}>{state === "done" ? <Icon name="check" size={13} stroke={3} /> : i + 1}</div>
                  {i < steps.length - 1 && <div className={`pline ${i < at ? "done" : ""}`} />}
                </div>
                <b style={{ fontSize: 13.5 }}>{st.label}</b>
                <span className="muted" style={{ fontSize: 12, lineHeight: 1.4 }}>{st.sub}</span>
              </div>
            );
          })}
        </div>
      </div>
      <div className="col" style={{ alignItems: "center", gap: 14 }}>
        {finished && !bad && (
          me.role === "teacher"
            ? <button className="btn primary" onClick={() => go("results", id)}>View the result</button>
            : <button className="btn primary" onClick={() => go("home")}>Back to dashboard</button>
        )}
        {bad && <button className="btn primary" onClick={() => go("new")}>Try another recording</button>}
        {!finished && <span className="muted small">You can leave this page. Your result will appear in Recent Assessments.</span>}
      </div>
    </div>
  );
}
