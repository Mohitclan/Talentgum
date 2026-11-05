import { useState } from "react";
import type { Assessment, Review } from "../api";
import { Badge, badgesFor, barMarks, Garden, kidTips, STAR_CHEER, Stars, starsFor, type BarMark } from "../components/kid";
import { displayScore, groupScores, type Roll } from "../model";
import { useApp } from "../state";

const MARK_LABEL: Record<BarMark, string> = { star: "Perfect!", good: "Nice!", practice: "Practise me" };

/** The student's result: stars, a garden of skills, badges and the song path. No numbers or note names. */
export function KidResults({ a, roll, review }: { a: Assessment; roll: Roll; review: Review | null }) {
  const { go, songTitle, criteria } = useApp();
  const score = displayScore(a, review);
  const stars = starsFor(score);
  const groups = groupScores(a, criteria);
  const badges = badgesFor(a);
  const tips = kidTips((a.feedback ?? []).slice(1));
  const bars = barMarks(roll);
  const [picked, setPicked] = useState<number | null>(null);
  const pickedTip = picked != null ? tips.find((t) => t.bar === picked) : null;
  const pickedMark = bars.find((b) => b.m === picked)?.mark;
  const won = badges.filter((b) => b.earned).length;

  if (score == null) {
    return (
      <div className="kid col" style={{ gap: 20 }}>
        <a className="back" onClick={() => go("home")}>← Home</a>
        <div className="card kid-hero">
          <h1 className="kid-title">Hmm, that sounded like a different song!</h1>
          <p className="kid-sub">We couldn't match your playing to <b>{songTitle(a.song_id)}</b>. Check you picked the right song, then play it again.</p>
          <button className="btn kid-btn" onClick={() => go("new")}>Play again</button>
        </div>
      </div>
    );
  }

  return (
    <div className="kid col" style={{ gap: 22 }}>
      <a className="back" onClick={() => go("home")}>← Home</a>

      <div className="card kid-hero">
        <span className="kid-song">{songTitle(a.song_id)}</span>
        <h1 className="kid-title">{STAR_CHEER[stars]}</h1>
        <Stars n={stars} size={46} />
        <p className="kid-sub">You earned <b>{stars} {stars === 1 ? "star" : "stars"}</b>{won ? ` and ${won} ${won === 1 ? "badge" : "badges"}` : ""}!</p>
        <div className="row" style={{ justifyContent: "center", flexWrap: "wrap" }}>
          <button className="btn kid-btn" onClick={() => go("new")}>Play it again</button>
        </div>
      </div>

      {review?.comment && (
        <div className="kid-note">
          <div className="avatar">{(review.reviewer[0] ?? "T").toUpperCase()}</div>
          <div className="bubble"><b>{review.reviewer} says:</b> {review.comment}</div>
        </div>
      )}

      <div className="card col" style={{ gap: 6 }}>
        <h2 className="kid-h2">Your music garden</h2>
        <span className="muted small">Each flower is a skill. The more you practise, the more it blooms!</span>
        <Garden groups={groups} />
      </div>

      <div className="card col" style={{ gap: 12 }}>
        <h2 className="kid-h2">Your song path</h2>
        <span className="muted small">Every stone is one bar of the song. Tap a stone to see how it went.</span>
        <div className="path">
          {bars.map((b) => (
            <button key={b.m} className={`stone ${b.mark} ${picked === b.m ? "on" : ""}`} onClick={() => setPicked(picked === b.m ? null : b.m)}
              aria-label={`Bar ${b.m}: ${MARK_LABEL[b.mark]}`}>
              <span className="stone-ico">{b.mark === "star" ? "★" : b.mark === "good" ? "✓" : "♪"}</span>
              <span className="stone-n">{b.m}</span>
            </button>
          ))}
        </div>
        <div className="legend kid-legend">
          <span><i className="dot star" />Perfect</span><span><i className="dot good" />Nice</span><span><i className="dot practice" />Practise me</span>
        </div>
        {picked != null && (
          <div className="bubble big">
            <b>Bar {picked}: {MARK_LABEL[pickedMark ?? "good"]}</b>{" "}
            {pickedTip ? pickedTip.text.replace(/^In bar \d+, (\w)/, (_, c: string) => c.toUpperCase()) : pickedMark === "star" ? "Every note was spot on. Brilliant!" : pickedMark === "good" ? "Nearly perfect. Well done!" : "Play this bar slowly a few times, then join it to the next one."}
          </div>
        )}
      </div>

      {badges.length > 0 && (
        <div className="card col" style={{ gap: 12 }}>
          <h2 className="kid-h2">Badges</h2>
          <div className="badges">{badges.map((b) => <Badge key={b.id} b={b} />)}</div>
        </div>
      )}

      <div className="card col" style={{ gap: 12 }}>
        <h2 className="kid-h2">Let's practise</h2>
        {tips.length === 0 ? <p className="kid-sub" style={{ margin: 0 }}>Nothing to fix. You're a star!</p> : tips.map((t, i) => (
          <div key={i} className="kid-tip" onClick={() => t.bar && setPicked(t.bar)} style={{ cursor: t.bar ? "pointer" : undefined }}>
            <span className="kid-tip-n">{i + 1}</span>
            <span>{t.text}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
