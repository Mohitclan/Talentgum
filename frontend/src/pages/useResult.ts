import { useMemo } from "react";
import { api, type Assessment, type Note, type Review } from "../api";
import { buildRoll, buildSections } from "../model";
import { useLoad } from "../state";

const FINISHED = ["completed", "review_required", "reviewed"];

/** One assessment with its notes and teacher reviews, plus the derived roll and sections. */
export function useResult(id: string) {
  const data = useLoad(async () => {
    const a: Assessment = await api.assessment(id);
    const [notes, reviews]: [Note[], Review[]] = FINISHED.includes(a.status)
      ? await Promise.all([api.notes(id), api.reviews(id)])
      : [[], []];
    return { a, notes, reviews };
  }, [id]);
  const roll = useMemo(() => (data.data ? buildRoll(data.data.notes) : null), [data.data]);
  const sections = useMemo(() => (roll ? buildSections(roll) : []), [roll]);
  const review = data.data?.reviews[data.data.reviews.length - 1] ?? null;
  return { ...data, roll, sections, review };
}
