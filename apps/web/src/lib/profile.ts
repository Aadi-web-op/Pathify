import type { Board, Goal, Stage, StudentProfile, Stream } from "./api";

export const STAGES: { value: Stage; label: string; detail: string }[] = [
  { value: "school_9_10", label: "Class 9 or 10", detail: "Choosing a stream for Class 11" },
  { value: "school_11_12", label: "Class 11 or 12", detail: "Choosing exams and a degree" },
  { value: "college", label: "In college", detail: "Getting ready for a first role" },
  { value: "graduate", label: "Graduated", detail: "Finding or changing direction" },
];

export const STREAMS: { value: Stream; label: string; detail: string }[] = [
  { value: "science_pcm", label: "Science (PCM)", detail: "Physics, Chemistry, Maths" },
  { value: "science_pcb", label: "Science (PCB)", detail: "Physics, Chemistry, Biology" },
  { value: "science_pcmb", label: "Science (PCM)B", detail: "All four" },
  { value: "commerce", label: "Commerce", detail: "Accounts, Business, Economics" },
  { value: "humanities", label: "Humanities / Arts", detail: "History, Political Science, Psychology…" },
  { value: "vocational", label: "Vocational", detail: "Skill-based subjects" },
  { value: "undecided", label: "Not decided", detail: "Still choosing" },
];

export const BOARDS: { value: Board; label: string }[] = [
  { value: "cbse", label: "CBSE" },
  { value: "icse", label: "ICSE / ISC" },
  { value: "state", label: "State board" },
  { value: "ib", label: "IB" },
  { value: "other", label: "Other" },
];

export const GOALS: { value: Goal; label: string; detail: string }[] = [
  { value: "choose_stream", label: "Choose a stream", detail: "Science, Commerce or Humanities after Class 10" },
  { value: "choose_exams_degree", label: "Pick exams and a degree", detail: "JEE, NEET, CUET, CLAT, design and more" },
  { value: "first_job", label: "Land a first job or internship", detail: "Skills, projects and applications" },
  { value: "switch_field", label: "Switch to a different field", detail: "Change course or career direction" },
  { value: "explore", label: "Just explore", detail: "See what's out there" },
];

export const SUGGESTED_GOALS: Record<Stage, Goal[]> = {
  school_9_10: ["choose_stream"],
  school_11_12: ["choose_exams_degree"],
  college: ["first_job"],
  graduate: ["first_job"],
};

export const DEGREE_SUGGESTIONS = [
  "B.Tech / B.E.",
  "B.Sc",
  "B.Com",
  "BA",
  "BBA",
  "BCA",
  "MBBS",
  "B.Des",
  "B.Arch",
  "LLB / BA LLB",
  "B.Pharm",
  "Diploma / Polytechnic",
  "M.Tech",
  "M.Sc",
  "MBA",
  "MA",
];

export const INTEREST_SUGGESTIONS = [
  "Coding",
  "Biology",
  "Drawing",
  "Writing",
  "Maths puzzles",
  "Business",
  "Music",
  "Sports",
  "Helping people",
  "Debating",
  "Space",
  "Gaming",
];

export const INDIAN_STATES = [
  "Andaman and Nicobar Islands",
  "Andhra Pradesh",
  "Arunachal Pradesh",
  "Assam",
  "Bihar",
  "Chandigarh",
  "Chhattisgarh",
  "Dadra and Nagar Haveli and Daman and Diu",
  "Delhi",
  "Goa",
  "Gujarat",
  "Haryana",
  "Himachal Pradesh",
  "Jammu and Kashmir",
  "Jharkhand",
  "Karnataka",
  "Kerala",
  "Ladakh",
  "Lakshadweep",
  "Madhya Pradesh",
  "Maharashtra",
  "Manipur",
  "Meghalaya",
  "Mizoram",
  "Nagaland",
  "Odisha",
  "Puducherry",
  "Punjab",
  "Rajasthan",
  "Sikkim",
  "Tamil Nadu",
  "Telangana",
  "Tripura",
  "Uttar Pradesh",
  "Uttarakhand",
  "West Bengal",
];

export const isSchool = (stage: Stage) => stage === "school_9_10" || stage === "school_11_12";

export function stageLabel(profile: StudentProfile): string {
  if (isSchool(profile.stage)) {
    const stream = STREAMS.find((s) => s.value === profile.stream);
    return `Class ${profile.grade}${stream && profile.stream !== "undecided" ? `, ${stream.label}` : ""}`;
  }
  const year = profile.year_of_study ? `, year ${profile.year_of_study}` : "";
  return profile.stage === "graduate" ? `Graduate, ${profile.degree}` : `${profile.degree}${year}`;
}

export type Station = { label: string; detail: string };

/** The decisions ahead of a student, starting where they are now. Used by the route on Home. */
export function routeAhead(profile: StudentProfile): Station[] {
  const all: Station[] = [
    { label: "Class 9–10", detail: "Board exams" },
    { label: "Choose a stream", detail: "Science, Commerce, Humanities" },
    { label: "Class 11–12", detail: "Stream subjects" },
    { label: "Exams and degree", detail: "JEE, NEET, CUET, CLAT…" },
    { label: "College", detail: "Degree and skills" },
    { label: "First role", detail: "Internship or job" },
    { label: "Grow or switch", detail: "Specialise, study further, change field" },
  ];
  const position = { school_9_10: 0, school_11_12: 2, college: 4, graduate: 5 }[profile.stage];
  return all.slice(position);
}
