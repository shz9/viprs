// Durations are stored in seconds; display whole elapsed minutes and seconds.
export function formatDuration(value) {
  if (value === null || value === undefined || value === "" || !Number.isFinite(Number(value)) || Number(value) < 0) return "Not recorded";
  const seconds = Math.floor(Number(value));
  return `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
}
