export type Theme = "light" | "dark" | "system";

const KEY = "pathify-theme";

/** Runs before first paint (inlined in <head>) so there's no flash of the wrong theme. */
export const themeBootScript = `(function(){try{var t=localStorage.getItem("${KEY}");if(t==="light"||t==="dark"){document.documentElement.dataset.theme=t}}catch(e){}})();`;

export function applyTheme(theme: Theme) {
  const root = document.documentElement;
  if (theme === "system") delete root.dataset.theme;
  else root.dataset.theme = theme;
  try {
    localStorage.setItem(KEY, theme);
  } catch {
    // Storage can be unavailable in private windows; the theme still applies for this visit.
  }
}

export function storedTheme(): Theme {
  try {
    const value = localStorage.getItem(KEY);
    return value === "light" || value === "dark" ? value : "system";
  } catch {
    return "system";
  }
}
