import { useEffect, useState } from "react";

type Theme = "system" | "light" | "dark";
function preference(): Theme {
  try {
    const value = localStorage.getItem("platform-theme");
    if (value === "light" || value === "dark") return value;
  } catch {
    /* Browser storage can be disabled. */
  }
  return "system";
}
function apply(theme: Theme) {
  const dark =
    theme === "dark" ||
    (theme === "system" && matchMedia("(prefers-color-scheme: dark)").matches);
  document.documentElement.classList.toggle("dark", dark);
  document.documentElement.style.colorScheme = dark ? "dark" : "light";
}
apply(preference());

export function ThemePicker() {
  const [theme, setTheme] = useState<Theme>(preference);
  useEffect(() => {
    apply(theme);
    const query = matchMedia("(prefers-color-scheme: dark)");
    const update = () => apply(theme);
    query.addEventListener("change", update);
    return () => query.removeEventListener("change", update);
  }, [theme]);
  return (
    <label className="theme-control">
      Appearance
      <select
        aria-label="Appearance"
        value={theme}
        onChange={(e) => {
          const value = e.target.value as Theme;
          setTheme(value);
          try {
            localStorage.setItem("platform-theme", value);
          } catch {
            /* Optional persistence. */
          }
        }}
      >
        <option value="system">System</option>
        <option value="light">Light</option>
        <option value="dark">Dark</option>
      </select>
    </label>
  );
}
