import { client } from "./client/client.gen";

const KEY = "platform.access-token";
let expiryTimer: ReturnType<typeof setTimeout> | undefined;
export function tokenExpiry(value: string): number {
  try {
    const payload = JSON.parse(
      atob(value.split(".")[1].replace(/-/g, "+").replace(/_/g, "/")),
    );
    return typeof payload.exp === "number" ? payload.exp * 1000 : 0;
  } catch {
    return 0;
  }
}
export function setToken(value: string) {
  clearTimeout(expiryTimer);
  const remaining = tokenExpiry(value) - Date.now();
  if (value && remaining > 0) {
    localStorage.setItem(KEY, value);
    expiryTimer = setTimeout(
      () => {
        setToken("");
        location.assign("/connect");
      },
      Math.min(remaining, 2147483647),
    );
  } else {
    localStorage.removeItem(KEY);
    for (const key of Object.keys(localStorage)) {
      if (key.startsWith("oidc.user:")) localStorage.removeItem(key);
    }
  }
}
function token() {
  const value = localStorage.getItem(KEY) ?? "";
  return tokenExpiry(value) > Date.now() ? value : "";
}
export function hasToken() {
  return token().length > 0;
}
setToken(localStorage.getItem(KEY) ?? "");
window.addEventListener("storage", (event) => {
  if (event.key === KEY) {
    setToken(event.newValue ?? "");
    if (!hasToken()) location.assign("/connect");
  }
});
client.setConfig({ baseUrl: "", auth: token });
