import { StrictMode, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  createBrowserRouter,
  RouterProvider,
  useNavigate,
  useRouteError,
  useLoaderData,
  redirect,
} from "react-router";
import Console, { action, loader } from "./routes/console";
import { readIdentity } from "./client/sdk.gen";
import { setToken } from "./session";
import "./style.css";
import { authConfig, signIn, completeSignIn } from "./oidc";
import { Button } from "./components/ui/button";
import { Input } from "./components/ui/input";
import { ThemePicker } from "./theme";

function Connect() {
  const [value, setValue] = useState("");
  const [error, setError] = useState("");
  const [pending, setPending] = useState(false);
  const navigate = useNavigate();
  const config = useLoaderData<typeof authConfig>();
  return (
    <div className="connect">
      <div className="eyebrow">
        PLATFORM / {config.environment.toUpperCase()}
      </div>
      <h1>Connect to your platform.</h1>
      <p>
        {config.issuer
          ? "Sign in with your platform identity."
          : "Use the short-lived API token from bin/local-token."}
      </p>
      {config.issuer ? (
        <Button
          className="primary"
          disabled={pending}
          onClick={async () => {
            setPending(true);
            setError("");
            try {
              await signIn();
            } catch {
              setError(
                "Sign-in could not start. Check identity configuration.",
              );
              setPending(false);
            }
          }}
        >
          Sign in
        </Button>
      ) : (
        <form
          onSubmit={async (e) => {
            e.preventDefault();
            setPending(true);
            setError("");
            setToken(value.trim());
            try {
              await readIdentity({ throwOnError: true });
              setValue("");
              await navigate("/");
            } catch {
              setToken("");
              setError(
                "Unable to authenticate. Check the API and token expiry.",
              );
            } finally {
              setPending(false);
            }
          }}
        >
          <label htmlFor="token">Development API token</label>
          <Input
            id="token"
            type="password"
            autoComplete="off"
            value={value}
            onChange={(e) => setValue(e.target.value)}
            required
          />
          <Button className="primary" disabled={pending}>
            {pending ? "Connecting…" : "Connect"}
          </Button>
        </form>
      )}
      <p role="alert" className="error">
        {error}
      </p>
      <small>The token stays in memory and is cleared on page refresh.</small>
      <ThemePicker />
    </div>
  );
}
function ErrorPage() {
  useRouteError();
  return (
    <div className="connect">
      <h1>Could not load platform data.</h1>
      <p>Check the server connection and token permissions.</p>
      <a href="/connect">Reconnect</a>
    </div>
  );
}
const router = createBrowserRouter([
  { path: "/connect", Component: Connect, loader: authConfig },
  {
    path: "/auth/callback",
    loader: async () => {
      await completeSignIn();
      return redirect("/");
    },
    errorElement: <ErrorPage />,
  },
  {
    path: "/",
    loader,
    action,
    Component: Console,
    errorElement: <ErrorPage />,
  },
]);
createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>,
);
