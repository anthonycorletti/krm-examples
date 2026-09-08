import {
  InMemoryWebStorage,
  UserManager,
  WebStorageStateStore,
} from "oidc-client-ts";
import { readAuthConfig } from "./client/sdk.gen";
import { setToken } from "./session";

export const authConfig = () =>
  readAuthConfig({ throwOnError: true }).then((r) => r.data);
let manager: UserManager | undefined;
async function oidc() {
  if (manager) return manager;
  const config = await authConfig();
  if (!config.issuer || !config.client_id)
    throw new Error("OIDC is not configured");
  manager = new UserManager({
    authority: config.issuer,
    client_id: config.client_id,
    redirect_uri: `${location.origin}/auth/callback`,
    response_type: "code",
    scope: "openid profile platform:read platform:verify",
    automaticSilentRenew: false,
    loadUserInfo: false,
    userStore: new WebStorageStateStore({ store: new InMemoryWebStorage() }),
    stateStore: new WebStorageStateStore({ store: sessionStorage }),
  });
  manager.events.addAccessTokenExpired(() => setToken(""));
  return manager;
}
export async function signIn() {
  await (await oidc()).signinRedirect();
}
export async function completeSignIn() {
  const user = await (await oidc()).signinRedirectCallback();
  setToken(user.access_token);
}
export async function disconnect() {
  setToken("");
  if (manager) await manager.removeUser();
}
