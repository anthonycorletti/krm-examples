import { UserManager, WebStorageStateStore } from "oidc-client-ts";
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
    post_logout_redirect_uri: `${location.origin}/connect`,
    scope: "openid platform:read platform:verify",
    automaticSilentRenew: false,
    loadUserInfo: false,
    userStore: new WebStorageStateStore({ store: localStorage }),
    stateStore: new WebStorageStateStore({ store: sessionStorage }),
  });
  manager.events.addAccessTokenExpired(() => {
    setToken("");
    location.assign("/connect");
  });
  return manager;
}
export async function signIn() {
  await (await oidc()).signinRedirect();
}
export async function completeSignIn() {
  const user = await (await oidc()).signinRedirectCallback();
  user.refresh_token = undefined;
  await (await oidc()).storeUser(user);
  setToken(user.access_token);
}
export async function disconnect() {
  const config = await authConfig();
  if (!config.issuer) {
    setToken("");
    return;
  }
  const instance = await oidc();
  const user = await instance.getUser();
  setToken("");
  await instance.removeUser();
  await instance.signoutRedirect({ id_token_hint: user?.id_token });
}
