import { client } from "./client/client.gen";
// Development token stays in memory and is discarded on refresh/sign-out.
let token = "";
export function setToken(value: string) {
  token = value;
}
export function hasToken() {
  return token.length > 0;
}
client.setConfig({ baseUrl: "", auth: () => token });
