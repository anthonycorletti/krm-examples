# Vendored shadcn components

Button, Input, Textarea, and Badge were generated from the official shadcn registry with `bunx --bun shadcn@4.21.0 add button input textarea badge`. The upstream MIT notice is retained in [LICENSE.md](LICENSE.md). Source: https://github.com/shadcn-ui/ui/blob/main/LICENSE.md.

Use bin/web-ui for explicit component updates. It invokes the pinned CLI with Bun, pins the installed dependency versions, updates the lockfile, and performs a frozen install. Registry component source can change independently of the CLI version; review the source diff as well as the lockfile during upgrades. Checked-in component source is the reproducible build input.

components.json uses the stone base color. Application CSS maps shadcn tokens to Tailwind stone colors, sets square corners, and supplies light/dark themes. ThemePicker follows system changes in System mode and persists only the appearance preference.
