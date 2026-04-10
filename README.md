# latent.alpha

Research landing page for a hierarchical AI agent architecture exploring autonomous trading in Indian financial markets.

## Quick start

This is a single-file static page — no build step required.

```bash
# Option 1: Open directly
open index.html

# Option 2: Local server (Python)
python3 -m http.server 3000
# → http://localhost:3000

# Option 3: Local server (Node)
npx serve .
# → http://localhost:3000
```

## Deploy to Vercel

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy (from this directory)
vercel

# Production deploy
vercel --prod
```

Vercel will detect the static site automatically. No framework configuration needed — it serves `index.html` at the root.

### Alternative: Vercel dashboard

1. Push this folder to a GitHub/GitLab repo
2. Go to [vercel.com/new](https://vercel.com/new)
3. Import the repo — Vercel auto-detects the static site
4. Deploy

## Project structure

```
.
├── index.html   # Complete landing page (HTML + CSS + JS, single file)
└── README.md    # This file
```

## What's included

- **Sticky navbar** with smooth-scroll anchor links
- **Hero** with research badge, abstract block (serif-styled), and metadata row
- **Preliminary findings** — three metric cards with Sharpe ratios and drawdown data
- **Architecture** — 2×2 grid with layer tags and tech pills
- **Research scope** — honest "is / isn't / goal / open questions" grid
- **Email capture** — logs to console (wire up to your backend or Mailchimp/Resend)
- **Legal disclaimer** — full regulatory language in footer
- Scroll-triggered fade-in animations (IntersectionObserver)
- Fully responsive down to 375px

## Customisation

**Email capture**: The form currently logs to `console.log`. To connect a real backend, replace the submit handler in the `<script>` block at the bottom of `index.html`.

**Fonts**: The page loads Inter from Google Fonts. The abstract uses Georgia (system serif). To self-host fonts, download Inter from [rsms.me/inter](https://rsms.me/inter) and update the `@font-face` declarations.

**Colors**: All colors are defined as CSS custom properties in `:root` at the top of the `<style>` block.

## License

MIT
