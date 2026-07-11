# EMPOWEREDAI Landing Page

This repository contains the `EMPOWEREDAI` premium workshop landing page.

## Structure

- `index.html` — main landing page
- `dax-flashcards.html` — served at the clean URL `/dax-flashcards` via `vercel.json`
- `why-dashboards-fail.html` — supporting guide page
- `assets/images/` — image assets and brand visuals
- `vercel.json` — clean URL rewrite for `/dax-flashcards` and 301 redirects from legacy URLs
- `.gitignore` — repository ignores for static site development

## Notes

- The site is a static HTML/CSS landing page designed for enterprise workshop positioning, deployed on Vercel.
- `index.html` is the entry point for deployment.
- The DAX Flashcards page is served at `/dax-flashcards` (clean URL). The underlying file is `dax-flashcards.html`; `vercel.json` rewrites the clean path to it and 301-redirects the legacy `dax-flashcards.html` and `dax-flashcards-v3.html` paths.

## Next steps

- Add a remote repository and push the initial commit.
- If you want, the site can be migrated to a lightweight build system later, but it is currently deployable as a plain static site.
