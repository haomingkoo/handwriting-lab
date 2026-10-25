# Handwriting Lab design

Extends the shared kooexperience design core (`~/.agent-memory/design/DESIGN-core.md`). Where this file is silent, the core applies.

## Audience and job
Portfolio visitors who want to try a machine learning model in ten seconds: draw a digit, see the guess and how sure the model is.

## Mood
A sketchbook page: paper, black ink, one bold answer. Light theme, because the canvas is white paper and the drawing is ink.

## Tokens (`src/mnist_fastapi/web/site.css`)
- Paper `#f6f6f3`, surface `#ffffff`, sunken `#efefeb`, line `#dcdcd6`, control border `#8b8b84`.
- Ink `#141414`, muted `#55554f` (7.5:1 on white, 6.9:1 on paper). The accent is the ink itself: primary button, active tab, top bar, the drawn stroke.
- Danger `#a32118` on `#fbecea`, only for request errors.
- Type: Atkinson Hyperlegible 400/700, self-hosted. Designed by the Braille Institute so every character is unambiguous (slashed zero, distinct 1 and l), which suits a page about telling digits apart. Sizes 14, 16, 20, 32, plus the 120px answer numeral (88px under 420px). The confusion table drops to 13px under 480px so all ten columns fit.
- Radius 6 controls, 10 cards, 16 panels. No shadows.

## Signature move
The answer: one large ink numeral, a plain sentence ("Very sure: 99.8%"), the runner-up, and ten probability bars where only the guess is solid ink. Beside it, the 28 x 28 image the model actually saw, pixelated on purpose.

## Rules for this app
- The canvas comes first. The model guesses when the pen lifts (650 ms debounce); the Guess button repeats it for keyboard and touch users.
- Never show API paths, file names, latency, devices or raw ISO dates. Dates read "25 Feb 2026". Percentages to one decimal.
- States: empty ("Your guess appears here"), no ink, thinking, result, error with retry (rate limit, server, offline), test results loading and failed.
- Accuracy is one sentence of fact, a shaded confusion table, the commonest mistakes and a per-digit table with plain definitions.
- How it works is five plain steps. The developer pipeline stays folded under "Training pipeline, for developers".
- Legal pages share the header, footer and stylesheet.

## Dials
ENERGY 2 / RHYTHM 1 / MOTION 1. One authored moment: the numeral rises in over 320 ms and the bars grow; both are off under reduced motion.
