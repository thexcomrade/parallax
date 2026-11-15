# PARALLAX Design System: Forensic Signal Bench

## 1. Vision & Architectural Philosophy
The PARALLAX interface is not a marketing dashboard or a generic SaaS application. It is a precision lab instrument — a **Forensic Signal Bench** for multimodal digital evidence verification. Every pixel serves to communicate evidentiary signal, confidence thresholds, and cross-modal telemetry.

There is zero decorative fluff. If an element does not carry signal, telemetry, or forensic provenance, it does not exist.

---

## 2. Palette & Contrast Matrix
The color system reflects strict signal hierarchy: **80% Midnight Black (canvas), 15% Electric Cobalt (structural/video), <=5% Acid Lime (signal/audio/alert-positive), with Alert Red reserved solely for malicious synthesis (FAKE).**

### 2.1 CSS Custom Properties (`tokens.css`)
```css
:root {
  /* Surface Scale (80% Total Surface) */
  --black-950: #04060B; /* Primary viewport canvas */
  --black-900: #080B12; /* Bento cell surface */
  --black-800: #0F141E; /* Raised cards, inputs, overlays */
  --black-700: #161D2B; /* Hover states, active controls */

  /* Structural Hairlines */
  --line: #1B2231;        /* 1px cell borders, dividers */
  --line-strong: #2A3550; /* Active cell borders, crop marks */

  /* Primary Accent: Electric Cobalt (15% Total Surface) */
  --cobalt-600: #1438BF;
  --cobalt-500: #1F4DFF; /* Focus rings, video stream, active navigation */
  --cobalt-400: #547AFF;
  --cobalt-300: #8BA3FF; /* Accessible small text (AA on black >= 4.5:1) */

  /* Signal Accent: Acid Lime (<= 5% Total Surface - Rare Signal) */
  --lime-400: #C8FF2E; /* Primary CTA, AUTHENTIC verdict, audio signal */
  --lime-300: #D8FF5E;
  --lime-ink: #0B1200; /* High-contrast text on acid lime */

  /* Threat / Discrepancy Accent (Functional only) */
  --alert-500: #FF4D5E; /* FAKE verdict, critical anomaly threshold */
  --alert-bg: rgba(255, 77, 94, 0.12);

  /* Typography Text Scale */
  --text-100: #EEF2FA; /* Primary headers and values */
  --text-300: #A9B3C7; /* Body copy, descriptions */
  --text-500: #6B778F; /* Micro labels, metadata, inactive icons */

  /* Radii & Layout */
  --radius-sm: 4px;
  --radius-md: 8px;
  --radius-cell: 10px;
  --border-hairline: 1px solid var(--line);
}
```

### 2.2 Contrast & Accessibility Guardrails
- **WCAG AA Compliance**: All text elements meet a minimum 4.5:1 contrast ratio against their immediate background.
- **Cobalt Rule**: `--cobalt-500` (#1F4DFF) has a ~3.4:1 contrast on `#04060B`. It is strictly prohibited for body text or small labels. For small text, use `--cobalt-300` (#8BA3FF) which achieves > 5.5:1.
- **Multimodal Redundancy**: Anomaly indicators and `FAKE` verdicts must **never** rely on color alone. The `FAKE` verdict chip incorporates a dual-diagonal 45° hazard-stripe pattern and distinct iconography to guarantee full accessibility for color-vision deficiency.

---

## 3. Typography Hierarchy
Typography is self-hosted via Fontsource to ensure zero third-party font request leakage, zero telemetry tracking, and immediate offline rendering.

| Role | Font Family | Variable Axes / Weights | Usage |
|---|---|---|---|
| **Display** | `Anybody` | Variable width (60% to 120%), 700/800 | Header metrics, verdict readouts, hero typography |
| **Body** | `Hanken Grotesk` | 400, 500, 600 | General descriptions, modal copy, telemetry notes |
| **Data / Telemetry** | `Martian Mono` | 400, 500, 700 | Timecodes, hashes, tensor dims, channel IDs (`CH-V`, `SYNC`) |

### Scale Constants
- **Display Big**: `clamp(2.5rem, 5vw, 4.5rem)` with letter-spacing `-0.03em`.
- **Display Medium**: `clamp(1.75rem, 3vw, 2.5rem)` with letter-spacing `-0.02em`.
- **Instrument Heading**: `1.125rem` (18px) `Hanken Grotesk`, weight 600.
- **Body Regular**: `0.9375rem` (15px/22px) `Hanken Grotesk`, weight 400.
- **Data Readout**: `0.8125rem` (13px) `Martian Mono`, tabular figures.
- **Micro Channel Label**: `0.6875rem` (11px) `Martian Mono`, uppercase, letter-spacing `0.08em`.

---

## 4. Bento Grid Architecture & Crop Marks

### 4.1 Grid Dimensions
- **Canvas Baseline**: 8px baseline rhythm.
- **Columns**: 12-column CSS Grid desktop, 6-column tablet, 1-column mobile.
- **Gap**: Fixed 12px horizontal and vertical gutters.
- **Container Queries**: Every `BentoCell` establishes a container query context (`container-type: inline-size`), allowing its internal telemetry components to reflow based on individual cell width rather than global viewport size.

### 4.2 Signature Detail: The Forensic Crop Mark
Every `BentoCell` is bordered by a 1px hairline (`var(--line)`) and features 4 corner crop marks. Each crop mark is an L-shaped corner indicator rendered via `::before` and `::after` pseudo-elements using `var(--line-strong)`.
- Length: 8px on both horizontal and vertical arms.
- Offset: Aligned with the outer boundary of the 10px rounded corner.
- Channel Label: Top-left micro label (e.g., `CH-V // 01`, `DWF-FUSION`, `SPECTROGRAM-L0`).

---

## 5. Motion Vocabulary & Physics

All UI animations use hardware-accelerated transforms and opacity only. No layout shifts or geometry reflow animations.

### 5.1 Physics Constants
- **Standard Easing**: `cubic-bezier(0.2, 0.8, 0.2, 1)`
- **Spring Physics (Fusion Balance)**: Stiffness 280, damping 24.
- **Durations**:
  - Micro-interactions (hover, focus): 120 ms
  - Panel transitions, popovers: 240 ms
  - Scanline wipes & reveals: 320 ms
  - Complex modal / stage build-up: 400 ms

### 5.2 Forensic Motion Primitives
1. **SignalStage Hero**: Procedural point-cloud face wireframe scanned by a 1px acid-lime scanline; audio waveform morphs between harmonic and jittery synthesis.
2. **Scanline Wipe**: 240 ms horizontal or vertical scanline curtain that passes over cells upon data load.
3. **Contact Sheet Frame Develop**: Extracted video frames transition from dark silhouette to normalized color in progressive 80 ms staggered sequences.
4. **Verdict Reveal**: Font width expands dynamically (`font-stretch: 60% -> 120%`) over 300 ms. Authentic triggers one crisp lime pulse. Fake triggers an immediate 300 ms RGB aberration glitch before settling.
5. **Linked Timecode Cursor**: Scrubbing anywhere on the frame strip, spectrogram, or audio waveform propagates a single continuous synchronized time-cursor across all three views.

### 5.3 Reduced Motion & Power Efficiency
- Every animation respects `prefers-reduced-motion: reduce`. Motion defaults to instantaneous state changes or subtle crossfades.
- All 2D Canvas rendering loops (`SignalStage`, `WaveformCanvas`) listen to document visibility changes and **pause execution immediately** when the tab or container is hidden.

---

## 6. Forbidden "AI-Template" Patterns (Zero Tolerance)

The following design tropes are strictly forbidden in PARALLAX:
- ❌ **No purple/pink/violet gradients** or generic multi-color hero blobs.
- ❌ **No glassmorphism, blur backdrops, or frosty translucent cards**.
- ❌ **No floating blurred spheres or ambient gradient lighting balls**.
- ❌ **No emoji in UI controls, status badges, or system notifications**.
- ❌ **No Inter, Roboto, Poppins, or Space Grotesk** as primary brand typefaces.
- ❌ **No generic centered hero with three identical feature cards below**.
- ❌ **No conversational marketing copy** ("Welcome back 👋", "Empowering trust with next-gen AI").
- ❌ **No identical rounded-2xl bubbly cards**.
- ❌ **No generic icon-in-a-colored-circle feature rows**.
- ❌ **No lorem ipsum or dummy text**. Every piece of data and copy reads as precise technical instrumentation.

---

## 7. Component Contract & Atomic Tokens

Components must strictly consume CSS tokens defined in `tokens.css`.
- **Button**:
  - `primary`: Background `var(--lime-400)`, text `var(--lime-ink)`, hover `var(--lime-300)`, focus `var(--cobalt-500)`.
  - `secondary`: Hairline border `var(--line-strong)`, background `var(--black-800)`, text `var(--text-100)`.
  - `ghost`: Transparent, text `var(--text-300)`, hover `var(--black-800)`.
- **Field**:
  - Background `var(--black-800)`, border `var(--line)`, focus border `var(--cobalt-500)`, mono hint in `var(--text-500)`.
- **Badge**:
  - `authentic`: Acid lime fill `rgba(200, 255, 46, 0.15)`, text `var(--lime-400)`, border `var(--lime-400)`.
  - `fake`: Alert red fill `var(--alert-bg)`, text `var(--alert-500)`, border `var(--alert-500)`, hazard-stripe texture.
  - `inconclusive`: Cobalt-300 fill `rgba(139, 163, 255, 0.12)`, text `var(--cobalt-300)`, border `var(--cobalt-400)`.

---

## 8. Layout Shell, Fluid 1760px Grid & Panel Variants (Master Spec §6)

### 8.1 Desktop Grid Scaling (≥ 1280px to 1920px)
- **Max Width**: `max-w-[1760px]` fluid canvas (`mx-auto w-full`). Do not constrain desktop layouts to a 1280px narrow column.
- **Sidebar Rail**: 240–260px expanded navigation (`CommandRail`).
- **Header Status Strip**: 48–56px high instrument telemetry header (`StatusStrip`).
- **Surface Elevation Layers**:
  - `Canvas`: Deep navy base (`#04060B` / `#060911`) with subtle atmospheric blue radial glow (`radial-gradient(circle at 50% 0%, rgba(31, 77, 255, 0.08), transparent 70%)`).
  - `Surface`: Default panel background (`#080B12` / `#0B111D`).
  - `Surface Elevated`: Interactive cards, dropdowns, inputs (`#0F141E` / `#10192A`).
  - `Hairlines`: Subtle borders `rgba(151, 174, 210, 0.14)` and focused borders `rgba(151, 174, 210, 0.26)`.

### 8.2 Distinct Modality Channel Tokens
- **Visual / Video Channel**: Cobalt / Electric Blue (`--color-ch-visual: #3B82F6; --ch-visual: #3B82F6;`).
- **Audio / Acoustic Channel**: Cyan (`--color-ch-audio: #06B6D4; --ch-audio: #06B6D4;`).
- **Action / Success**: Acid Lime (`--color-lime-400: #C8FF2E;`), used sparingly for primary CTA and verified/success state.
- **Threat / Discrepancy**: Alert Red (`--color-alert-500: #FF4D5E;`).
- **Warning / Pending**: Amber (`--color-warning: #F59E0B;`).

### 8.3 Reusable Panel Variants
1. **Telemetry Cell**: Corner crop marks, uppercase mono header, micro status badge.
2. **Hero Workspace**: Elevated surface with subtle radial gradient, primary action button, model status provenance chip.
3. **Metric Card**: Large tabular display number (`Anybody`), uppercase label (`Hanken Grotesk`), subtext telemetry (`Martian Mono`).
4. **Evidence Workbench**: Split dual-channel display comparing synchronized facial frames (Cobalt accent) and Mel-spectrogram (Cyan accent).

