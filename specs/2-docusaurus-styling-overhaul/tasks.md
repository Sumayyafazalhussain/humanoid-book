# Tasks: Docusaurus Styling Overhaul

**Input**: `specs/2-docusaurus-styling-overhaul/spec.md`

## Phase 1: CSS Architecture & Theming

**Purpose**: Establish the foundational CSS architecture and define the robotics-themed design system.

- [ ] T001: Create the main custom CSS file at `book_source/src/css/custom.css`.
- [ ] T002: Update `docusaurus.config.js` to import the new custom CSS file.
- [ ] T003: Define the robotics color palette, typography, spacing, and other design tokens as CSS custom properties in `custom.css`.
- [ ] T004: Apply base styles, including a modern CSS reset and the circuit board background pattern.

## Phase 2: Component Styling

**Purpose**: Apply the new design system to individual Docusaurus components.

- [ ] T005: Style the Navbar with a fixed position, custom gradient, and branding.
- [ ] T006: Style the Sidebar to be collapsible and include module indicators.
- [ ] T007: Style the main DocPage for an enhanced reading experience.
- [ ] T008: Style the Footer with a professional, tech-oriented design.
- [ ] T009: Style Code Blocks to mimic a VS Code-like appearance.
- [ ] T010: Style Tables for professional data presentation.
- [ ] T011: Style Alerts and notifications with a modern look.
- [ ] T012: Style Cards for modules and chapters.

## Phase 3: Interactive Features

**Purpose**: Implement interactive and dynamic user-facing features.

- [ ] T013: Implement a smooth dark/light mode toggle.
- [ ] T014: Implement a CSS-only or minimal JS module progress tracker.
- [ ] T015: Set up the structure for the Monaco Editor-based code execution playground.
- [ ] T016: Set up the structure for Mermaid.js and Three.js interactive diagrams.

## Phase 4: Responsive Design & Final Touches

**Purpose**: Ensure the new design is responsive and add final polishing animations.

- [ ] T017: Add media queries to ensure the design is responsive across mobile, tablet, and desktop devices.
- [ ] T018: Implement subtle animations for interactive elements and 3D transform effects for diagrams.
