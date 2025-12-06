---
description: "Task list for generating book content"
---

**Constitution Reminder**: All tasks must adhere to the project constitution, focusing on `/book_source`, using correct content formats, and respecting Docusaurus configurations.

# Tasks: Generate Book Content

**Input**: Design documents from `specs/1-generate-book-content/`
**Prerequisites**: spec.md (required for user stories)

## Phase 1: Content Scaffolding

**Purpose**: Prepare the `docs` directory and create the chapter files.

- [ ] T001 [US1] Delete all existing files in the `/book_source/docs/` directory.
- [ ] T002 [P] [US1] Create an empty file at `/book_source/docs/01-intro.md`.
- [ ] T003 [P] [US2] Create an empty file at `/book_source/docs/02-rag-systems.md`.
- [ ] T004 [P] [US3] Create an empty file at `/book_source/docs/03-agents.md`.
- [ ] T005 [P] [US4] Create an empty file at `/book_source/docs/04-backend.md`.

---

## Phase 2: Content Generation

**Purpose**: Populate the chapter files with detailed content. These tasks are sequential as they are part of a single book.

- [ ] T006 [US1] Generate at least 500 words of content for `book_source/docs/01-intro.md` covering "Introduction to AI Engineering", Generative AI, and the shift from MLOps to LLMOps.
- [ ] T007 [US2] Generate at least 500 words of content for `book_source/docs/02-rag-systems.md` covering "Building RAG Systems", Vector Databases (mentioning Qdrant), and embedding and chunking strategies.
- [ ] T008 [US3] Generate at least 500 words of content for `book_source/docs/03-agents.md` covering "Autonomous Agents", the difference between chatbots and agents, and using the OpenAI Agents SDK.
- [ ] T009 [US4] Generate at least 500 words of content for `book_source/docs/04-backend.md` covering "Modern AI Backend", setting up FastAPI with Neon DB, and authentication flows.

---

## Phase 3: Docusaurus Configuration

**Purpose**: Update the site configuration to reflect the new book.

- [ ] T010 [US5] Update the `docusaurus.config.ts` file to set the `title` to "The GenAI Engineering Handbook".
- [ ] T011 [US5] Update the `docusaurus.config.ts` file to set the `tagline` to "Your guide to building modern AI applications.".

---

## Dependencies & Execution Order

- **Phase 1 (Content Scaffolding)** must be completed before Phase 2.
- **Phase 2 (Content Generation)** tasks should be completed sequentially.
- **Phase 3 (Docusaurus Configuration)** can be done in parallel with Phase 1 or 2, but is grouped here for logical separation.
