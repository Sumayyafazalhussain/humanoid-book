---
id: 002
title: Add Physical AI Course Modules
stage: green
date: 2025-12-07
surface: agent
model: Gemini
feature: 1-generate-book-content
branch: 1-generate-book-content
user: N/A
command: /sp.implement
labels: [implementation, content-generation, docusaurus]
links:
  spec: null
  ticket: null
  adr: null
  pr: null
files:
 - book_source/docs/physical-ai-course/module-1-intro/_category_.json
 - book_source/docs/physical-ai-course/module-1-intro/01-why-physical-ai-matters.md
 - book_source/docs/physical-ai-course/module-1-intro/02-learning-outcomes.md
 - book_source/docs/physical-ai-course/module-1-intro/03-intro-to-physical-ai.md
 - book_source/docs/physical-ai-course/module-2-ros/_category_.json
 - book_source/docs/physical-ai-course/module-2-ros/01-ros2-fundamentals.md
 - book_source/docs/physical-ai-course/module-3-simulation/_category_.json
 - book_source/docs/physical-ai-course/module-3-simulation/01-gazebo-simulation.md
 - book_source/docs/physical-ai-course/module-4-nvidia-isaac/_category_.json
 - book_source/docs/physical-ai-course/module-4-nvidia-isaac/01-nvidia-isaac-platform.md
 - book_source/sidebars.ts
tests:
 - "N/A"
---

## Prompt

Yes, proceed with the implementation on the `1-generate-book-content` branch. 

Apply this implementation plan to add the "Physical AI & Humanoid Robotics" course modules to our Docusaurus project.

Follow the safety precautions strictly:
1. First create backup of existing files
2. Check before creating any new files/folders
3. Preserve all existing content in docs/ folder
4. Only add new modules (module-1 through module-4)

Start with the Safety First steps, then proceed with the full implementation as outlined in the prompt.

## Response snapshot

I have successfully added the "Physical AI & Humanoid Robotics" course modules to your Docusaurus project. I have followed all safety precautions and preserved your existing content.

Here is a summary of the implementation:

### 1. Safety First: Backups Created

*   A backup of your `docs` directory has been created at `book_source/docs_backup`.
*   A backup of your `sidebars.ts` file has been created at `book_source/sidebars.ts.bak`.

### 2. Directory Setup: New Module Folders

I have created the following directory structure for the new course:
```
book_source/docs/
└── physical-ai-course/
    ├── module-1-intro/
    ├── module-2-ros/
    ├── module-3-simulation/
    └── module-4-nvidia-isaac/
```

### 3. File Creation: Course Content Added

I have added the new course content into the module folders, broken down into individual Markdown files.

### 4. Sidebar Integration: `sidebars.ts` Updated

I have updated the `book_source/sidebars.ts` file to explicitly include the new "Physical AI & Humanoid Robotics" course in the sidebar, alongside your existing "The GenAI Engineering Handbook" chapters.

### 5. Verification Steps: Implementation Confirmed

I have verified that all new directories and files have been created correctly. The updated `sidebars.ts` file is also in place.

### 6. Troubleshooting

Here are some common issues you might encounter and how to resolve them:

*   **Sidebar Not Updating:** If the new course modules do not appear in the sidebar, try stopping and restarting the Docusaurus development server.
*   **"Page Not Found" Errors:** If you get a 404 error when clicking on a new module link, ensure that the file names in `sidebars.ts` exactly match the file names in the `docs` directory.
*   **Build Errors:** If the Docusaurus project fails to build, check the console for any errors related to the `sidebars.ts` file. A missing comma or a typo can sometimes cause build failures.

You can now run your Docusaurus project to see the changes. If you need to roll back, you can delete the `physical-ai-course` directory and restore the `sidebars.ts` file from the `.bak` backup.

## Outcome

- ✅ Impact: The "Physical AI & Humanoid Robotics" course has been added to the Docusaurus project.
- 🧪 Tests: N/A
- 📁 Files: 
  - `book_source/docs/physical-ai-course/module-1-intro/_category_.json`
  - `book_source/docs/physical-ai-course/module-1-intro/01-why-physical-ai-matters.md`
  - `book_source/docs/physical-ai-course/module-1-intro/02-learning-outcomes.md`
  - `book_source/docs/physical-ai-course/module-1-intro/03-intro-to-physical-ai.md`
  - `book_source/docs/physical-ai-course/module-2-ros/_category_.json`
  - `book_source/docs/physical-ai-course/module-2-ros/01-ros2-fundamentals.md`
  - `book_source/docs/physical-ai-course/module-3-simulation/_category_.json`
  - `book_source/docs/physical-ai-course/module-3-simulation/01-gazebo-simulation.md`
  - `book_source/docs/physical-ai-course/module-4-nvidia-isaac/_category_.json`
  - `book_source/docs/physical-ai-course/module-4-nvidia-isaac/01-nvidia-isaac-platform.md`
  - `book_source/sidebars.ts`
- 🔁 Next prompts: The user can now build the Docusaurus site to see the new course.
- 🧠 Reflection: The implementation was successful. The user's instructions were followed carefully to ensure no data loss.

## Evaluation notes (flywheel)

- Failure modes observed: None.
- Graders run and results (PASS/FAIL): N/A
- Prompt variant (if applicable): N/A
- Next experiment (smallest change to try): N/A
