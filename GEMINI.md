# Gemini CLI Workspace Configuration

This file provides foundational mandates and context optimizations for Gemini CLI in this workspace.

## 1. Project Context
- **Domain:** Smart Airport Flow and Person Re-Identification (ReID).
- **Primary Languages:** Python (assumed for ReID models and data processing).

## 2. Agent Optimizations & Logic
- **Read-Only Datasets:** Strictly treat `awesome-reid-dataset/` and any other image or dataset directories as **READ-ONLY**. Do not attempt to read, edit, or search through `.png`, `.jpg`, `.jpeg`, or binary files.
- **Search Efficiency:** When using search tools (`grep_search`, `glob`), always explicitly exclude dataset directories (e.g., `exclude_pattern: "awesome-reid-dataset/**"`) to conserve context window and reduce search time.
- **Code Modifications:** When writing or modifying code, prioritize performance-oriented libraries common in computer vision (e.g., PyTorch, NumPy, OpenCV). 
- **Tool Delegation:** For bulk operations or complex refactoring, proactively use the `generalist` sub-agent to minimize context bloat in the main session.

## 3. Workflow Directives
- **Verification First:** Before implementing fixes for reported bugs, empirically verify the bug by running relevant scripts or tests.
- **Concise Reporting:** When describing model architectures or dataset statistics, use succinct tables or bullet points. Avoid conversational filler.