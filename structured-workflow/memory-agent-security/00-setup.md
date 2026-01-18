# 00-setup: Workflow Initialization and Environment Setup

## working directory
current working directory: /Volumes/usb drive/memory-agent-security

## project context
this is a comprehensive research project on memory agent security, investigating adversarial robustness of memory-augmented llm agents (mem0, a-mem, memgpt/letta) and developing a provenance-aware defense framework using dr. xuandong zhao's watermarking techniques. the project targets publication at neurips 2026 or acm ccs 2026, with a 6-month timeline from january 2026.

## output pattern
confirmed: structured-workflow/{task-name}/ pattern will be used for all workflow outputs. for this project, the task-name is 'memory-agent-security', so outputs will be organized in structured-workflow/memory-agent-security/

## tools available
available tools include:
- file system operations (create, read, edit files and directories)
- terminal execution for running commands
- search tools (grep, semantic search, file search)
- notebook tools for jupyter notebooks
- git tools for version control
- various specialized tools for python, testing, etc.
- structured workflow tools for research phases
- subagent tools for complex tasks

## environment verification
- python environment: clean_env virtual environment active
- workspace: /Volumes/usb drive/memory-agent-security
- external dependencies: mem0, amem, memgpt cloned in external/
- context documents: comprehensive documentation available in docs/context/ and docs/research/

## next steps
proceed to audit_inventory phase to analyze the codebase and catalog changes needed for attack implementation and defense framework development.