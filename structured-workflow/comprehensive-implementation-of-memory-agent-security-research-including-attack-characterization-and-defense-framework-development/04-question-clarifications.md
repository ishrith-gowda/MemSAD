# 04-question-clarifications: ambiguities, assumptions, and clarifications

## identified ambiguities

### attack implementation details
- exact hyperparameters and training procedures for agentpoison backdoor attacks
- specific prompt templates and injection mechanisms for minja attacks
- precise timing and context for injecmem single-interaction injections
- compatibility requirements between attack implementations and memory system apis

### watermarking algorithm parameters
- optimal delta and gamma values for unigram-watermark embedding
- permute-and-flip decoder configuration for different model sizes
- multi-bit watermarking schemes for metadata embedding
- detection thresholds for false positive minimization

### memory system integration
- exact api endpoints and authentication for external memory systems
- data format compatibility between different memory representations
- performance overhead tolerances for wrapper implementations
- version compatibility with external libraries

### evaluation metrics
- precise definitions of asr-r, asr-a, asr-t success rates
- statistical significance thresholds for experimental results
- baseline performance metrics for benign scenarios
- cross-system comparison methodologies

## formulated questions

### implementation specifics
1. what are the exact hyperparameters used in the original agentpoison paper for backdoor training?
2. how should minja attacks handle different memory system query interfaces?
3. what are the optimal watermark detection thresholds for different memory types?
4. how to handle api rate limits and authentication for external memory services?

### research scope
1. should we prioritize certain attack types over others for initial implementation?
2. what level of optimization is required for production-level performance?
3. how to balance research reproducibility with implementation efficiency?

## documented assumptions

### technical assumptions
- python 3.10+ environment with all required packages available
- external memory system apis are accessible and stable
- sufficient computational resources for attack training and evaluation
- all external libraries are compatible with the chosen python version

### research assumptions
- attack implementations follow the methodologies described in respective papers
- watermarking provides adequate provenance tracking without excessive overhead
- evaluation metrics capture meaningful security properties
- results will be reproducible across different system configurations

### implementation assumptions
- modular design allows independent development of attacks and defenses
- configuration-driven approach enables parameter sweeps
- logging infrastructure provides adequate debugging information
- testing framework covers critical functionality

## clarifications

### resolved ambiguities
- project timeline: january-june 2026, 15-20 hours/week
- target venues: neurips 2026, acm ccs 2026 cycle 2
- memory systems: mem0, a-mem, memgpt/letta
- attacks: agentpoison, minja, injecmem
- defenses: watermarking-based provenance verification

### user input needed
- no critical decisions require immediate user input
- implementation can proceed with documented assumptions
- questions can be resolved through literature review and experimentation
- scope and priorities are sufficiently defined

## summary
the identified ambiguities are primarily technical implementation details that can be resolved through literature review and iterative development. no blocking questions require user input at this stage. the documented assumptions provide a solid foundation for proceeding with the implementation plan.