# Hypothesis: Execution-Guided Detection of Semantic Code Perturbations

## Problem Statement

Static analysis (AST features) and embedding-based methods (CodeBERT) fail
to detect semantic perturbations in AI-generated code. Our previous work
(Papers 1 and 2) established:

- Boundary inversion: 0% recall via static analysis, 90% false approval
  rate by AI reviewers
- Import aliasing: 7% recall via static analysis
- Structural perturbations: 100% recall (trivially detected)

## Core Hypothesis

Execution traces — records of what code *actually does* when run on
diverse inputs — contain sufficient signal to detect semantic
perturbations that preserve syntactic structure but change behavior.

Specifically:

1. Boundary inversion produces different branch decisions when test
   inputs fall near the flipped boundary
2. Import aliasing produces different function outputs when the aliased
   import is invoked
3. These behavioral differences are detectable via trace comparison
   even when the code looks identical syntactically

## Research Questions

RQ1: Can execution traces detect semantic perturbations with >80% recall
    while maintaining >90% precision?

RQ2: Which trace features (branch decisions, output values, memory
    states) carry the most detection signal?

RQ3: How many test inputs are needed for reliable detection? Is there a
    minimum threshold?

RQ4: Does execution-guided detection generalize across perturbation
    types and code styles?

## Expected Contributions

1. A novel execution-guided detection framework for adversarial code
2. A C-based sandbox executor for safe, fast code execution
3. A property-based test input generator for Python functions
4. Empirical validation on 550 samples (reusing Papers 1 & 2 dataset)
5. Comparison against static and embedding baselines from prior work
6. Theoretical analysis of when execution traces can distinguish
   semantically different code

## Success Criteria

- Boundary inversion recall: >80% (currently 0%)
- Import aliasing recall: >70% (currently 7%)
- Overall detection accuracy: >90%
- Statistical significance: p < 0.01 vs baselines
- 12-page paper with theoretical grounding
- Reproducible from scratch on CPU only
