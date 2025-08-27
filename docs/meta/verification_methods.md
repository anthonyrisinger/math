# Verification Methods

## Computational Verification Standards

### High-Precision Peak Finding
- Resolution: 50,000 sample points
- Range: d ∈ [0.1, 15.0]
- Method: Numerical optimization of measure functions
- Validation: Cross-checked with analytical derivatives where available

### Numerical Stability Testing
- Gamma function stability via `gammaln_safe()` implementation
- Edge case handling for d=0, d→∞
- Warning system for negative/large dimensions

### Test Suite Coverage
- 109 passing tests across core functionality
- Property-based testing for mathematical correctness
- Integration tests for CLI and visualization

## Theoretical Verification Standards

### Verified Claims
- Must be computationally reproducible
- Must match established mathematical literature where applicable
- Must include precision estimates and error bounds

### Conjectural Claims
- Clearly labeled as unverified
- Include research questions and testable predictions
- Separate from computational results

### Documentation Standards
- Source code references for all computational claims
- Version control for mathematical results
- Clear separation between computation and interpretation