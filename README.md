# Error Correction Codes

## Overview
This repository contains implementations of various error detection and correction algorithms. The code demonstrates how different coding schemes can be used to protect data transmission over noisy channels.

## Included Algorithms
- **Parity Bit**: Simple error detection using a single bit
- **Hamming Code (7,4)**: Can detect up to 2-bit errors and correct 1-bit errors
- **Reed-Solomon Code**: Simplified implementation that works well for burst errors

## Features
- Complete implementation of encoding and decoding functions
- Error simulation with configurable error rates
- Performance analysis and comparison between different coding schemes
- Visualization of results (detection rate, correction rate, overhead, timing)

## Code Structure
The repository includes:
- Core implementations of each error correction algorithm
- Helper classes for Galois Field arithmetic (for Reed-Solomon)
- Testing framework to evaluate and compare coding schemes
- Visualization utilities to present results

## Usage Example
```python
# Basic usage of parity encoding/decoding
message = [1, 0, 1, 0]
encoded = parity_encode(message)
decoded, valid = parity_decode(encoded)

# Basic usage of Hamming code
hamming_encoded = hamming_encode(message)
decoded, error_detected, corrected = hamming_decode(hamming_encoded)

# Performance testing
results = test_error_correction(message_size=4, num_messages=1000)
plot_results(results)
```

## Requirements
- Python 3.6+
- NumPy
- Matplotlib

## Applications
This code demonstrates techniques used in:
- Data storage systems
- Network communications
- Deep space communications
- Any system that needs to protect data integrity in noisy environments
