import numpy as np
import matplotlib.pyplot as plt
import random
import time
from typing import List, Tuple

def bit_flip(bit):
    """Flips a bit from 0 to 1 or from 1 to 0"""
    return 1 - bit

def introduce_errors(data: List[int], error_rate: float) -> List[int]:
    """Introduces random errors in the data according to the given error rate"""
    corrupted_data = data.copy()
    for i in range(len(corrupted_data)):
        if random.random() < error_rate:
            corrupted_data[i] = bit_flip(corrupted_data[i])
    return corrupted_data

# Implementation of a simple parity bit scheme
def parity_encode(data: List[int]) -> List[int]:
    """Encodes data by adding a parity bit at the end"""
    parity_bit = sum(data) % 2
    return data + [parity_bit]

def parity_check(data: List[int]) -> bool:
    """Checks if there is an error using the parity bit"""
    return sum(data) % 2 == 0

def parity_decode(data: List[int]) -> Tuple[List[int], bool]:
    """Decodes data with a parity bit and returns the original data and the presence of an error"""
    is_valid = parity_check(data)
    return data[:-1], is_valid

# Implementation of Hamming code (7,4)
def hamming_encode(data: List[int]) -> List[int]:
    """
    Encodes 4 bits of data into a Hamming code (7,4)
    Positions of parity bits: 1, 2, 4
    Positions of data bits: 3, 5, 6, 7
    """
    if len(data) != 4:
        raise ValueError("Hamming (7,4) requires exactly 4 bits of input data")
    
    encoded = [0] * 7
    
    # Placing data in the corresponding positions
    encoded[2] = data[0]
    encoded[4] = data[1]
    encoded[5] = data[2]
    encoded[6] = data[3]
    
    # Calculating parity bits
    # p1 (position 1): checks positions 3, 5, 7 (indices 2, 4, 6)
    encoded[0] = (encoded[2] + encoded[4] + encoded[6]) % 2
    
    # p2 (position 2): checks positions 3, 6, 7 (indices 2, 5, 6)
    encoded[1] = (encoded[2] + encoded[5] + encoded[6]) % 2
    
    # p4 (position 4): checks positions 5, 6, 7 (indices 4, 5, 6)
    encoded[3] = (encoded[4] + encoded[5] + encoded[6]) % 2
    
    return encoded

def hamming_decode(data: List[int]) -> Tuple[List[int], bool, List[int]]:
    """
    Decodes a Hamming code (7,4) and corrects a single error if one exists
    Returns: (original_data, was_there_an_error, corrected_data)
    """
    if len(data) != 7:
        raise ValueError("Hamming (7,4) requires exactly 7 bits for decoding")
    
    received = data.copy()
    
    # Calculating the syndrome
    s1 = (received[0] + received[2] + received[4] + received[6]) % 2
    s2 = (received[1] + received[2] + received[5] + received[6]) % 2
    s3 = (received[3] + received[4] + received[5] + received[6]) % 2
    
    syndrome = s1 + 2*s2 + 4*s3
    
    # Extracting the data
    decoded = [received[2], received[4], received[5], received[6]]
    
    # Error checking
    error_detected = syndrome != 0
    corrected_data = received.copy()
    
    # Correcting the error if one exists
    if error_detected and syndrome <= 7:
        # The syndrome indicates the position of the error (from 1 to 7)
        corrected_data[syndrome - 1] = bit_flip(corrected_data[syndrome - 1])
        
    # Extracting the data after correction
    corrected_decoded = [corrected_data[2], corrected_data[4], corrected_data[5], corrected_data[6]]
    
    return decoded, error_detected, corrected_decoded

# Implementation of Reed-Solomon code (simplified version)
class GF256:
    """Helper class for arithmetic operations in the Galois Field GF(2^8)"""
    
    @staticmethod
    def add(a, b):
        """Addition in GF(2^8) is XOR operation"""
        return a ^ b
    
    @staticmethod
    def mul(a, b):
        """Multiplication in GF(2^8)"""
        p = 0
        for i in range(8):
            if b & 1:
                p ^= a
            hi_bit_set = a & 0x80
            a <<= 1
            if hi_bit_set:
                a ^= 0x1D  # Polynomial modulus: x^8 + x^4 + x^3 + x^2 + 1
            b >>= 1
        return p & 0xFF

def rs_encode(data, n, k):
    """
    Simplified version of Reed-Solomon encoding
    data: input data (k bytes)
    n: total size of the encoded word
    k: size of the original data
    """
    if len(data) != k:
        raise ValueError(f"Input data must be of length {k}")
    
    # Adding parity
    encoded = data.copy()
    for _ in range(n - k):
        encoded.append(0)
    
    # Simplified version - adding XOR of all bytes as parity
    for i in range(k):
        for j in range(n - k):
            encoded[k + j] = GF256.add(encoded[k + j], GF256.mul(data[i], (i + 1) ** (j + 1)))
    
    return encoded

def rs_decode(data, n, k):
    """
    Simplified version of Reed-Solomon decoding
    Returns: (data, error_detected)
    """
    # In this simplified version, we only check if the XOR matches
    received = data.copy()
    original = received[:k]
    
    # Calculating the expected parity symbols
    expected_parity = [0] * (n - k)
    for i in range(k):
        for j in range(n - k):
            expected_parity[j] = GF256.add(expected_parity[j], GF256.mul(original[i], (i + 1) ** (j + 1)))
    
    # Checking parity
    actual_parity = received[k:]
    error_detected = False
    
    for i in range(n - k):
        if expected_parity[i] != actual_parity[i]:
            error_detected = True
            break
    
    return original, error_detected

# Testing and analysis
def test_error_correction(message_size=4, num_messages=1000, error_rates=[0.01, 0.05, 0.1, 0.2]):
    """Tests and compares different error correction schemes"""
    results = {}
    
    for error_rate in error_rates:
        results[error_rate] = {
            'parity': {'detected': 0, 'corrected': 0, 'undetected': 0, 'overhead': 0, 'time': 0},
            'hamming': {'detected': 0, 'corrected': 0, 'undetected': 0, 'overhead': 0, 'time': 0},
            'reed_solomon': {'detected': 0, 'corrected': 0, 'undetected': 0, 'overhead': 0, 'time': 0}
        }
        
        for _ in range(num_messages):
            # Generating a random message
            original_message = [random.randint(0, 1) for _ in range(message_size)]
            
            # Testing parity bit
            start_time = time.time()
            parity_encoded = parity_encode(original_message)
            parity_overhead = len(parity_encoded) / len(original_message)
            
            # Introducing errors
            parity_corrupted = introduce_errors(parity_encoded, error_rate)
            
            # Checking if the error is detected
            parity_decoded, parity_valid = parity_decode(parity_corrupted)
            error_introduced = parity_corrupted != parity_encoded
            
            if error_introduced and parity_valid:
                results[error_rate]['parity']['undetected'] += 1
            elif error_introduced and not parity_valid:
                results[error_rate]['parity']['detected'] += 1
            
            results[error_rate]['parity']['time'] += time.time() - start_time
            results[error_rate]['parity']['overhead'] += parity_overhead
            
            # Testing Hamming code
            if message_size == 4:  # For Hamming (7,4) we need exactly 4 bits
                start_time = time.time()
                hamming_encoded = hamming_encode(original_message)
                hamming_overhead = len(hamming_encoded) / len(original_message)
                
                # Introducing errors
                hamming_corrupted = introduce_errors(hamming_encoded, error_rate)
                
                # Decoding
                hamming_decoded, error_detected, hamming_corrected = hamming_decode(hamming_corrupted)
                error_introduced = hamming_corrupted != hamming_encoded
                
                if error_introduced and not error_detected:
                    results[error_rate]['hamming']['undetected'] += 1
                elif error_introduced and error_detected and hamming_corrected == original_message:
                    results[error_rate]['hamming']['corrected'] += 1
                elif error_introduced and error_detected:
                    results[error_rate]['hamming']['detected'] += 1
                
                results[error_rate]['hamming']['time'] += time.time() - start_time
                results[error_rate]['hamming']['overhead'] += hamming_overhead
            
            # Testing Reed-Solomon
            if message_size % 2 == 0:  # Simplified check for Reed-Solomon
                k = message_size // 2
                n = k + 2
                
                # Converting bits to bytes for RS
                byte_data = []
                for i in range(0, message_size, 8):
                    if i + 8 <= message_size:
                        byte = 0
                        for j in range(8):
                            byte = (byte << 1) | original_message[i + j]
                        byte_data.append(byte)
                
                if len(byte_data) == k:
                    start_time = time.time()
                    rs_encoded = rs_encode(byte_data, n, k)
                    rs_overhead = len(rs_encoded) / len(byte_data)
                    
                    # Errors
                    rs_corrupted = rs_encoded.copy()
                    for i in range(len(rs_corrupted)):
                        if random.random() < error_rate:
                            rs_corrupted[i] = rs_corrupted[i] ^ (1 << random.randint(0, 7))
                    
                    # Decoding
                    rs_decoded, error_detected = rs_decode(rs_corrupted, n, k)
                    error_introduced = rs_corrupted != rs_encoded
                    
                    if error_introduced and not error_detected:
                        results[error_rate]['reed_solomon']['undetected'] += 1
                    elif error_introduced and error_detected:
                        results[error_rate]['reed_solomon']['detected'] += 1
                    
                    results[error_rate]['reed_solomon']['time'] += time.time() - start_time
                    results[error_rate]['reed_solomon']['overhead'] += rs_overhead
    
    # Normalizing the results
    for error_rate in error_rates:
        for scheme in ['parity', 'hamming', 'reed_solomon']:
            results[error_rate][scheme]['detected'] /= num_messages
            results[error_rate][scheme]['corrected'] /= num_messages
            results[error_rate][scheme]['undetected'] /= num_messages
            results[error_rate][scheme]['overhead'] /= num_messages
            results[error_rate][scheme]['time'] /= num_messages
    
    return results

def plot_results(results):
    """Visualizing the test results"""
    error_rates = list(results.keys())
    schemes = ['parity', 'hamming', 'reed_solomon']
    metrics = ['detected', 'corrected', 'undetected', 'overhead', 'time']
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        
        for scheme in schemes:
            values = [results[rate][scheme][metric] for rate in error_rates]
            plt.plot(error_rates, values, marker='o', label=scheme)
        
        plt.xlabel('Error rate')
        plt.ylabel(metric.capitalize())
        plt.title(f'{metric.capitalize()} vs Error Rate')
        plt.legend()
        plt.grid(True)
        plt.show()

# Demonstration of usage
if __name__ == "__main__":
    # Example of using the codes
    message = [1, 0, 1, 0]
    print(f"Original message: {message}")
    
    # Parity bit
    parity_encoded = parity_encode(message)
    print(f"Encoded with parity bit: {parity_encoded}")
    
    # Simulating an error
    parity_corrupted = parity_encoded.copy()
    parity_corrupted[1] = bit_flip(parity_corrupted[1])
    print(f"Парити с грешка: {parity_corrupted}")
    
    decoded, valid = parity_decode(parity_corrupted)
    print(f"Декодирано: {decoded}, Валидно: {valid}")
    
    # Hamming code
    hamming_encoded = hamming_encode(message)
    print(f"Кодирано с Хеминг (7,4): {hamming_encoded}")
    
    # Error simulation
    hamming_corrupted = hamming_encoded.copy()
    hamming_corrupted[2] = bit_flip(hamming_corrupted[2])
    print(f"Хеминг с грешка: {hamming_corrupted}")
    
    decoded, error_detected, corrected = hamming_decode(hamming_corrupted)
    print(f"Декодирано: {decoded}, Грешка открита: {error_detected}, Коригирано: {corrected}")
    
    # Testing and analysis of results
    results = test_error_correction(message_size=4, num_messages=100)
    plot_results(results)