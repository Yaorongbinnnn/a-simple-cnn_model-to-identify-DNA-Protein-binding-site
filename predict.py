#!/usr/bin/env python3
# ====================== DNA Sequence Prediction Script =======================
import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# ====================== Fix Random Seeds =======================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ====================== Path Configuration =======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")

# ====================== Sequence Encoder =======================
class DNASequenceEncoder:
    def __init__(self):
        self.base_dict = {
            'A': [1, 0, 0, 0],
            'T': [0, 1, 0, 0],
            'C': [0, 0, 1, 0],
            'G': [0, 0, 0, 1],
            'N': [0.25, 0.25, 0.25, 0.25]
        }
    
    def encode(self, sequences):
        encoded_seqs = []
        for seq in sequences:
            cleaned_seq = [c if c in self.base_dict else 'N' for c in seq.upper()]
            encoded = [self.base_dict[c] for c in cleaned_seq]
            encoded_seqs.append(np.array(encoded))
        return np.stack(encoded_seqs)

def preprocess_sequences(sequences, fixed_length=50):
    processed = []
    for seq in sequences:
        seq = seq.upper()
        if len(seq) > fixed_length:
            processed.append(seq[:fixed_length])
        else:
            processed.append(seq + 'N' * (fixed_length - len(seq)))
    return processed

# ====================== Saliency Analyzer =======================
class SaliencyAnalyzer:
    def __init__(self, model):
        self.model = model
    
    def analyze(self, sequence):
        seq_tensor = tf.convert_to_tensor(np.expand_dims(sequence, axis=0), dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(seq_tensor)
            predictions = self.model(seq_tensor, training=False)
            target = predictions[:, 1]
        grads = tape.gradient(target, seq_tensor)
        
        # Calculate saliency scores (corrected axis)
        saliency = tf.reduce_max(tf.abs(grads), axis=-1).numpy()[0]
        return saliency, predictions.numpy()[0]

# ====================== Main Function =======================
def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description='DNA Sequence Binding Prediction')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-s', '--sequence', type=str, help='Input single DNA sequence')
    group.add_argument('-f', '--file', type=str, help='Input file with multiple sequences')
    
    parser.add_argument('-S', '--saliency', action='store_true',
                       help='Enable saliency analysis visualization')
    parser.add_argument('-O', '--output', type=str,
                       help='Output directory for saliency plots', default='saliency_results')
    
    args = parser.parse_args()

    # ====================== Load Model =======================
    model_path = os.path.join(MODEL_DIR, "dna_model.keras")
    encoder = DNASequenceEncoder()
    model = load_model(model_path, compile=False)
    model.trainable = False  # Freeze model weights

    # ====================== Read Input =======================
    sequences = []
    if args.sequence:
        sequences = [args.sequence]
    elif args.file:
        with open(args.file, 'r') as f:
            sequences = [line.strip() for line in f.readlines()]

    # ====================== Preprocess Sequences =======================
    processed_seqs = preprocess_sequences(sequences)
    encoded_seqs = encoder.encode(processed_seqs)

    # ====================== Run Prediction =======================
    predictions = model.predict(encoded_seqs, verbose=0)
    
    # ====================== Format Output =======================
    print("\nPrediction Results:")
    print("=" * 50)
    for i, seq in enumerate(sequences):
        class_0_prob = predictions[i][0] * 100
        class_1_prob = predictions[i][1] * 100
        
        if class_0_prob >= class_1_prob:
            pred_class = '0'
            confidence = class_0_prob
        else:
            pred_class = '1'
            confidence = class_1_prob
        
        print(f"Sequence: {seq[:50]}")
        print(f"Prediction: ['{pred_class}'] (Confidence: {confidence:.2f}%)")
        print("-" * 50)

    # ====================== Saliency Visualization =======================
    if args.saliency:
        analyzer = SaliencyAnalyzer(model)
        os.makedirs(args.output, exist_ok=True)
        
        for idx, (seq, encoded) in enumerate(zip(processed_seqs, encoded_seqs)):
            saliency, pred_probs = analyzer.analyze(encoded)
            
            plt.figure(figsize=(15, 5))
            
            # Subplot 1: Saliency Map
            plt.subplot(1, 2, 1)
            plt.bar(range(len(saliency)), saliency)
            plt.title(f"Sequence {idx+1} Saliency Analysis")
            plt.xlabel("Base Position")
            plt.ylabel("Saliency Score")
            
            # Set X-axis: two-line display (base letters + position numbers)
            positions = range(0, len(seq))
            x_labels = []
            for pos in positions:
                if pos % 10 == 0:  # Show numbers every 10 positions
                    x_labels.append(f"{seq[pos]}\n{pos+1}")
                else:
                    x_labels.append(f"{seq[pos]}\n")  # Only show base, no number
            
            plt.xticks(positions, x_labels, rotation=0)  # Set custom labels
            plt.grid(True, linestyle='--', alpha=0.6)
            
            # Subplot 2: Class Probabilities
            plt.subplot(1, 2, 2)
            labels = ['Class 0', 'Class 1']
            plt.bar(labels, pred_probs)
            plt.title("Class Probability Distribution")
            plt.ylim(0, 1)
            plt.grid(True, linestyle='--', alpha=0.6)
            
            # Save plot
            save_path = os.path.join(args.output, f"saliency_{idx+1}.png")
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            
            print(f"Saliency plot saved to: {save_path}")

if __name__ == "__main__":
    # Set matplotlib backend for server environments
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    main()
