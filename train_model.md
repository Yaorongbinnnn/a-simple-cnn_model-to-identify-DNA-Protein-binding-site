# a-simple-cnn_model-to-identify-DNA-Protein-binding-site
# ====================== Environment Configuration ======================
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

# ====================== Library Imports ======================
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

# ====================== Random Seed ======================
np.random.seed(42)
tf.random.set_seed(42)
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()

# ====================== Path Configuration ======================
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ====================== Data Encoder ======================
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

# ====================== Model Architecture ======================
def build_model(input_shape, num_classes):
    inputs = Input(shape=input_shape) # Input layer
    x = Conv1D(32, 12, activation='relu', padding='same')(inputs) # Convolutional layer
    x = MaxPooling1D(4)(x) # Pooling layer
    x = Flatten()(x) # Flatten layer
    x = Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x) # Fully connected layer
    x = Dropout(0.5)(x) # Dropout layer
    outputs = Dense(num_classes, activation='softmax')(x) # Output layer
    return Model(inputs, outputs)

# ====================== Visualization ======================
def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    
    # Accuracy curve
    plt.subplot(1, 2, 1)
    max_train_acc = max(history.history['accuracy'])
    max_val_acc = max(history.history['val_accuracy'])
    plt.plot(history.history['accuracy'], label=f'Train ({max_train_acc:.2f})', color='#1f77b4', linewidth=2)
    plt.plot(history.history['val_accuracy'], label=f'Validation ({max_val_acc:.2f})', color='#ff7f0e', linestyle='--', linewidth=2)
    plt.title('Accuracy Curves', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)

    # Loss curve
    plt.subplot(1, 2, 2)
    min_train_loss = min(history.history['loss'])
    min_val_loss = min(history.history['val_loss'])
    plt.plot(history.history['loss'], label=f'Train ({min_train_loss:.2f})', color='#2ca02c', linewidth=2)
    plt.plot(history.history['val_loss'], label=f'Validation ({min_val_loss:.2f})', color='#d62728', linestyle='--', linewidth=2)
    plt.title('Loss Curves', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'training_history.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix', fontsize=14)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm_norm.max() / 2.
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            plt.text(j, i, f"{cm_norm[i, j]:.2f}",
                     horizontalalignment="center",
                     color="white" if cm_norm[i, j] > thresh else "black")

    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrix.png'))
    plt.close()

# ====================== Saliency Analysis ======================
class SaliencyAnalyzer:
    def __init__(self, model):
        self.model = model
    
    def analyze(self, sequence, target_class=1):
        seq_tensor = tf.convert_to_tensor(np.expand_dims(sequence, axis=0), dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(seq_tensor)
            predictions = self.model(seq_tensor, training=False)
            target = predictions[:, target_class]
        grads = tape.gradient(target, seq_tensor) # Calculate gradients
        saliency = tf.reduce_max(tf.abs(grads), axis=-1).numpy()[0] # Take maximum absolute value
        return saliency

def plot_saliency_heatmap(saliency_matrix, sample_labels, sequence_length):
    plt.figure(figsize=(18, 10))
    plt.imshow(saliency_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label='Saliency Value')
    plt.title('Saliency Heatmap (First 100 Positive Samples in Test Set)', fontsize=16)
    plt.xlabel('Position', fontsize=12)
    plt.ylabel('Sample Index', fontsize=12)
    
    plt.yticks(ticks=range(len(sample_labels)), labels=sample_labels)
    x_step = max(1, sequence_length // 20)
    plt.xticks(ticks=range(0, sequence_length, x_step), rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'saliency_heatmap.png'))
    plt.close()

# ====================== Data Preprocessing ======================
def preprocess_sequences(sequences, fixed_length):
    processed_sequences = []
    for seq in sequences:
        seq = seq.upper() # Convert to uppercase
        if len(seq) > fixed_length:
            processed_seq = seq[:fixed_length] # Truncate
        else:
            processed_seq = seq + 'N' * (fixed_length - len(seq)) # Pad
        processed_sequences.append(processed_seq)
    return processed_sequences

# ====================== Main Process ======================
def main():
    try:
        print("===== DNA Sequence Classifier v1.4 =====")
        
        # Data loading
        print("Loading data...")
        with open(os.path.join(DATA_DIR, "sequences.txt"), 'r') as f:
            sequences = [line.strip() for line in f if line.strip()]
        with open(os.path.join(DATA_DIR, "labels.txt"), 'r') as f:
            labels = np.array([line.strip() for line in f if line.strip()])

        # Data preprocessing
        print("Preprocessing data...")
        fixed_length = 50
        processed_sequences = preprocess_sequences(sequences, fixed_length)

        # Data encoding
        print("Encoding data...")
        encoder = DNASequenceEncoder()
        X = encoder.encode(processed_sequences)
        label_encoder = OneHotEncoder(sparse_output=False)
        y = label_encoder.fit_transform(labels.reshape(-1, 1))

        # Save encoder
        joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))

        # Data splitting
        print("Splitting dataset...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Model building
        print("Building model...")
        model = build_model((X_train.shape[1], X_train.shape[2]), y.shape[1])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Model training
        print("Starting training...")
        history = model.fit(
            X_train, y_train,
            epochs=30,
            batch_size=32,
            validation_split=0.2,
            verbose=1
        )

        # Model saving
        model_path = os.path.join(MODEL_DIR, "dna_model.keras")
        model.save(model_path)
        print(f"Model saved to: {model_path}")

        # Training curve visualization
        plot_training_history(history)

        # Confusion matrix
        y_pred = model.predict(X_test).argmax(axis=1)
        y_true = y_test.argmax(axis=1)
        plot_confusion_matrix(y_true, y_pred, label_encoder.categories_[0])

        # Saliency analysis (only for positive class samples)
        print("Performing saliency analysis...")
        analyzer = SaliencyAnalyzer(model)
        y_test_labels = y_test.argmax(axis=1)
        test_positive_indices = np.where(y_test_labels == 1)[0]

        if len(test_positive_indices) < 100:
            raise ValueError(f"Not enough positive samples in test set, actual number: {len(test_positive_indices)}")

        test_positive_indices = test_positive_indices[:100]
        saliency_matrix = []
        for idx in test_positive_indices:
            saliency = analyzer.analyze(X_test[idx])
            saliency_matrix.append(saliency)
        saliency_matrix = np.array(saliency_matrix)

        # Visualize saliency map
        plot_saliency_heatmap(
            saliency_matrix,
            [f"S{i}" for i in range(len(test_positive_indices))],
            X_test.shape[1]
        )

        # Visualize strongest sample
        saliency_sums = np.sum(saliency_matrix, axis=1)
        strongest_sample_idx = np.argmax(saliency_sums)
        strongest_saliency = saliency_matrix[strongest_sample_idx]
        strongest_sequence = processed_sequences[test_positive_indices[strongest_sample_idx]]

        plt.figure(figsize=(12, 6))
        positions = np.arange(len(strongest_saliency))
        plt.bar(positions, strongest_saliency, color='skyblue', edgecolor='black', linewidth=0.5)
        
        for i, base in enumerate(strongest_sequence):
            plt.text(i, strongest_saliency[i] + 0.005, base, 
                    ha='center', va='bottom', fontsize=8, rotation=90)
        
        plt.ylim(0, max(strongest_saliency) * 1.1)
        plt.title(f'Saliency Analysis of Strongest Sample (S{strongest_sample_idx})', fontsize=14)
        plt.xlabel('Position', fontsize=12)
        plt.ylabel('Saliency Value', fontsize=12)
        plt.xticks(positions, positions, rotation=45)
        plt.grid(alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, 'saliency_analysis.png'))
        plt.close()

    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("Troubleshooting suggestions:")
        print("1. Check TensorFlow version: pip show tensorflow (should be ≥2.15.0)")
        print("2. Verify data files exist:")
        print(f"   - {os.path.join(DATA_DIR, 'sequences.txt')}")
        print(f"   - {os.path.join(DATA_DIR, 'labels.txt')}")
        print("3. Check data format: sequences should be single-line DNA sequences, labels should be classification labels")
        print("4. Try CPU mode: export CUDA_VISIBLE_DEVICES=''")

if __name__ == "__main__":
    main()
    print("===== Training Complete =====")
