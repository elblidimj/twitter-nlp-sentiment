import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Conv1D, Dense, Dropout, GlobalMaxPooling1D, BatchNormalization
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras import regularizers

MAX_LEN = 30 
SAVE_PATH = "twitter-datasets"

def build_cnn_model(embeddings: np.ndarray):
    vocab_size = embeddings.shape[0]  
    embedding_dim = embeddings.shape[1] 
    
    model = Sequential(name="Sentiment_CNN")
    
    # 1. Embedding
    model.add(Embedding(
        input_dim=vocab_size, 
        output_dim=embedding_dim,
        weights=[embeddings],       
        input_length=MAX_LEN,       
        trainable=True,
        name="embedding_layer"
    ))
    
    # 2. Conv Block (The one we will visualize)
    model.add(Conv1D(filters=256, kernel_size=7, activation='relu', padding='same', name="conv_layer_1"))
    model.add(BatchNormalization())
    
    # 3. Second Conv Block
    model.add(Conv1D(filters=128, kernel_size=5, activation='relu', padding='same', name="conv_layer_2"))
    
    # 4. Pooling & Dense
    model.add(GlobalMaxPooling1D()) 
    model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.6))
    model.add(Dense(1, activation='sigmoid', name="output_layer")) 
    
    model.compile(optimizer=Adam(learning_rate=0.0005), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    return model

def save_cnn_visualizations(model, history, sample_tweet_indices=None, vocab_inv=None):
    """
    Generates and saves Training History, Architecture, and Feature Heatmaps.
    Fixes 'Layer never called' error by ensuring model is built.
    """
    
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # 0. Ensure model is built and has input/output shapes defined
    # This prevents the "AttributeError: The layer has never been called"
    if not model.built:
        dummy_input = np.zeros((1, MAX_LEN))
        model(dummy_input)

    # --- 1. Training History (Accuracy/Loss Curves) ---
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.savefig(f"{SAVE_PATH}/training_history.png")
    plt.close()
    print(f"Saved: {SAVE_PATH}/training_history.png")

    # --- 2. Architecture Diagram ---
    try:
        from tensorflow.keras.utils import plot_model
        plot_model(
            model, 
            to_file=f"{SAVE_PATH}/architecture_workflow.png", 
            show_shapes=True, 
            show_layer_names=True
        )
        print(f"Saved: {SAVE_PATH}/architecture_workflow.png")
    except Exception as e:
        print(f"Skipping architecture plot: {e}")
        print("Tip: Run 'pip install pydot graphviz' and install Graphviz software to enable this.")

    # --- 3. Feature Map Visualization (Heatmap) ---
    if sample_tweet_indices is not None and vocab_inv is not None:
        try:
            # Create a sub-model that outputs the conv_layer_1 activations
            # Using model.inputs (plural) is more stable in newer Keras/TF versions
            intermediate_layer_model = Model(inputs=model.inputs,
                                             outputs=model.get_layer("conv_layer_1").output)
            
            # Prepare sample input (batch size 1)
            sample_input = np.array([sample_tweet_indices])
            intermediate_output = intermediate_layer_model.predict(sample_input, verbose=0)
            
            plt.figure(figsize=(16, 10))
            data_to_plot = intermediate_output[0, :, :50].T
            
            sns.heatmap(data_to_plot, xticklabels=vocab_inv, cmap='viridis', cbar_kws={'label': 'Activation Intensity'})
            plt.title('CNN Feature Map: First 50 Filters Activations')
            plt.xlabel('Tweet Tokens')
            plt.ylabel('Filter Index')
            plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig(f"{SAVE_PATH}/cnn_heatmap.png")
            plt.close()
            print(f"Saved: {SAVE_PATH}/cnn_heatmap.png")
        except Exception as e:
            print(f"Heatmap generation failed: {e}")

    print(f"✅ Visualization process complete.")