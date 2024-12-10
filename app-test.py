import streamlit as st
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict

# Set Streamlit page configuration
st.set_page_config(page_title="Effective Label Flipping Attack on MNIST", layout="wide")

# ----------------------------
# 1. Utility Functions
# ----------------------------

@st.cache_resource
def load_data():
    """Load and preprocess MNIST data."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Normalize
    x_train = x_train.astype('float32') / 255.0
    x_test  = x_test.astype('float32') / 255.0
    # Reshape
    x_train = x_train.reshape(-1,28,28,1)
    x_test  = x_test.reshape(-1,28,28,1)
    # One-hot encode
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat  = to_categorical(y_test, 10)
    return x_train, y_train, y_train_cat, x_test, y_test, y_test_cat

def create_model():
    """Build and compile the CNN model."""
    model = Sequential([
        Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(64, kernel_size=(3,3), activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(x, y, epochs=20, batch_size=128):
    """Train the model on given data."""
    model = create_model()
    model.fit(x, y, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=0)
    return model

def label_flipping(x, y, flip_ratio=0.1, source_label=7, target_label=1, seed=42):
    """Implement Label Flipping Attack: Change source_label to target_label."""
    np.random.seed(seed)
    # Find all indices with the source_label
    flip_indices = np.where(y == source_label)[0]
    n_flip = int(len(flip_indices) * flip_ratio)
    if n_flip == 0:
        return x, y, []
    # Randomly choose indices to flip
    flip_indices = np.random.choice(flip_indices, n_flip, replace=False)
    y_flipped = np.copy(y)
    y_flipped[flip_indices] = target_label
    return x, y_flipped, flip_indices

def save_model(model, filename):
    """Save the trained model."""
    model.save(filename)

def load_trained_model(filename):
    """Load a trained model."""
    if os.path.exists(filename):
        return load_model(filename)
    else:
        return None

def get_test_indices_by_label(y_test):
    """Organize test indices by their labels."""
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(y_test):
        label_to_indices[label].append(idx)
    return label_to_indices

# ----------------------------
# 2. Initialize or Load Models
# ----------------------------

def initialize_models(x_train, y_train_cat):
    """Train and save base model if not already saved."""
    if 'base_model' not in st.session_state:
        if not os.path.exists('base_model.h5'):
            with st.spinner('🔄 Training Original Model...'):
                base_model = train_model(x_train, y_train_cat)
                save_model(base_model, 'base_model.h5')
            st.success("✅ Original model trained and saved as 'base_model.h5'.")
        else:
            base_model = load_trained_model('base_model.h5')
            st.success("✅ Original model loaded from 'base_model.h5'.")
        st.session_state['base_model'] = base_model
    return st.session_state['base_model']

def initialize_poisoned_model():
    """Load poisoned model if exists; else, initialize as None."""
    if 'poisoned_model' not in st.session_state:
        if os.path.exists('poisoned_model.h5'):
            poisoned_model = load_trained_model('poisoned_model.h5')
            st.session_state['poisoned_model'] = poisoned_model
            st.sidebar.success("✅ Poisoned model loaded from 'poisoned_model.h5'.")
        else:
            st.session_state['poisoned_model'] = None
            st.sidebar.info("ℹ️ Poisoned model not available. Apply the attack to create it.")
    return st.session_state['poisoned_model']

# ----------------------------
# 3. Main Streamlit App
# ----------------------------

def main():
    st.title("🔒 Effective Label Flipping Attack on MNIST Classification")

    # Load data
    x_train, y_train, y_train_cat, x_test, y_test, y_test_cat = load_data()

    # Initialize models
    base_model = initialize_models(x_train, y_train_cat)
    poisoned_model = initialize_poisoned_model()

    # Organize test indices by label
    label_to_indices = get_test_indices_by_label(y_test)

    # Sidebar
    st.sidebar.header("🔧 Poisoning Attack Settings")

    st.sidebar.markdown("**Label Flipping Attack:** Change selected labels.")

    # Select Source and Target Labels
    source_label = st.sidebar.selectbox("🔄 Source Label (to flip from)", list(range(10)), index=7)
    target_label = st.sidebar.selectbox("🎯 Target Label (to flip to)", list(range(10)), index=1)

    # Ensure source and target labels are different
    if source_label == target_label:
        st.sidebar.warning("⚠️ Source and Target labels must be different.")
        st.sidebar.button("⚔️ Apply Label Flipping and Retrain", disabled=True)
    else:
        # Flip ratio slider
        flip_ratio = st.sidebar.slider("📊 Flip Ratio (%)", 0, 50, 10, step=1) / 100.0  # Increased max to 50%
        st.sidebar.write(f"**Current Flip Ratio:** {flip_ratio * 100:.0f}%")

        # Button to apply poisoning and retrain
        if st.sidebar.button("⚔️ Apply Label Flipping and Retrain"):
            with st.spinner('🔄 Applying label flipping attack and retraining poisoned model...'):
                x_poisoned, y_poisoned, flipped_indices = label_flipping(
                    x_train, y_train, flip_ratio=flip_ratio, source_label=source_label, target_label=target_label, seed=42)
                if len(flipped_indices) == 0:
                    st.warning("⚠️ No labels were flipped. Increase the flip ratio or ensure there are samples with the selected source label.")
                else:
                    y_poisoned_cat = to_categorical(y_poisoned, 10)
                    # Retrain poisoned model
                    poisoned_model = create_model()
                    poisoned_model.fit(x_poisoned, y_poisoned_cat, epochs=20, batch_size=128, validation_split=0.1, verbose=0)  # Increased epochs to 20
                    save_model(poisoned_model, 'poisoned_model.h5')
                    st.session_state['poisoned_model'] = poisoned_model
                    st.success(f"✅ Label flipping applied to {len(flipped_indices)} samples (from {source_label} to {target_label}). Poisoned model retrained and saved as 'poisoned_model.h5'.")

    st.markdown("---")
    st.header("📊 Model Predictions Comparison")

    # Image Selection
    st.subheader("🔍 Select Test Images by Label")
    selected_label = st.selectbox("📂 Choose a label to view its top 20 test images", options=list(range(10)), index=0)
    available_indices = label_to_indices[selected_label]

    if not available_indices:
        st.warning(f"⚠️ No test images found for label {selected_label}.")
    else:
        # Select top 20 images
        top_n = 20
        indices_to_display = available_indices[:top_n] if len(available_indices) >= top_n else available_indices
        st.write(f"📄 Displaying top {len(indices_to_display)} images for label **{selected_label}**.")

        # Display images in a grid with predictions
        cols = st.columns(4)  # 4 images per row
        for idx, img_idx in enumerate(indices_to_display):
            img = x_test[img_idx]
            true_label = y_test[img_idx]

            # Original Model Prediction
            base_pred = base_model.predict(np.expand_dims(img, axis=0))
            base_pred_label = np.argmax(base_pred)
            base_pred_conf = np.max(base_pred)

            # Poisoned Model Prediction
            if poisoned_model is not None:
                poisoned_pred = poisoned_model.predict(np.expand_dims(img, axis=0))
                poisoned_pred_label = np.argmax(poisoned_pred)
                poisoned_pred_conf = np.max(poisoned_pred)
            else:
                poisoned_pred_label = "🚫 Not Available"
                poisoned_pred_conf = "🚫 Not Available"

            # Check if this image was flipped
            is_flipped = False
            if poisoned_model is not None:
                # Since we have retrained, it's hard to track which specific test images were flipped.
                # Instead, we can infer that images with true_label == source_label might have been mispredicted as target_label.
                if true_label == source_label and poisoned_pred_label == target_label:
                    is_flipped = True

            # Create prediction text
            base_text = f"**Original:** {base_pred_label} ({base_pred_conf*100:.1f}%)"
            if poisoned_model is not None:
                poisoned_text = f"**Poisoned:** {poisoned_pred_label} ({poisoned_pred_conf*100:.1f}%)"
                if is_flipped:
                    poisoned_text += " 🔄"  # Indicate that this was a flipped sample
            else:
                poisoned_text = "🚫 Not Available"

            # Display in columns
            with cols[idx % 4]:
                st.image(img.squeeze(), caption=f"True: {true_label}", width=100)
                st.markdown(base_text)
                st.markdown(poisoned_text)

        if poisoned_model is not None:
            st.markdown("🔄 *Samples marked with 🔄 indicate that their predictions were altered due to label flipping.*")
        else:
            st.markdown("ℹ️ *Apply the label flipping attack to view poisoned model predictions.*")

    st.markdown("---")
    st.header("📈 Evaluate Model Performance")

    if st.button("📊 Evaluate Models on Test Set"):
        with st.spinner('📈 Evaluating models...'):
            base_loss, base_acc = base_model.evaluate(x_test, y_test_cat, verbose=0)
            if 'poisoned_model' in st.session_state and st.session_state['poisoned_model'] is not None:
                poisoned_model = st.session_state['poisoned_model']
                poisoned_loss, poisoned_acc = poisoned_model.evaluate(x_test, y_test_cat, verbose=0)
            else:
                poisoned_loss, poisoned_acc = None, None

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🔹 Original Model")
            st.write(f"**Accuracy:** {base_acc * 100:.2f}%")
            st.write(f"**Loss:** {base_loss:.4f}")
        with col2:
            st.subheader("🔸 Poisoned Model")
            if poisoned_acc is not None:
                st.write(f"**Accuracy:** {poisoned_acc * 100:.2f}%")
                st.write(f"**Loss:** {poisoned_loss:.4f}")
            else:
                st.write("🚫 Poisoned model not available.")

        # Plot comparison
        if poisoned_acc is not None:
            fig, ax = plt.subplots(figsize=(6,4))
            labels = ['Accuracy', 'Loss']
            original = [base_acc, base_loss]
            poisoned = [poisoned_acc, poisoned_loss]

            x = np.arange(len(labels))
            width = 0.35

            ax.bar(x - width/2, original, width, label='Original', color='blue')
            ax.bar(x + width/2, poisoned, width, label='Poisoned', color='red')

            ax.set_ylabel('Scores')
            ax.set_title('Model Performance Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.legend()

            st.pyplot(fig)
        else:
            st.info("🚫 Poisoned model not available. Apply the attack and retrain first.")

    st.markdown("---")
    st.header("📝 Explanation of Label Flipping Attack")

    st.write("""
    **Label Flipping Attack** is a type of data poisoning where the attacker intentionally changes the labels of certain training samples to incorrect classes. In this demo:

    - **Source Label:** The original label you want to flip from (e.g., 7).
    - **Target Label:** The label you want to flip to (e.g., 1).
    - **Flip Ratio:** The percentage of source labels to flip.

    **Impact:**

    - **Original Model:** Trained on clean data without any label flipping.
    - **Poisoned Model:** Retrained on data where a subset of selected source labels have been mislabeled as the target label.

    By flipping labels, the poisoned model learns incorrect mappings, which can degrade its overall performance and introduce specific vulnerabilities.

    **Usage:**

    1. **Set Attack Parameters:**
       - **Source Label:** Choose the label you want to flip from (e.g., 7).
       - **Target Label:** Choose the label you want to flip to (e.g., 1).
       - **Flip Ratio:** Adjust the percentage of source labels to flip (up to 50% for stronger attacks).
    2. **Apply Attack:**
       - Click the "**⚔️ Apply Label Flipping and Retrain**" button to perform the attack and retrain the poisoned model.
    3. **Compare Predictions:**
       - Select a label from the dropdown to view its top 20 test images.
       - For each image, view predictions from both the original and poisoned models side-by-side.
       - Images with altered predictions due to label flipping are marked with a 🔄 icon.
    4. **Evaluate Performance:**
       - Click on "**📊 Evaluate Models on Test Set**" to compare the accuracy and loss of both models.
       - View the bar chart to visualize the performance differences.

    **Observing the Effects:**

    After applying the label flipping attack with a higher flip ratio (e.g., 50%), you should observe that:

    - The **Poisoned Model** misclassifies a significant portion of images from the **Source Label** as the **Target Label**.
    - Overall model accuracy decreases due to the introduction of mislabeled data.
    - The loss increases as the model struggles to fit the conflicting labels.
    - Confidence scores for mispredicted labels may be lower, indicating uncertainty.

    **Note:** This demonstration is for educational purposes to illustrate the effects of data poisoning on machine learning models.
    """)

    st.markdown("---")
    st.markdown("**⚠️ Disclaimer:** This demo is intended for educational purposes only. Do not use it for malicious activities.")

if __name__ == '__main__':
    main()