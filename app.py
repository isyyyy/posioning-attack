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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support

# Set Streamlit page configuration
st.set_page_config(page_title="Comprehensive Label Flipping Attack on MNIST", layout="wide")


# ----------------------------
# 1. Utility Functions
# ----------------------------

@st.cache_resource
def load_data():
    """Load and preprocess MNIST data."""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Normalize
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    # Reshape
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    # One-hot encode
    y_train_cat = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test, 10)
    return x_train, y_train, y_train_cat, x_test, y_test, y_test_cat


def create_model():
    """Build and compile the CNN model."""
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_model(x, y, epochs=30, batch_size=128):
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


def plot_confusion_matrix(y_true, y_pred, title):
    """Plot confusion matrix using sklearn."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title(title)
    st.pyplot(fig)


def plot_precision_recall_fscore(source_label, target_label, y_true, y_pred, title):
    """Plot Precision, Recall, and F1-Score for Source and Target Labels."""
    precision, recall, fscore, _ = precision_recall_fscore_support(y_true, y_pred, labels=[source_label, target_label],
                                                                   zero_division=0)

    labels = [f"Label {source_label}", f"Label {target_label}"]
    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - width, precision, width, label='Precision', color='skyblue')
    ax.bar(x, recall, width, label='Recall', color='lightgreen')
    ax.bar(x + width, fscore, width, label='F1-Score', color='salmon')

    ax.set_xlabel('Labels')
    ax.set_ylabel('Scores')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.legend()

    st.pyplot(fig)


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
            if base_model is None:
                with st.spinner('🔄 Training Original Model...'):
                    base_model = train_model(x_train, y_train_cat)
                    save_model(base_model, 'base_model.h5')
                st.success("✅ Original model trained and saved as 'base_model.h5'.")
            else:
                st.sidebar.success("✅ Original model loaded from 'base_model.h5'.")
        st.session_state['base_model'] = base_model
    return st.session_state['base_model']


def initialize_poisoned_model():
    """Load poisoned model if exists; else, initialize as None."""
    if 'poisoned_model' not in st.session_state:
        if os.path.exists('poisoned_model.h5'):
            poisoned_model = load_trained_model('poisoned_model.h5')
            if poisoned_model is not None:
                st.sidebar.success("✅ Poisoned model loaded from 'poisoned_model.h5'.")
                st.session_state['poisoned_model'] = poisoned_model
            else:
                st.sidebar.error("❌ Poisoned model file found but failed to load.")
                st.session_state['poisoned_model'] = None
        else:
            st.session_state['poisoned_model'] = None
            st.sidebar.info("ℹ️ Poisoned model not available. Apply the attack to create it.")
    return st.session_state['poisoned_model']


# ----------------------------
# 3. Main Streamlit App
# ----------------------------

def main():
    st.title("🔒 Comprehensive Label Flipping Attack on MNIST Classification")

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
                    x_train, y_train, flip_ratio=flip_ratio, source_label=source_label, target_label=target_label,
                    seed=42)
                if len(flipped_indices) == 0:
                    st.warning(
                        "⚠️ No labels were flipped. Increase the flip ratio or ensure there are samples with the selected source label.")
                else:
                    y_poisoned_cat = to_categorical(y_poisoned, 10)
                    # Retrain poisoned model
                    poisoned_model = create_model()
                    poisoned_model.fit(x_poisoned, y_poisoned_cat, epochs=30, batch_size=128, validation_split=0.1,
                                       verbose=0)  # Increased epochs to 30
                    save_model(poisoned_model, 'poisoned_model.h5')
                    st.session_state['poisoned_model'] = poisoned_model
                    st.success(
                        f"✅ Label flipping applied to {len(flipped_indices)} samples (from {source_label} to {target_label}). Poisoned model retrained and saved as 'poisoned_model.h5'.")

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
                # Since labels are flipped in training, but test labels are not altered,
                # we infer mispredictions where the model predicts target_label for true_label.
                if true_label == source_label and poisoned_pred_label == target_label:
                    is_flipped = True

            # Create prediction text
            base_text = f"**Original:** {base_pred_label} ({base_pred_conf * 100:.1f}%)"
            if poisoned_model is not None:
                poisoned_text = f"**Poisoned:** {poisoned_pred_label} ({poisoned_pred_conf * 100:.1f}%)"
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
            st.markdown(
                "🔄 *Samples marked with 🔄 indicate that their predictions were altered due to label flipping.*")
        else:
            st.markdown("ℹ️ *Apply the label flipping attack to view poisoned model predictions.*")

    st.markdown("---")
    st.header("📈 Evaluate Model Performance")

    # Button to evaluate models
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

        # Plot confusion matrices
        st.subheader("📈 Confusion Matrix Comparison")
        y_pred_original = base_model.predict(x_test)
        y_pred_original = np.argmax(y_pred_original, axis=1)
        if poisoned_model is not None:
            y_pred_poisoned = poisoned_model.predict(x_test)
            y_pred_poisoned = np.argmax(y_pred_poisoned, axis=1)
        else:
            y_pred_poisoned = None

        # Original Model Confusion Matrix
        st.markdown("**🔹 Original Model Confusion Matrix:**")
        plot_confusion_matrix(y_test, y_pred_original, "Original Model Confusion Matrix")

        if poisoned_model is not None:
            # Poisoned Model Confusion Matrix
            st.markdown("**🔸 Poisoned Model Confusion Matrix:**")
            plot_confusion_matrix(y_test, y_pred_poisoned, "Poisoned Model Confusion Matrix")

        # Plot prediction distribution
        st.subheader("📊 Prediction Distribution Comparison")
        fig, ax = plt.subplots(figsize=(10, 6))
        # Original Model
        unique_orig, counts_orig = np.unique(y_pred_original, return_counts=True)
        counts_orig_dict = dict(zip(unique_orig, counts_orig))
        # Poisoned Model
        if y_pred_poisoned is not None:
            unique_poisoned, counts_poisoned = np.unique(y_pred_poisoned, return_counts=True)
            counts_poisoned_dict = dict(zip(unique_poisoned, counts_poisoned))
        # Prepare data for plotting
        classes = list(range(10))
        orig_counts = [counts_orig_dict.get(cls, 0) for cls in classes]
        if y_pred_poisoned is not None:
            poisoned_counts = [counts_poisoned_dict.get(cls, 0) for cls in classes]
        # Plot
        width = 0.35
        x = np.arange(len(classes))
        ax.bar(x - width / 2, orig_counts, width, label='Original', color='blue')
        if y_pred_poisoned is not None:
            ax.bar(x + width / 2, poisoned_counts, width, label='Poisoned', color='red')
        ax.set_xlabel('Digit Class')
        ax.set_ylabel('Number of Predictions')
        ax.set_title('Prediction Distribution Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.legend()
        st.pyplot(fig)

    st.markdown("---")
    st.header("📈 Summary Analysis of Label Flipping Attack")

    # Button to view summary analysis
    if 'poisoned_model' in st.session_state and st.session_state['poisoned_model'] is not None:
        if st.button("📈 View Summary Analysis"):
            with st.spinner('📈 Analyzing label flipping attack...'):
                poisoned_model = st.session_state['poisoned_model']
                y_pred_original = np.argmax(base_model.predict(x_test), axis=1)
                y_pred_poisoned = np.argmax(poisoned_model.predict(x_test), axis=1)

                # Analyze mispredictions for source label
                source_mispredictions = np.sum((y_test == source_label) & (y_pred_poisoned == target_label))
                total_source = np.sum(y_test == source_label)
                percentage_mispredicted = (source_mispredictions / total_source) * 100 if total_source > 0 else 0

                st.write(f"**Total Test Images with Source Label ({source_label}):** {total_source}")
                st.write(
                    f"**Number of {source_label} Images Misclassified as {target_label}:** {source_mispredictions}")
                st.write(
                    f"**Percentage of {source_label} Images Misclassified as {target_label}:** {percentage_mispredicted:.2f}%")

                # Plot pie chart
                fig, ax = plt.subplots()
                labels = [f"Correctly Classified as {source_label}", f"Mispredicted as {target_label}"]
                sizes = [total_source - source_mispredictions, source_mispredictions]
                colors = ['lightgreen', 'lightcoral']
                ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                plt.title(f"Mispredictions of Label {source_label} as {target_label}")
                st.pyplot(fig)

                # Precision, Recall, F1-Score Analysis
                st.subheader("📈 Precision, Recall, and F1-Score Comparison")
                plot_precision_recall_fscore(source_label, target_label, y_test, y_pred_original,
                                             "Original Model Precision, Recall, and F1-Score")
                if y_pred_poisoned is not None:
                    plot_precision_recall_fscore(source_label, target_label, y_test, y_pred_poisoned,
                                                 "Poisoned Model Precision, Recall, and F1-Score")
    else:
        st.info(
            "ℹ️ Poisoned model not available. Apply the label flipping attack and retrain to view summary analysis.")

    st.markdown("---")
    st.header("📝 Explanation of Label Flipping Attack")

    st.write("""
    **Label Flipping Attack** is a type of data poisoning where the attacker intentionally changes the labels of certain training samples to incorrect classes. In this demo:

    - **Source Label:** The original label you want to flip from (e.g., 7).
    - **Target Label:** The label you want to flip to (e.g., 1).
    - **Flip Ratio:** The percentage of source labels to flip (up to 50% for stronger attacks).

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
       - View the confusion matrices and prediction distribution charts to visualize the performance differences.
    5. **Analyze Attack Effectiveness:**
       - Click on "**📈 View Summary Analysis**" to see how many labels were affected and visualize the impact through charts.
       - Review the **Summary Statistics** and **Visualizations** to understand the extent of the attack.

    **Observing the Effects:**

    After applying the label flipping attack with a higher flip ratio (e.g., 50%), you should observe that:

    - The **Poisoned Model** misclassifies a significant portion of images from the **Source Label** as the **Target Label**.
    - Overall model accuracy decreases due to the introduction of mislabeled data.
    - The loss increases as the model struggles to fit the conflicting labels.
    - Precision, Recall, and F1-Score for the source and target labels are adversely affected.

    **Example Observations:**

    - **Original Model:** Accurately predicts the true label with high confidence.
    - **Poisoned Model:** Frequently mispredicts the source label as the target label, especially on images whose labels were flipped.

    **Note:** This demonstration is for educational purposes to illustrate the effects of data poisoning on machine learning models.
    """)

    st.markdown("---")
    st.markdown(
        "**⚠️ Disclaimer:** This demo is intended for educational purposes only. Do not use it for malicious activities.")


if __name__ == '__main__':
    main()