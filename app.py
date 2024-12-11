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
import json

# Set Streamlit page configuration
st.set_page_config(page_title="🔒 Comprehensive Label Flipping Attack on MNIST", layout="wide")


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


def save_model(model, attack_params):
    """
    Save the trained model with a filename that encodes attack parameters.

    Args:
        model: Trained Keras model.
        attack_params: Dictionary containing attack parameters.

    Returns:
        filepath: Path where the model is saved.
    """
    base_dir = "models/label_flipping"
    os.makedirs(base_dir, exist_ok=True)
    filename = f"poisoned_label_flip_src{attack_params['source_label']}_tgt{attack_params['target_label']}_ratio{int(attack_params['flip_ratio'] * 100)}.h5"
    filepath = os.path.join(base_dir, filename)
    model.save(filepath)

    # Include 'filepath' in attack_params
    attack_params['filepath'] = filepath

    # Update the attack metadata
    update_attack_metadata(attack_params)
    return filepath


def load_trained_model(filepath):
    """Load a trained model from the given filepath."""
    if os.path.exists(filepath):
        try:
            model = load_model(filepath)
            return model
        except Exception as e:
            st.error(f"Error loading model from {filepath}: {e}")
            return None
    else:
        st.error(f"Model file {filepath} does not exist.")
        return None


def get_label_flipping_models():
    """Retrieve all saved label flipping models and their parameters."""
    base_dir = "models/label_flipping"
    os.makedirs(base_dir, exist_ok=True)
    models = []
    metadata = get_attack_metadata()
    for attack in metadata:
        if 'filepath' in attack:
            models.append(attack)
        else:
            st.warning(f"⚠️ Missing 'filepath' for attack: {attack}. Skipping this entry.")
    return models


def update_attack_metadata(attack_params):
    """
    Update the attack metadata JSON file with new attack parameters.

    Args:
        attack_params: Dictionary containing attack parameters.
    """
    metadata_file = "models/label_flipping/attack_metadata.json"
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                st.error(f"❌ Failed to decode JSON from {metadata_file}: {e}")
                return
    else:
        data = []
    data.append(attack_params)
    with open(metadata_file, 'w') as f:
        json.dump(data, f, indent=4)


def get_attack_metadata():
    """Retrieve attack metadata from the JSON file."""
    metadata_file = "models/label_flipping/attack_metadata.json"
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                st.error(f"❌ Failed to decode JSON from {metadata_file}: {e}")
                return []
    else:
        data = []
    return data


def get_test_indices_by_label(y_test):
    """Organize test indices by their labels."""
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(y_test):
        label_to_indices[label].append(idx)
    return label_to_indices


def plot_confusion_matrix_custom(y_true, y_pred, title):
    """Plot confusion matrix using sklearn."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title(title)
    st.pyplot(fig)


def plot_precision_recall_fscore(source_label, target_label, y_true, y_pred, title):
    """Plot Precision, Recall, and F1-Score for Source and Target Labels."""
    precision, recall, fscore, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[source_label, target_label], zero_division=0)

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
        original_model_path = 'models/original/base_model.h5'
        os.makedirs('models/original', exist_ok=True)
        if not os.path.exists(original_model_path):
            with st.spinner('🔄 Training Original Model...'):
                base_model = train_model(x_train, y_train_cat)
                base_model.save(original_model_path)
            st.success("✅ Original model trained and saved as 'models/original/base_model.h5'.")
        else:
            try:
                base_model = load_model(original_model_path)
                st.sidebar.success("✅ Original model loaded from 'models/original/base_model.h5'.")
            except Exception as e:
                st.error(f"❌ Failed to load original model: {e}")
                return None
        st.session_state['base_model'] = base_model
    return st.session_state['base_model']


# ----------------------------
# 3. Main Streamlit App
# ----------------------------

def main():
    st.title("🔒 Comprehensive Label Flipping Attack on MNIST Classification")

    # Load data
    x_train, y_train, y_train_cat, x_test, y_test, y_test_cat = load_data()

    # Initialize models
    base_model = initialize_models(x_train, y_train_cat)
    if base_model is None:
        st.stop()

    # Organize test indices by label
    label_to_indices = get_test_indices_by_label(y_test)

    # Precompute original model predictions and cache them
    if 'y_pred_original' not in st.session_state:
        with st.spinner('🔄 Computing Original Model Predictions...'):
            y_pred_original = np.argmax(base_model.predict(x_test), axis=1)
            st.session_state['y_pred_original'] = y_pred_original
    else:
        y_pred_original = st.session_state['y_pred_original']

    # ----------------------------
    # 4. Original Model Performance
    # ----------------------------

    st.header("🔹 Original Model Performance")

    # Evaluation Metrics
    with st.expander("📈 View Original Model Evaluation Metrics"):
        base_loss, base_acc = base_model.evaluate(x_test, y_test_cat, verbose=0)
        st.write(f"**Accuracy:** {base_acc * 100:.2f}%")
        st.write(f"**Loss:** {base_loss:.4f}")

        # Confusion Matrix
        st.markdown("**Confusion Matrix:**")
        plot_confusion_matrix_custom(y_test, y_pred_original, "Original Model Confusion Matrix")

        # Prediction Distribution
        st.markdown("**Prediction Distribution:**")
        fig, ax = plt.subplots(figsize=(10, 6))
        unique_orig, counts_orig = np.unique(y_pred_original, return_counts=True)
        counts_orig_dict = dict(zip(unique_orig, counts_orig))
        classes = list(range(10))
        orig_counts = [counts_orig_dict.get(cls, 0) for cls in classes]
        ax.bar(classes, orig_counts, color='blue')
        ax.set_xlabel('Digit Class')
        ax.set_ylabel('Number of Predictions')
        ax.set_title('Original Model Prediction Distribution')
        st.pyplot(fig)

    # Sample Predictions
    st.subheader("🔍 Sample Predictions from Original Model")
    selected_label = st.selectbox("📂 Choose a label to view its sample test images", options=list(range(10)), index=0,
                                  key='original_label_selection')
    available_indices = label_to_indices[selected_label]

    if not available_indices:
        st.warning(f"⚠️ No test images found for label {selected_label}.")
    else:
        # Select top 20 images for the selected label
        top_n = 20
        selected_indices = available_indices[:top_n] if len(available_indices) >= top_n else available_indices

        st.write(f"📄 Displaying {len(selected_indices)} images for label **{selected_label}**.")

        # Display images in a grid with predictions
        cols = st.columns(4)  # 4 images per row
        for idx, img_idx in enumerate(selected_indices):
            img = x_test[img_idx]
            true_label = y_test[img_idx]

            # Original Model Prediction
            base_pred = y_pred_original[img_idx]

            # Poisoned Model Prediction
            if 'y_pred_poisoned' in st.session_state:
                y_pred_poisoned = st.session_state['y_pred_poisoned']
                poisoned_pred_label = y_pred_poisoned[img_idx]
            else:
                poisoned_pred_label = "🚫 Not Available"

            # Check if this image was influenced by label flipping
            is_flipped_influence = False
            if 'selected_model_index' in st.session_state and 'label_flipping_models' in locals():
                selected_model = label_flipping_models[st.session_state['selected_model_index']]
                if selected_model['source_label'] == selected_label:
                    if base_pred == selected_label and poisoned_pred_label == selected_model['target_label']:
                        is_flipped_influence = True

            # Create prediction text
            base_text = f"**Original:** {base_pred} ({'✅' if base_pred == selected_label else '❌'})"
            if poisoned_pred_label != "🚫 Not Available":
                poisoned_text = f"**Poisoned:** {poisoned_pred_label} ({'✅' if poisoned_pred_label == selected_label else '❌'})"
                if is_flipped_influence:
                    poisoned_text += " 🔄"  # Indicate that this prediction was influenced by label flipping
            else:
                poisoned_text = "🚫 Not Available"

            # Display in columns
            with cols[idx % 4]:
                st.image(img.squeeze(), caption=f"True Label: {true_label}", width=100)
                st.markdown(base_text)
                st.markdown(poisoned_text)

        if 'y_pred_poisoned' in st.session_state and 'selected_model' in locals():
            if any([y_pred_poisoned[idx] == selected_model['target_label'] for idx in selected_indices]):
                st.markdown("🔄 *🔄 indicates that the prediction was altered due to label flipping.*")
            else:
                st.markdown("ℹ️ *No predictions were altered due to label flipping for the selected images.*")
        else:
            st.markdown(
                "ℹ️ *Poisoned model predictions not available. Apply and select a poisoned model to see influence indicators.*")

    st.markdown("---")
    st.header("🔧 Label Flipping Attack Settings")

    st.sidebar.header("🔧 Label Flipping Attack Configuration")

    st.sidebar.markdown("**Configure Label Flipping Attack Parameters:**")

    # Select Source and Target Labels
    source_label = st.sidebar.selectbox("🔄 Source Label (to flip from)", list(range(10)), index=7,
                                        key='lf_source_label')
    target_label = st.sidebar.selectbox("🎯 Target Label (to flip to)", list(range(10)), index=1, key='lf_target_label')

    # Ensure source and target labels are different
    if source_label == target_label:
        st.sidebar.warning("⚠️ Source and Target labels must be different.")
        st.sidebar.button("⚔️ Apply Label Flipping and Retrain", disabled=True)
    else:
        # Flip ratio slider
        flip_ratio = st.sidebar.slider("📊 Flip Ratio (%)", 0, 50, 10, step=1,
                                       key='lf_flip_ratio') / 100.0  # Increased max to 50%
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
                    # Save poisoned model with parameters in filename
                    attack_params = {
                        'source_label': source_label,
                        'target_label': target_label,
                        'flip_ratio': flip_ratio,
                        'num_flipped': len(flipped_indices)
                    }
                    poisoned_model_path = save_model(poisoned_model, attack_params)
                    st.success(
                        f"✅ Label flipping applied to {len(flipped_indices)} samples (from {source_label} to {target_label}). Poisoned model retrained and saved as '{poisoned_model_path}'.")
                    # Store flipped_indices for display
                    st.session_state['flipped_indices'] = flipped_indices

    st.markdown("---")
    st.header("📊 Model Predictions Comparison")

    # Retrieve all label flipping models
    label_flipping_models = get_label_flipping_models()

    if not label_flipping_models:
        st.info("ℹ️ No poisoned models available. Apply a label flipping attack to create one.")
    else:
        # Dropdown to select a poisoned model
        st.subheader("🔍 Select a Poisoned Model for Comparison")
        model_options = [
            f"Source: {m['source_label']} → Target: {m['target_label']} | Flip Ratio: {int(m['flip_ratio'] * 100)}%" for
            m in label_flipping_models]
        selected_model_index = st.selectbox("📂 Choose a poisoned model:",
                                            options=list(range(len(label_flipping_models))),
                                            format_func=lambda x: model_options[x], key='model_selection')

        selected_model = label_flipping_models[selected_model_index]
        poisoned_model = load_trained_model(selected_model['filepath'])

        if poisoned_model is None:
            st.error(f"❌ Failed to load model from '{selected_model['filepath']}'.")
        else:
            # Precompute poisoned model predictions and cache them
            if 'y_pred_poisoned' not in st.session_state or st.session_state.get(
                    'selected_model_index') != selected_model_index:
                with st.spinner('🔄 Computing Poisoned Model Predictions...'):
                    y_pred_poisoned = np.argmax(poisoned_model.predict(x_test), axis=1)
                    st.session_state['y_pred_poisoned'] = y_pred_poisoned
                    st.session_state['selected_model_index'] = selected_model_index
            else:
                y_pred_poisoned = st.session_state['y_pred_poisoned']

            # Organize test indices by label and prediction correctness
            if 'label_correct_incorrect_indices' not in st.session_state:
                label_correct_incorrect_indices = defaultdict(lambda: {'correct': [], 'incorrect': []})
                for idx, label in enumerate(y_test):
                    pred = y_pred_original[idx]
                    if pred == label:
                        label_correct_incorrect_indices[label]['correct'].append(idx)
                    else:
                        label_correct_incorrect_indices[label]['incorrect'].append(idx)
                st.session_state['label_correct_incorrect_indices'] = label_correct_incorrect_indices
            else:
                label_correct_incorrect_indices = st.session_state['label_correct_incorrect_indices']

            # Image Selection
            st.subheader("🔍 Select Test Images by Label")
            selected_label = st.selectbox("📂 Choose a label to view its top 20 test images", options=list(range(10)),
                                          index=0, key='comparison_label_selection')
            available_indices = label_to_indices[selected_label]

            if not available_indices:
                st.warning(f"⚠️ No test images found for label {selected_label}.")
            else:
                # Select top 20 images (without balancing)
                top_n = 20
                selected_indices = available_indices[:top_n] if len(available_indices) >= top_n else available_indices

                st.write(f"📄 Displaying {len(selected_indices)} images for label **{selected_label}**.")

                # Display images in a grid with predictions
                cols = st.columns(4)  # 4 images per row
                for idx, img_idx in enumerate(selected_indices):
                    img = x_test[img_idx]
                    true_label = y_test[img_idx]

                    # Original Model Prediction
                    base_pred = y_pred_original[img_idx]

                    # Poisoned Model Prediction
                    poisoned_pred_label = y_pred_poisoned[img_idx]

                    # Check if this image was influenced by label flipping
                    is_flipped_influence = False
                    if selected_model['source_label'] == selected_label:
                        # If the source label was flipped to target, check if prediction is now target_label
                        if base_pred == selected_label and poisoned_pred_label == selected_model['target_label']:
                            is_flipped_influence = True

                    # Create prediction text
                    base_text = f"**Original:** {base_pred} ({'✅' if base_pred == selected_label else '❌'})"
                    poisoned_text = f"**Poisoned:** {poisoned_pred_label} ({'✅' if poisoned_pred_label == selected_label else '❌'})"
                    if is_flipped_influence:
                        poisoned_text += " 🔄"  # Indicate that this prediction was influenced by label flipping

                    # Display in columns
                    with cols[idx % 4]:
                        st.image(img.squeeze(), caption=f"True Label: {true_label}", width=100)
                        st.markdown(base_text)
                        st.markdown(poisoned_text)

                if any([y_pred_poisoned[idx] == selected_model['target_label'] for idx in selected_indices]):
                    st.markdown("🔄 *🔄 indicates that the prediction was altered due to label flipping.*")
                else:
                    st.markdown("ℹ️ *No predictions were altered due to label flipping for the selected images.*")

                # ----------------------------
                # 4. Model Performance Evaluation
                # ----------------------------

                st.markdown("---")
                st.header("📈 Model Performance Evaluation")

                # Button to evaluate models
                if st.button("📊 Evaluate Models on Test Set"):
                    with st.spinner('📈 Evaluating models...'):
                        # Original Model Evaluation
                        base_loss, base_acc = base_model.evaluate(x_test, y_test_cat, verbose=0)

                        # Poisoned Model Evaluation
                        if label_flipping_models:
                            poisoned_model = load_trained_model(label_flipping_models[selected_model_index]['filepath'])
                            if poisoned_model is not None:
                                poisoned_loss, poisoned_acc = poisoned_model.evaluate(x_test, y_test_cat, verbose=0)
                            else:
                                poisoned_loss, poisoned_acc = None, None
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
                    y_pred_original_eval = np.argmax(base_model.predict(x_test), axis=1)
                    if poisoned_acc is not None:
                        y_pred_poisoned_eval = np.argmax(poisoned_model.predict(x_test), axis=1)
                    else:
                        y_pred_poisoned_eval = None

                    # Original Model Confusion Matrix
                    st.markdown("**🔹 Original Model Confusion Matrix:**")
                    plot_confusion_matrix_custom(y_test, y_pred_original_eval, "Original Model Confusion Matrix")

                    if y_pred_poisoned_eval is not None:
                        # Poisoned Model Confusion Matrix
                        st.markdown("**🔸 Poisoned Model Confusion Matrix:**")
                        plot_confusion_matrix_custom(y_test, y_pred_poisoned_eval, "Poisoned Model Confusion Matrix")

                    # Plot prediction distribution
                    st.subheader("📊 Prediction Distribution Comparison")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    # Original Model
                    unique_orig, counts_orig = np.unique(y_pred_original_eval, return_counts=True)
                    counts_orig_dict = dict(zip(unique_orig, counts_orig))
                    # Poisoned Model
                    if y_pred_poisoned_eval is not None:
                        unique_poisoned, counts_poisoned = np.unique(y_pred_poisoned_eval, return_counts=True)
                        counts_poisoned_dict = dict(zip(unique_poisoned, counts_poisoned))
                    # Prepare data for plotting
                    classes = list(range(10))
                    orig_counts = [counts_orig_dict.get(cls, 0) for cls in classes]
                    if y_pred_poisoned_eval is not None:
                        poisoned_counts = [counts_poisoned_dict.get(cls, 0) for cls in classes]
                    # Plot
                    width = 0.35
                    x = np.arange(len(classes))
                    ax.bar(x - width / 2, orig_counts, width, label='Original', color='blue')
                    if y_pred_poisoned_eval is not None:
                        ax.bar(x + width / 2, poisoned_counts, width, label='Poisoned', color='red')
                    ax.set_xlabel('Digit Class')
                    ax.set_ylabel('Number of Predictions')
                    ax.set_title('Prediction Distribution Comparison')
                    ax.set_xticks(x)
                    ax.set_xticklabels(classes)
                    ax.legend()
                    st.pyplot(fig)

                # ----------------------------
                # 5. Summary Analysis
                # ----------------------------

                st.markdown("---")
                st.header("📈 Summary Analysis of Label Flipping Attack")

                # Button to view summary analysis
                if label_flipping_models:
                    if st.button("📈 View Summary Analysis"):
                        with st.spinner('📈 Analyzing label flipping attack...'):
                            selected_model = label_flipping_models[selected_model_index]
                            poisoned_model = load_trained_model(selected_model['filepath'])
                            if poisoned_model is not None:
                                y_pred_original = st.session_state['y_pred_original']
                                y_pred_poisoned = np.argmax(poisoned_model.predict(x_test), axis=1)

                                # Analyze mispredictions for source label
                                source_mispredictions = np.sum(
                                    (y_test == selected_model['source_label']) & (
                                                y_pred_poisoned == selected_model['target_label']))
                                total_source = np.sum(y_test == selected_model['source_label'])
                                percentage_mispredicted = (
                                                                      source_mispredictions / total_source) * 100 if total_source > 0 else 0

                                st.write(
                                    f"**Total Test Images with Source Label ({selected_model['source_label']}):** {total_source}")
                                st.write(
                                    f"**Number of {selected_model['source_label']} Images Misclassified as {selected_model['target_label']}:** {source_mispredictions}")
                                st.write(
                                    f"**Percentage of {selected_model['source_label']} Images Misclassified as {selected_model['target_label']}:** {percentage_mispredicted:.2f}%")

                                # Plot pie chart
                                fig, ax = plt.subplots()
                                labels = [f"Correctly Classified as {selected_model['source_label']}",
                                          f"Mispredicted as {selected_model['target_label']}"]
                                sizes = [total_source - source_mispredictions, source_mispredictions]
                                colors = ['lightgreen', 'lightcoral']
                                ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
                                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                                plt.title(
                                    f"Mispredictions of Label {selected_model['source_label']} as {selected_model['target_label']}")
                                st.pyplot(fig)

                                # Precision, Recall, F1-Score Analysis
                                st.subheader("📈 Precision, Recall, and F1-Score Comparison")
                                plot_precision_recall_fscore(selected_model['source_label'],
                                                             selected_model['target_label'],
                                                             y_test, y_pred_original,
                                                             "Original Model Precision, Recall, and F1-Score")
                                plot_precision_recall_fscore(selected_model['source_label'],
                                                             selected_model['target_label'],
                                                             y_test, y_pred_poisoned,
                                                             "Poisoned Model Precision, Recall, and F1-Score")
                            else:
                                st.warning("❌ Poisoned model not found.")
                else:
                    st.info("ℹ️ No poisoned models available. Apply a label flipping attack to create one.")

                # ----------------------------
                # 6. Explanation Section
                # ----------------------------

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
                   - Select a poisoned model from the dropdown to view its top 20 test images.
                   - For each image, view predictions from both the original and poisoned models side-by-side.
                   - Images with altered predictions due to label flipping are marked with a 🔄 icon.
                4. **Evaluate Performance:**
                   - Click on "**📊 Evaluate Models on Test Set**" to compare the accuracy and loss of both models.
                   - View the confusion matrices and prediction distribution charts to visualize the performance differences.
                5. **Analyze Attack Effectiveness:**
                   - Click on "**📈 View Summary Analysis**" to see how many labels were affected and visualize the impact through charts.
                   - Review the **Summary Statistics** and **Precision, Recall, and F1-Score Comparison** to understand the extent of the attack.

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