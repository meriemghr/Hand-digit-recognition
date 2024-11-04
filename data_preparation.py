import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Set up title and sidebar
st.title("MNIST Data Preparation and Visualization")

def display_images(images, labels, num_images=10):
    # Create a figure with subplots
    fig, axes = plt.subplots(1, num_images, figsize=(10, 3))
    
    # Loop over the number of images
    for i in range(num_images):
        axes[i].imshow(images[i], cmap='gray')
        axes[i].set_title(f"Label: {labels[i]}")
        axes[i].axis('off')  # Hide the axis
        
    st.pyplot(fig)


# Data Overview
st.header("Data Overview")
st.write("Displaying the first 10 images in the data set")
display_images(x_train, y_train, num_images=10)
st.write(f"Training Samples: {x_train.shape[0]}")
st.write(f"Test Samples: {x_test.shape[0]}")
st.write(f"Image Dimensions: {x_train.shape[1]}x{x_train.shape[2]} (Grayscale)")

st.header("Data Overview")
st.write(f"Training Samples: {x_train.shape[0]}")
st.write(f"Test Samples: {x_test.shape[0]}")
st.write(f"Image Dimensions: {x_train.shape[1]}x{x_train.shape[2]} (Grayscale)")


st.header("Class Distribution")
fig, ax = plt.subplots()
sns.countplot(y_train, ax=ax, palette="viridis")
ax.set_title("Distribution of Digits in the Training Set")
st.pyplot(fig)


st.header("Sample Images from MNIST Dataset")
indices = np.random.choice(len(x_train), 5)
fig, axs = plt.subplots(1, 5, figsize=(15, 3))
for i, idx in enumerate(indices):
    axs[i].imshow(x_train[idx], cmap="gray")
    axs[i].axis('off')
    axs[i].set_title(f"Label: {y_train[idx]}")
st.pyplot(fig)
