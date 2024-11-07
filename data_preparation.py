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
    fig, axes = plt.subplots(1, num_images, figsize=(10, 3))
    for i in range(num_images):
        axes[i].imshow(images[i], cmap='gray')
        axes[i].set_title(f"Label: {labels[i]}")
        axes[i].axis('off')
    st.pyplot(fig)
    plt.clf()

# Data Overview
st.header("Data Overview")
st.write("Displaying the first 10 images in the data set")
display_images(x_train, y_train, num_images=10)
st.write(f"Training Samples: {x_train.shape[0]}")
st.write(f"Test Samples: {x_test.shape[0]}")
st.write(f"Image Dimensions: {x_train.shape[1]}x{x_train.shape[2]} (Grayscale)")

# Class Distribution
st.header("Class Distribution")
fig, ax = plt.subplots()
sns.countplot(x=y_train, ax=ax, palette="viridis")
ax.set_title("Distribution of Digits in the Training Set")
st.pyplot(fig)
plt.clf()

# Sample Images from MNIST Dataset
st.header("Sample Images from MNIST Dataset")
indices = np.random.choice(len(x_train), 5)
fig, axs = plt.subplots(1, 5, figsize=(15, 3))
for i, idx in enumerate(indices):
    axs[i].imshow(x_train[idx], cmap="gray")
    axs[i].axis('off')
    axs[i].set_title(f"Label: {y_train[idx]}")
st.pyplot(fig)
plt.clf()

# Mean and Standard Deviation of Pixel Values
st.header("Mean and Standard Deviation of Pixel Values")
mean_pixel = np.mean(x_train)
std_pixel = np.std(x_train)
st.write(f"Mean Pixel Value: {mean_pixel:.2f}")
st.write(f"Standard Deviation of Pixel Value: {std_pixel:.2f}")

# Downsampled pixel values for histogram
flattened_pixels = x_train.flatten()
sampled_pixels = np.random.choice(flattened_pixels, size=5000, replace=False)  # Sample 5000 pixels for plotting

fig, ax = plt.subplots()
sns.histplot(sampled_pixels, kde=True, color='blue')
ax.set_title("Distribution of Pixel Values (Sampled)")
st.pyplot(fig)
plt.clf()

# Average Pixel Intensity Heatmap
st.header("Average Pixel Intensity Heatmap")
avg_digit = np.mean(x_train, axis=0)
fig, ax = plt.subplots()
sns.heatmap(avg_digit, cmap='coolwarm', ax=ax)
ax.set_title("Average Digit in the Training Set")
st.pyplot(fig)
plt.clf()

# Pixel Value Histogram for a Selected Digit
st.header("Pixel Value Histogram for a Selected Digit")
digit_choice = st.selectbox("Select a digit to explore:", np.arange(10))
digit_indices = np.where(y_train == digit_choice)[0]
random_digit = np.random.choice(digit_indices)
fig, ax = plt.subplots()
sns.histplot(x_train[random_digit].flatten(), kde=True, ax=ax)
ax.set_title(f"Pixel Value Distribution for Digit {digit_choice}")
st.pyplot(fig)
plt.clf()

# Image Normalization
st.header("Image Normalization")
st.write("Below is an example of a digit before and after normalization (scaling pixel values between 0 and 1).")

fig, axs = plt.subplots(1, 2, figsize=(10, 3))
random_idx = np.random.randint(0, len(x_train))

# Original image
axs[0].imshow(x_train[random_idx], cmap="gray")
axs[0].axis('off')
axs[0].set_title("Original Image")

# Normalized image
norm_img = x_train[random_idx] / 255.0
axs[1].imshow(norm_img, cmap="gray")
axs[1].axis('off')
axs[1].set_title("Normalized Image")

st.pyplot(fig)
plt.clf()

# Correlation Between Pixel Positions
st.header("Correlation Between Pixel Positions")
# Downsample the data for quicker computation
reshaped_data = x_train.reshape(x_train.shape[0], -1)[:, :500]  # Limit to 500 pixels for correlation
pixel_corr = np.corrcoef(reshaped_data, rowvar=False)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(pixel_corr, cmap="coolwarm", ax=ax)
ax.set_title("Pixel Correlation Heatmap")
st.pyplot(fig)
plt.clf()