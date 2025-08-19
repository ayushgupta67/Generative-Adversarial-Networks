# ⭐ Generative Adversarial Network (GAN) for CIFAR-10 ⭐

This project implements a Generative Adversarial Network (GAN) using the Keras library to generate new images based on the CIFAR-10 dataset.

# <u>📌 The Models</u>

🕵️‍♂️ Discriminator – A CNN that classifies images as real (from CIFAR-10) or fake (from the Generator).

🎨 Generator – Takes a random noise vector as input and outputs a new image, trying to fool the Discriminator.

🤝 GAN Model – Combines both. The Generator learns by trying to make the Discriminator misclassify its output as real.

# <u>🔎 How It Works</u>

The training process follows a repeating loop:

🔄 Generate Samples

Load a batch of real CIFAR-10 images.

Generate an equal-sized batch of fake images using the Generator.

# 🎯 Train Discriminator

Train on real images (labeled 1) and fake images (labeled 0).

Learns to distinguish real vs fake.

# 🧠 Train Generator (via GAN)

Freeze Discriminator weights.

Train the Generator to make the Discriminator classify fake images as real (1).

Updates only Generator weights → improves image quality.

# <u>🚀 Instructions (Google Colab Setup)</u>
🔧 Prerequisites

Install required packages:

!pip install tensorflow keras matplotlib numpy

📂 (Optional) Mount Google Drive

To save generated images permanently:

from google.colab import drive
drive.mount('/content/drive')

# 🏗️ Workflow

Load the CIFAR-10 dataset.

Define the Discriminator model.

Define the Generator model.

Define the GAN (combined model).

Train using the train() function.

Training logs (loss values) will be printed.

<u>💾 Save Final Images</u>

After training, you can generate and save final results to Google Drive.

# Create output folder
import os
output_dir = '/content/drive/My Drive/GAN_Output'
os.makedirs(output_dir, exist_ok=True)

# Generate final images
n_samples = 49
X_final, _ = generate_fake_samples(g_model, latent_dim, n_samples)

# Save plot
from matplotlib import pyplot
def save_final_plot(examples, n=7):
    examples = (examples + 1) / 2.0  # Rescale from [-1,1] to [0,1]
    for i in range(n * n):
        pyplot.subplot(n, n, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(examples[i])

    filename = os.path.join(output_dir, 'final_generated_images.png')
    pyplot.savefig(filename)
    pyplot.close()

save_final_plot(X_final)
print("✅ Final generated images saved to Google Drive.")
