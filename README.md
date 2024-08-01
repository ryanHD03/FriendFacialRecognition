# Friend Facial Recognition
The Friend Recognition Bot is a Discord bot that uses convolutional neural networks (CNNs) to recognize friends in images. It includes functionalities for image resizing, data augmentation, model training, and real-time image prediction within Discord.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ryanHD03/FriendFacialRecognition.git

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

## Usage

1. Start the Discord bot:
   ```bash
   python main.py
3. Use the bot in Doscrd by sending images and triggering the ".save" command to initiate the recognition process.

## Data Preparation
The "image_manipulation.py" script includes functions for resizing images and performing data augmentation. Ensure that image directories are correctly configured before running these functions.

## Training
The "CNN.py" script contains functions for building and training the CNN model. Modify the model architecture and hyperparameters as needed. Training data should be organized into appropriate directories for different classes.

## Testing
The "CNN.py" script also includes a function for testing the trained model on resized images. Ensure that test images are correctly resized and located in the specified directory.
