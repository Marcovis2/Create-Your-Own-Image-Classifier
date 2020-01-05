import numpy as np
import torch
import torch.nn.functional as F
import json
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd


def predict(json_path, checkpoint, image_path):
    # Label mapping
    with open(json_path, 'r') as f:
        cat_to_name = json.load(f)

    # There is a function that successfully loads a checkpoint and rebuilds the model
    def load_network(filename):
        checkpoint = torch.load(filename)
        model = checkpoint['arch']
        model.load_state_dict(checkpoint['model_state_dict'])
        model.class_to_idx = checkpoint['classes_mapping']
        criterion = checkpoint['criterion']
        lr = checkpoint['learning rate']
        optimizer = checkpoint['optimizer']
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epochs = checkpoint['training epochs']
        accuracy = checkpoint['checkpoint accuracy']

        return model, criterion, lr, optimizer, epochs, accuracy

    model, criterion, lr, optimizer, epochs, accuracy = load_network(checkpoint)

    # The process_image function successfully converts a PIL image into an object that can be used as input to a
    # trained model

    def process_image(image):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array
        '''
        size = 256, 256
        new_size = 224
        image = Image.open(image)
        image.thumbnail(size)
        image = image.crop((0, 0, new_size, new_size))
        np_image = np.array(image)
        np_image = np_image / 255
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        np_image = (np_image - mean) / std
        np_image = np_image.transpose((2, 0, 1))

        return np_image

    # To check your work, the function below (imshow) converts a PyTorch tensor and displays it in the notebook.
    # If your process_image function works, running the output through this function should return the original image
    # (except for the cropped out portions).

    def imshow(image, ax=None, title=None):
        """Imshow for Tensor."""
        if ax is None:
            fig, ax = plt.subplots()

        # PyTorch tensors assume the color channel is the first dimension
        # but matplotlib assumes is the third dimension
        image = image.transpose((1, 2, 0))

        # Undo preprocessing
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean

        # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
        image = np.clip(image, 0, 1)

        ax.imshow(image)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.tick_params(axis='both', length=0)
        ax.set_xticklabels('')
        ax.set_yticklabels('')
        plt.show()

        return ax

    im = process_image(image_path)
    imshow(im, ax=None, title=None)

    # Class Prediction

    def predict(image_path, model, topk):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
        # Now that the model is trained, we can use it for inference. We've done this before, but now we need to
        # remember to set the model in inference mode with model.eval(). You'll also want to turn off autograd with the
        # torch.no_grad() context.

        #  Test out your network!

        im = process_image(image_path)

        if torch.cuda.is_available():
            input = torch.FloatTensor(im).cuda()
        else:
            input = torch.FloatTensor(im)

        input.unsqueeze_(0)
        output = model.forward(input)
        result = F.softmax(output.data, dim=1)  # Alternative method: result = torch.exp(output)
        probs, classes = torch.topk(result, topk)
        probs = probs.data.cpu().numpy()[0]
        classes = classes.data.cpu().numpy()[0]

        predicted_classes = [classname for classname, val in model.class_to_idx.items() if val in classes]
        class_names = list(cat_to_name[key] for key in predicted_classes)

        return probs, class_names

    probs, class_names = predict(image_path, model, 5)

    # Class Prediction
    # Now that you can use a trained model for predictions, check to make sure it makes sense. Even if the testing
    # accuracy is high, it's always good to check that there aren't obvious bugs. Use matplotlib to plot the
    # probabilities for the top 5 classes as a bar graph, along with the input image.

    def sanity_check(image_path, probs, class_names, ax=None):
        img = Image.open(image_path)

        # Matplot/seaborn information
        base_color = sb.color_palette()[0]

        # create Dataframe
        data = {'probs': probs, 'classes': class_names}
        df = pd.DataFrame.from_dict(data)

        # Plot Picture & Graph
        plt.figure(figsize=[10, 5])
        plt.subplot(2, 1, 1)
        plt.imshow(img)
        plt.title(class_names[0])

        plt.subplot(2, 1, 2)
        sb.barplot(x='probs', y='classes', data=df, color=base_color)
        plt.show()

    sanity_check(image_path, probs, class_names)


