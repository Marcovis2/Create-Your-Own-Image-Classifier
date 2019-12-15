from collections import OrderedDict
import time
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms
from time import time

import torchvision.models as models
resnet50 = models.resnet50(pretrained=True)
alexnet = models.alexnet(pretrained=True)
densenet121 = models.densenet121(pretrained=True)


# All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of
# shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in to a range of
# [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].

models = {'densenet121': densenet121, 'alexnet': alexnet, 'resnet50': resnet50}


def train(image_dir, learning_rate, hidden_units, training_epochs, model_name, checkpoint):

    # Load the data
    data_dir = image_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

    # A pretrained network is loaded from torchvision.models
    model = models[model_name]

    # print(model)
    #
    #  Parameters are frozen so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # A new feedforward network is defined for use as a classifier using the features as input

    #  Hyperparameters for our network

    if model_name == 'densenet121':
        input_size = 1024
    elif model_name == 'alexnet':
        input_size = 9216
    elif model_name == 'resnet50':
        input_size = 2048

    hidden_size = int(hidden_units)
    output_size = len(train_data.class_to_idx)

    # Build a feed-forward network
    classifier = nn.Sequential(OrderedDict([
                      ('fc1', nn.Linear(input_size, hidden_size)),
                      ('relu', nn.ReLU()),
                      ('logits', nn.Linear(hidden_size, output_size)),
                      ('output', nn.LogSoftmax(dim=1))
    ]))

    if model_name == 'densenet121':
        model.classifier = classifier
    elif model_name == 'alexnet':
        model.classifier = classifier
    elif model_name == 'resnet50':
        model.fc = classifier

    model.class_to_idx = train_data.class_to_idx
    # print(model)

    #  The parameters of the feedforward classifier are appropriately trained, while the parameters of the feature
    # network are left static. During training, the validation loss and accuracy are displayed.

    def validation(model, dataloader, criterion, device='cpu'):
        test_loss = 0
        accuracy = 0
        model.to('cuda')
        for ii, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            output = model.forward(inputs)
            test_loss += criterion(output, labels).item()
            ps = torch.exp(output)
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()

        return test_loss, accuracy

    def do_deep_learning(model, trainloader, epochs, print_every, criterion, optimizer, device='cpu'):

        # Define start_time to measure total training runtime
        start_time = time()

        epochs = epochs
        print_every = print_every
        steps = 0

        # change to cuda
        model.to('cuda')

        for e in range(epochs):
            running_loss = 0
            for ii, (inputs, labels) in enumerate(trainloader):
                steps += 1

                inputs, labels = inputs.to('cuda'), labels.to('cuda')

                optimizer.zero_grad()

                # Forward and backward passes
                outputs = model.forward(inputs)  # inputs has dimensions: torch.Size([64, 3, 224, 224])
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    # Make sure network is in eval mode for inference
                    model.eval()

                    # Turn off gradients for validation, saves memory and computations
                    with torch.no_grad():
                        test_loss, accuracy = validation(model, validloader, criterion, 'gpu')

                    print("Epoch: {}/{}.. ".format(e + 1, epochs),
                          "Training Loss: {:.3f}.. ".format(running_loss / print_every),
                          "Validation Loss: {:.3f}.. ".format(test_loss / len(validloader)),
                          "Accuracy: {:.3f}".format(accuracy / len(validloader)))

                    running_loss = 0

                    # Make sure training is back on
                    model.train()

        # Prints training time in format hh:mm:ss
        end_time = time()
        tot_time = end_time - start_time
        print("\nTraining time:", str(int((tot_time / 3600))) + ":" +
              str(int(((tot_time % 3600) / 60))) + ":" + str(int(((tot_time % 3600) % 60))))

    criterion = nn.NLLLoss()
    lr = float(learning_rate)

    if model_name == 'densenet121':
        optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    elif model_name == 'alexnet':
        optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    elif model_name == 'resnet50':
        optimizer = optim.Adam(model.fc.parameters(), lr=lr)

    epochs = int(training_epochs)
    print_every = 40
    do_deep_learning(model, trainloader, epochs, print_every, criterion, optimizer, 'gpu')

    # The network's accuracy is measured on the test data:
    # Make sure network is in eval mode for inference
    model.eval()

    # Turn off gradients for validation, saves memory and computations
    with torch.no_grad():
        test_loss, accuracy = validation(model, testloader, criterion, 'gpu')
        model_accuracy = (100 * (accuracy / len(testloader)))
        print('Accuracy of the network on the test images: %d %%' % model_accuracy)

    # The trained model is saved as a checkpoint along with associated hyperparameters and the class_to_idx dictionary.
    # You probably want to save other things such as the mapping of classes to indices which you get from one of the
    # image datasets: image_datasets['train'].class_to_idx. You can attach this to the model as an attribute which
    # makes inference easier later on.

    # There is a function that successfully loads a checkpoint and accuracy
    def load_network_test(filename):
        checkpoint = torch.load(filename)
        accuracy = checkpoint['checkpoint accuracy']

        return accuracy

    accuracy = load_network_test(checkpoint)

    def save_checkpoint(state, filename='checkpoint.pth.tar'):
        torch.save(state, filename)

    if model_accuracy > accuracy:
        save_checkpoint({
            'classes_mapping': train_data.class_to_idx,
            'arch': model,
            'model_state_dict': model.state_dict(),
            'criterion': criterion,
            'learning rate': lr,
            'optimizer': optimizer,
            'optimizer_state_dict': optimizer.state_dict(),
            'training epochs': epochs,
            'checkpoint accuracy': model_accuracy},
            filename='checkpoint.pth.tar')

        print("This new model is better than your previously saved one. New model saved!")

    else:
        print("Your previously saved model is better!")











