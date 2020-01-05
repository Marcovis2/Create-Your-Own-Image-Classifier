# PROGRAMMER: Marco Viscardi
# DATE CREATED: 28-10-2018
# REVISED DATE: -
# PURPOSE: Image Classifier with Deep Learning
# Example call:
# Main.py --dir flowers --arch densenet121 --testimage image_06514.jpg --lr 0.001 --Hidden_units 500 --Training_Epochs 3
##

# Import python modules:
import argparse
import train
import predict


def main():

    # Get command line arguments
    in_arg = get_input_args()
    print("Command Line Arguments:\n   dir=", in_arg.dir, "\n   arch=", in_arg.arch,
          "\n   testimage=", in_arg.testimage, "\n   lr=", in_arg.lr, "\n   Hidden_units=", in_arg.Hidden_units,
          "\n   Training_Epochs=", in_arg.Training_Epochs)

    train.train(in_arg.dir, in_arg.lr, in_arg.Hidden_units, in_arg.Training_Epochs, in_arg.arch, 'checkpoint.pth.tar')

    predict.predict('cat_to_name.json', 'checkpoint.pth.tar', in_arg.testimage)


def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as an
    ArgumentParser object.
    """

    # Creates Argument Parser object named parser
    parser = argparse.ArgumentParser()

    # Create 3 command line arguments:

    # Argument 1: args.dir specify path to images files
    parser.add_argument('--dir', type=str, default='flowers',
                        help='path to the folder of images')

    # Argument 2: args.arch specify which CNN model to use for classification
    parser.add_argument('--arch', type=str, default='densenet121',
                        help='chosen model')

    # Argument 3: args.labels specify the path to the text file with names of dogs
    parser.add_argument('--testimage', type=str, default='image_06514.jpg',
                        help='Test Image')

    # Argument 3: args.labels specify the path to the text file with names of dogs
    parser.add_argument('--lr', type=str, default='0.001',
                        help='Learning rate')

    # Argument 3: args.labels specify the path to the text file with names of dogs
    parser.add_argument('--Hidden_units', type=str, default='500',
                        help='hidden_units')

    # Argument 3: args.labels specify the path to the text file with names of dogs
    parser.add_argument('--Training_Epochs', type=str, default='3',
                        help='Training epochs')

    # Returns parsed argument collection
    return parser.parse_args()


# Call to main function to run the program
if __name__ == "__main__":
    main()





















