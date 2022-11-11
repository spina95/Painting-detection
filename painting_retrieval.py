# We can use a pretrained MobileNetv2 on ImageNet to get the feature of an image
# Do we need to fine tune the model?
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import cv2
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
# Methods usable for this task are ORB, KAZE, pretrained CNN in particular ResNet18

from PIL import Image

class PaintingFinder:

    resnet18Model = None
    resnet18ModelLayer = None

    def __init__(self):
        self.resnet18Model = models.resnet18(pretrained=True)
        self.resnet18ModelLayer = self.resnet18Model._modules.get('avgpool')
        self.resnet18Model.eval()

    def getRankedListOfSimilars(self, inputImages, paintingsDB, method=1):
        results = []
        for inputImage in inputImages:
            if method == 0:
                results.append(self.useResNet18Approach(inputImage, paintingsDB))
            elif method == 1:
                results.append(self.useORBApproach(inputImage, paintingsDB))
            elif method == 2:
                results.append(self.useAKAZEApproach(inputImage, paintingsDB))
        return results

    def compareTwoImages(self, imageOne, imageTwo, method=0):
        if method == 0:
            return self.useAKAZEApproachSingleImage(imageOne, imageTwo)
        elif method == 1:
            return self.useORBApproachSingleImage(imageOne, imageTwo)

    def useAKAZEApproachSingleImage(self, inputImage, secondImage):
        # Load the image convert to RGB and get gray version
        #inputImage = cv2.imread(inputImageFilename)
        inputImgRGB = cv2.cvtColor(inputImage, cv2.COLOR_BGR2RGB)
        inputImgGray = cv2.cvtColor(inputImgRGB, cv2.COLOR_RGB2GRAY)
        akaze = cv2.AKAZE_create()
        inputImageKeypoints, inputImageDescriptor = akaze.detectAndCompute(inputImgGray, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        inputImgRGB1 = cv2.cvtColor(secondImage, cv2.COLOR_BGR2RGB)
        inputImgGray1 = cv2.cvtColor(inputImgRGB1, cv2.COLOR_RGB2GRAY)
        tmpImageKeypoints, tmpImageDescriptor = akaze.detectAndCompute(inputImgGray1, None)

        # Perform the matching between the ORB descriptors of the input image and the testing image
        nn_matches = bf.knnMatch(inputImageDescriptor, tmpImageDescriptor,k=2)

        good = []
        nn_match_ratio = 0.8  # Nearest neighbor matching ratio
        for m, n in nn_matches:
            if m.distance < nn_match_ratio * n.distance:
                good.append([m])

        return len(good)

    def useORBApproachSingleImage(self, inputImage, secondImage):
        # Load the image convert to RGB and get gray version
        #inputImage = cv2.imread(inputImageFilename)
        inputImgRGB = cv2.cvtColor(inputImage, cv2.COLOR_BGR2RGB)
        inputImgGray = cv2.cvtColor(inputImgRGB, cv2.COLOR_RGB2GRAY)
        orb = cv2.ORB_create()
        inputImageKeypoints, inputImageDescriptor = orb.detectAndCompute(inputImgGray, None)
        # Create a Brute Force Matcher object.
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        inputImgRGB1 = cv2.cvtColor(secondImage, cv2.COLOR_BGR2RGB)
        inputImgGray1 = cv2.cvtColor(inputImgRGB1, cv2.COLOR_RGB2GRAY)
        # if os.path.join(DBPath, imagefile) == inputImageFilename:
        # continue
        # 1. Load the image convert to RGB and get gray version
        tmpImageKeypoints, tmpImageDescriptor = orb.detectAndCompute(inputImgGray1, None)

        # Perform the matching between the ORB descriptors of the input image and the testing image
        matches = bf.knnMatch(inputImageDescriptor, tmpImageDescriptor, k=2)
        good = []
        nn_match_ratio = 0.8  # Nearest neighbor matching ratio
        for m, n in matches:
            if m.distance < nn_match_ratio * n.distance:
                good.append([m])
        # The matches with shorter distance are the ones we want.
        #matches = sorted(matches, key=lambda x: x.distance)

        # Compute the cosine similarity
        #cos_sim = cosine_similarity(inputTensor.reshape((1, -1)), tmpFeaturesVector.reshape((1, -1)))[0][0]
        #scoresDictionary[imagefile] = cos_sim
        return len(good)


    def useAKAZEApproach(self, inputImage, paintingDB):
        # Load the image convert to RGB and get gray version
        #inputImage = cv2.imread(inputImageFilename)
        inputImgRGB = cv2.cvtColor(inputImage, cv2.COLOR_BGR2RGB)
        inputImgGray = cv2.cvtColor(inputImgRGB, cv2.COLOR_RGB2GRAY)
        akaze = cv2.AKAZE_create()
        inputImageKeypoints, inputImageDescriptor = akaze.detectAndCompute(inputImgGray, None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        scoresDictionary = {}
        for imagefile, painting in paintingDB.items():
            tmpImgGray = painting[2]
            tmpImageKeypoints, tmpImageDescriptor = akaze.detectAndCompute(tmpImgGray, None)

            # Perform the matching between the ORB descriptors of the input image and the testing image
            nn_matches = bf.knnMatch(inputImageDescriptor, tmpImageDescriptor,k=2)

            good = []
            nn_match_ratio = 0.8  # Nearest neighbor matching ratio
            for m, n in nn_matches:
                if m.distance < nn_match_ratio * n.distance:
                    good.append([m])

            scoresDictionary[imagefile] = len(good)
        scoresDictionary = {k: v for k, v in sorted(scoresDictionary.items(), key=lambda item: item[1], reverse=True)}
        for k, v in scoresDictionary.items():
            print("IMAGE: %s - SCORE: %f\n" % (paintingDB[k][3], v))
        return scoresDictionary

    def useORBApproach(self, inputImage, paintingDB):
        # Load the image convert to RGB and get gray version
        #inputImage = cv2.imread(inputImageFilename)
        inputImgRGB = cv2.cvtColor(inputImage, cv2.COLOR_BGR2RGB)
        inputImgGray = cv2.cvtColor(inputImgRGB, cv2.COLOR_RGB2GRAY)
        orb = cv2.ORB_create()
        inputImageKeypoints, inputImageDescriptor = orb.detectAndCompute(inputImgGray, None)
        # Create a Brute Force Matcher object.
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        scoresDictionary = {}
        for imagefile, painting in paintingDB.items():
            # if os.path.join(DBPath, imagefile) == inputImageFilename:
            # continue
            # 1. Load the image convert to RGB and get gray version
            tmpImgGray = painting[2]
            tmpImageKeypoints, tmpImageDescriptor = orb.detectAndCompute(tmpImgGray, None)

            # Perform the matching between the ORB descriptors of the input image and the testing image
            matches = bf.knnMatch(inputImageDescriptor, tmpImageDescriptor, k=2)
            good = []
            nn_match_ratio = 0.8  # Nearest neighbor matching ratio
            for m, n in matches:
                if m.distance < nn_match_ratio * n.distance:
                    good.append([m])
            # The matches with shorter distance are the ones we want.
            #matches = sorted(matches, key=lambda x: x.distance)

            # Compute the cosine similarity
            #cos_sim = cosine_similarity(inputTensor.reshape((1, -1)), tmpFeaturesVector.reshape((1, -1)))[0][0]
            #scoresDictionary[imagefile] = cos_sim
            scoresDictionary[imagefile] = len(good)
        scoresDictionary = {k: v for k, v in sorted(scoresDictionary.items(), key=lambda item: item[1], reverse=True)}
        for k, v in scoresDictionary.items():
            print("PAINTING: %s - SCORE: %f\n" % (paintingDB[k][3], v))
        return scoresDictionary


    def useResNet18Approach(self, inputImage, paintingDB):
        # Load a pretrained resnet18 model
        #model = models.resnet18(pretrained=True)
        # Use the features layer of the model
        #layer = model._modules.get('avgpool')
        # Set the model to evaluation mode
        #model.eval()
        inputTensor = self.getFeaturesVectorResNet18(inputImage)
        scoresDictionary = {}
        for imagefile, painting in paintingDB.items():
            #if os.path.join(DBPath, imagefile) == inputImageFilename:
                #continue
            # Create the features vector for the input image
            tmpFeaturesVector = self.getFeaturesVectorResNet18(painting[1])
            # Compute the cosine similarity
            #cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            #cos_sim = cos(inputTensor.unsqueeze(0), tmpFeaturesVector.unsqueeze(0))
            cos_sim = cosine_similarity(inputTensor.reshape((1, -1)), tmpFeaturesVector.reshape((1, -1)))[0][0]

            scoresDictionary[imagefile] = cos_sim
        scoresDictionary = {k: v for k, v in sorted(scoresDictionary.items(), key=lambda item: item[1], reverse=True)}
        for k,v in scoresDictionary.items():
            print("PAINTING: %s - SCORE: %f\n" % (paintingDB[k][3], v))
        return scoresDictionary


    def getFeaturesVectorResNet18(self, inputImage):
        # 1. Load the image with Pillow library
        # MOD: inputImage is given in opencv format so it must be converted to PIL Image
        #img = Image.open(inputImageFilename)
        img = cv2.cvtColor(inputImage, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(inputImage)

        # 2. Create a PyTorch Variable with the transformed image
        preprocess = transforms.Compose([
            transforms.Resize((244, 244)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        t_img = Variable(preprocess(img).unsqueeze(0))
        # Note: .unsqueeze(0) is neeeded since PyTorch expects a 4-dimensional input, the first dimension being the number of samples
        # 3. Create a vector of zeros that will hold our feature vector
        #    The 'avgpool' layer has an output size of 512
        featuresVector = torch.zeros(1, 512, 1, 1)
        # 4. Define a function that will copy the output of a layer
        def copy_data(m, i, o):
            featuresVector.copy_(o.data)
        # 5. Attach that function to our selected layer
        h = self.resnet18ModelLayer.register_forward_hook(copy_data)
        # 6. Run the model on our transformed image
        self.resnet18Model(t_img)
        # 7. Detach our copy function from the layer
        h.remove()
        # 8. Return the feature vector
        return featuresVector