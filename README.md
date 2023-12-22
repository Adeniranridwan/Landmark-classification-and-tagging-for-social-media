**Problem Statement**
Photo sharing and photo storage services like to have location data for each photo that is uploaded. With the location data, these services can build advanced features, such as automatic suggestion of relevant tags or automatic photo organization, which help provide a compelling user experience. Although a photo's location can often be obtained by looking at the photo's metadata, many photos uploaded to these services will not have location metadata available. This can happen when, for example, the camera capturing the picture does not have GPS or if a photo's metadata is scrubbed due to privacy concerns.
If no location metadata for an image is available, one way to infer the location is to detect and classify a discernible landmark in the image. Given the large number of landmarks across the world and the immense volume of images that are uploaded to photo sharing services, using human judgment to classify these landmarks would not be feasible.
In this project, the problem was first addressed first by building models to automatically predict the location of the image based on any landmarks depicted in the image.
**Step**
-The image landmark dataset was assessed, visualized and processed for training. 
-Created a convolutional neural network from scratch to Classify Landmarks (from Scratch) 
-Created a CNN to Classify Landmarks (using Transfer Learning) – 
-Different pre-trained models was investigated to select the best suitable for the classification task. 
-The vgg-16 was eventally selected which was trained and tested to 70% accuracy.
-The transfer learning model was exported using Torch Script
-The algorithm was deployed in an app for user  

