To install all dependencies, use 'pip3 install -r requirements.txt'
Note: This has been tested to work on python3.7 and 3.8.

To test out our training loop please go to the scripts folder and run 
python3.7 main.py -[Conv2T][RegConv][VAE][Gray] -train

#this runs our training loop on the images in trainHuman and trainCartoon for 10epochs, so it won't take long
#when running the train loop, it will also show the model architecture we implemented 

To evaluate our trained model go to the scripts folder and run
python3.7 main.py -[Conv2T][RegConv][VAE][Gray]

#this will generate images in the cooresponding folder as well as plotting the model visualizations
#note that for args.buffer_train is not really meant to work outside of google colab but we kept it in anyways for submission, to use buffer please set the batch size to 10
