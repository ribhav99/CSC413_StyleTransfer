To install all dependencies, use 'pip3 install -r requirements.txt'
Note: This has been tested to work on python3.7 and 3.8.

All the code is under scripts.
The file main.py creates, trains and tests models. 
The Attrdict inside main.py can be changed to change the architecture and hyperparameters of the model.
Everything is modular.
The architecture of the models is printed everytime you train or test a model.

When a model trains, it saves a version of itself every 10 epochs. It also keeps tracks of the losses in a text file.

Datasets used in our experiments, and extra datasets not presented in our report can be found in the data folder.

Everything here, except the .ipynb file, is code for the Cycle-GAN. The .ipynb file is to run the pix2pix model.


Here are the architecture and paraterms that can be changed from main.py:

'dis_learning_rate': discriminator learning rate,
'gen_learning_rate': generator learning rate,
'batch_size': batch size,
'num_epochs': total number of epochs to train for,
'starting_epoch': what epoch to start from. Useful when loading a model to train further,
'channel_list': array representing the number of channels each layer of the network extracts. Larger the array, larger the model,
'image_dim': dimensions of the input and output image,
'kernel': kernel size,
'stride': stride,
'x': path for data from domain X,
'y': path of data from domain Y,
'save_path': where the models and loss files will be saved,
'act_fn_gen': activation function for generator,
'act_fn_dis': activation function for discriminator,
'norm_type': the type of norm that will be used. eg: batch, instance,
'num_res': number of residual layers. Look at architecture for more details,
'lambda_cycle': value to weight importance of domain shift during cycle consistency,
'gray': boolean: grayscale image or coloured,
'conv2T': boolean: use transpose convolution for upsampling,
'buffer_train': create a buffer to stabilise gan training,
'decay': decay learning rate,
'train': train or test,
'load_models': load a previously trained model,
'model_path': path of the model to load,
'save_epoch': save the model every x epochs
