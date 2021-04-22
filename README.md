# Computer Vision Final Project 2021: ASL Finger Spelling to Captioning

Instructions:
To train the model with our novel architecture, run `python run.py`. 

To specify which model architecture is used when training, use the `--architecture` flag. The model can be trained using "ASL" (our model), "VGG", "AlexNet" and "LeNet". 

To load a checkpoint from a previously trained model, use the `--load-checkpoint` flag. 

To evaluate the model, use the evaluate flag and load a checkpoint. For example, run: `python run.py --load_checkpoint checkpoints/ASLModel/042121-213405/your.weights.e031-acc0.9085.h5 --evaluate`.

To run the real-time video to ASL caption componet, use the video flag and load a checkpoint. For example, run: `python run.py --load_checkpoint checkpoints/ASLModel/042121-213405/your.weights.e031-acc0.9085.h5 --video`. 

