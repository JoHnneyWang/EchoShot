This is a demo training code of EchoShot during paper submission for better assessment. Please don't distribute in any form.

# Env
```
python==3.10.12
torch==2.4.0
diffusers==0.32.1
```

# Train
We give an data example in __train.json__. After adjust the configs in __config_train.py__, run this code to start training:
```
bash train.sh
```

# Inference
We give a prompt example in __inference.json__. After adjust the configs in __config_inference.py__, run this code to start sampling:
```
bash eval.sh
```