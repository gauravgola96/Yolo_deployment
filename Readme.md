### Deploy yolov2 on tensorflow serving
##### TF version : 1.11.0
You can change your configuration from config.py file like labels, input name , output name etc.
##### Setup Darkflow
Not in the  current repo

```
git clone https://github.com/thtrieu/darkflow.git
cd darkflow/
python3 setup.py build_ext --inplace
pip install -e .
pip install .
```
 
 
##### Step 1:
```
cd Yolo_deployment/
Change you wts/model config file path in config.py and run
python export_tfserving.py
```  
 
 
##### Step 2 : Run TF serving in a separate terminal 
```
sudo chmod +x ./run_tfserving.sh
bash ./run_tfserving.sh
```

##### Step 3
```
python start_server.py

```

##### Step4 
```
http://localhost:8000/docs
Returns a class with max prob.
```

####BLOG : Deploying Yolo on Tensorflow Serving
[ Part 1](https://medium.com/@gauravgola/deploying-yolo-on-tensorflow-serving-part-1-4586f97f0dd9)

[ Part 2](https://medium.com/@gauravgola/deploying-yolo-on-tensorflow-serving-part-2-4ecd5edbe776)
