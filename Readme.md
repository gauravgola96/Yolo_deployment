### Deploy yolov2 on tensorflow serving
##### TF version : 1.11.0
You can change your configuration from config.py file like labels, input name , output name etc.
##### Setup Darkflow

```
cd ./darkflow/
python3 setup.py build_ext --inplace
pip install .
```
 
 
##### Step 1:
```
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