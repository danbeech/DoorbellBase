# DoorbellBase

Zip everything with the exception of get_stream.py into a file

Use scp to transfer to raspi
> e.g. scp FILENAME.zip pi@raspberry.local:DIRECTORY/NESTED_DIRECTORY

Install dependencies
> No requirements.txt, my bad. Will fill in later.

Navigate to containing folder in cmd, run get_stream.py with
> python get_stream.py

SSH into raspi

Run send_stream.py with
> python3 detect_video.py --model mobilenet_ssd_v2/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite --labels mobilenet_ssd_v2/coco_labels.txt

Enjoy!