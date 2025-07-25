
```bash
pt install stai-mpu-tflite
python3 -m venv --system-site-packages ~/.venv-st
source ~/.venv-st
sed -i '/silence-tensorflow/d' requirements.txt
pip install -r requirements.txt
pip install --upgrade pillow

f=custom-mux-pc-u8f32; stedgeai generate -m $f.tflite --target stm32mp25 && mv stm32ai_output/$f.nb . && scp ../models/$f.nb    root@192.168.38.141:models/

python3 train.py --run_mp2 True

v4l2-ctl -d  /dev/video7 --list-formats-ext

# to capture:
v4l2-ctl -d /dev/video7 --set-fmt-video=width=640,height=480,pixelformat=MJPG
v4l2-ctl -d /dev/video7 --stream-mmap --stream-count=1 --stream-to=output.jpg
```
