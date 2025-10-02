from roboflow import Roboflow
rf = Roboflow(api_key="Bss7EdeWmd2GLQ1b03sf")
project = rf.workspace("humanoidrobot-tdqqz").project("yolov8-humanoid-robot-y0xqf")
version = project.version(2)
dataset = version.download("yolov8-obb")
                