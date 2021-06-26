from paddlelite.lite import *
import numpy as np
from PIL import Image

config = MobileConfig()
config.set_model_dir(r"D:\program\paddlepaddle\model\deploy\mobilenet_v1_opt_deploy.nb")

predictor = create_paddle_predictor(config)

image = Image.open(r"D:\program\paddlepaddle\data\test.jpg")
resize_img = image.resize((224,224),Image.BILINEAR)

image_data = np.array(resize_img).transpose(2,0,1).reshape(1,3,224,224)

input_tensor = predictor.get_input(0)
input_tensor.from_numpy(image_data)

predictor.run()

out_tensor = predictor.get_output(0)

print(out_tensor.shape())
print(out_tensor.numpy())