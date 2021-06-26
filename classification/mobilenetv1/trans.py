import os
import paddlelite.lite as lite

def get_nb_model(input_path, output_path, type, platform):
    a = lite.Opt()
    a.set_model_dir(input_path)
    a.set_optimize_out(output_path)
    a.set_model_type(type)
    a.set_valid_places(platform)
    a.run()
if __name__ == '__main__':
    #deploy
    # input_path  = r"D:\program\paddlepaddle\paddlels\model\mobilenet_v1\mobilenet_v1"
    # output_path = os.getcwd()+"\mobilenet_v1_opt_deploy"
    # type = "naive_buffer"
    # platform = "x86"
    # visual
    input_path = r"D:\program\paddlepaddle\paddlels\model\mobilenet_v1\mobilenet_v1"
    output_path = os.getcwd() + "\mobilenet_v1_opt_visual"
    type = "protobuf"
    platform = "x86"
    get_nb_model(input_path, output_path, type, platform)
