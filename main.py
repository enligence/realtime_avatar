import hydra
from omegaconf import DictConfig
from modules.video_writer import generate_video
from modules.driver_video_processor import process_driver_video

@hydra.main(config_path="conf", config_name="demo")
def avatar_gen(cfg: DictConfig):
    if cfg.driver_video_processor.enabled:
        process_driver_video(cfg.driver_video_processor)
    if cfg.avatar_image_processor.enabled:
        pass
        # process_avatar_image(cfg.avatar_image_processor)
    # so on...
    if cfg.video_writer.enabled:
        pass
        generate_video(cfg.video_writer)

def print_cfg(cfg: DictConfig):
    print(cfg.pretty())

if __name__ == "__main__":
    avatar_gen()

#python main.py --config-name your_config
